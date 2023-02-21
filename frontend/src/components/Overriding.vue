<template>
<div>
    <section class="whole">
      <div id="horizontalBar" class="left"></div>
      <div id="histogram" class="right"></div>
    </section>
</div>
</template>

<script>
import { EventBus } from '../main.js'
import * as Plotly from 'plotly.js'
import * as d3Base from 'd3'
// attach all d3 plugins to the d3 library
const d3v5 = Object.assign(d3Base)

export default {
  name: 'Overriding',
  data () {
    return {
      problematic: 0,
      surrogateData: 0,
      checkingRule: -1,
      flagClicked: true
    }
  },
  methods: {
        fisheyeScale(scaleType) {
            return this.d3_fisheye_scale(scaleType(), 3, 0);
        },
        fisheyeOrdinal(focusInit) {
            return this.d3_fisheye_scale_ordinal(d3.scale.ordinal(), 3, 0)
        },
        fisheyeCircular() {
            var radius = 200,
                distortion = 2,
                k0,
                k1,
                focus = [0, 0];

            function fisheye(d) {
                var dx = d.x - focus[0],
                    dy = d.y - focus[1],
                    dd = Math.sqrt(dx * dx + dy * dy);
                if (!dd || dd >= radius) return {x: d.x, y: d.y, z: 1};
                var k = k0 * (1 - Math.exp(-dd * k1)) / dd * .75 + .25;
                return {x: focus[0] + dx * k, y: focus[1] + dy * k, z: Math.min(k, 10)};
            }

            function rescale() {
                k0 = Math.exp(distortion);
                k0 = k0 / (k0 - 1) * radius;
                k1 = distortion / radius;
                return fisheye;
            }

            fisheye.radius = function(_) {
                if (!arguments.length) return radius;
                radius = +_;
                return rescale();
            };

            fisheye.distortion = function(_) {
                if (!arguments.length) return distortion;
                distortion = +_;
                return rescale();
            };

            fisheye.focus = function(_) {
                if (!arguments.length) return focus;
                focus = _;
                return fisheye;
            };

            return rescale();
        },
      d3_fisheye_scale(scale, d, a) {

          function fisheye(_) {
              var x = scale(_),
                  left = x < a,
                  range = d3.extent(scale.range()),
                  min = range[0],
                  max = range[1],
                  m = left ? a - min : max - a;
              if (m == 0) m = max - min;
              return (left ? -1 : 1) * m * (d + 1) / (d + (m / Math.abs(x - a))) + a;
          }

          fisheye.distortion = function (_) {
              if (!arguments.length) return d;
              d = +_;
              return fisheye;
          };

          fisheye.focus = function (_) {
              if (!arguments.length) return a;
              a = +_;
              return fisheye;
          };

          fisheye.copy = function () {
              return d3_fisheye_scale(scale.copy(), d, a);
          };

          fisheye.nice = scale.nice;
          fisheye.ticks = scale.ticks;
          fisheye.tickFormat = scale.tickFormat;
          return d3.rebind(fisheye, scale, "domain", "range");
        },
        d3_fisheye_scale_ordinal(scale, d, a) {

            function scale_factor(x) {
                var 
                    left = x < a,
                    range = scale.rangeExtent(),
                    min = range[0],
                    max = range[1],
                    m = left ? a - min : max - a;

                if (m == 0) m = max - min;
                var factor = (left ? -1 : 1) * m * (d + 1) / (d + (m / Math.abs(x - a)));
                return factor + a;
            };

            function fisheye(_) {
                return scale_factor(scale(_));
            };

            fisheye.distortion = function (_) {
                if (!arguments.length) return d;
                d = +_;
                return fisheye;
            };

            fisheye.focus = function (_) {
                if (!arguments.length) return a;
                a = +_;
                return fisheye;
            };

            fisheye.copy = function () {
                return d3_fisheye_scale_ordinal(scale.copy(), d, a);
            };

            fisheye.rangeBand = function (_) {
                var band = scale.rangeBand(),
                    x = scale(_),
                    x1 = scale_factor(x),
                    x2 = scale_factor(x + band);

                return Math.abs(x2 - x1);
            };

          
            fisheye.rangeRoundBands = function (x, padding, outerPadding) {
                var roundBands = arguments.length === 3 ? scale.rangeRoundBands(x, padding, outerPadding) : arguments.length === 2 ? scale.rangeRoundBands(x, padding) : scale.rangeRoundBands(x);
                fisheye.padding = padding * scale.rangeBand();
                fisheye.outerPadding = outerPadding;
                return fisheye;
            };

            return d3.rebind(fisheye, scale, "domain",  "rangeExtent", "range");
        },
    problemIdentifier () {

      Plotly.purge('horizontalBar')

      var gridPosRuleOver = this.problematic[34]
      var purityListRuleOver = this.problematic[35]
      var probamulalphaListRuleOver = this.problematic[36]

      var statistics = JSON.parse(JSON.parse(this.surrogateData[6]))
      var probamulalpha = Object.values(statistics['probamulalpha'])
      var maximumProbaMul = Math.max(...probamulalpha)
      var minimumProbaMul = Math.min(...probamulalpha)

      if (this.flagClicked) {
        this.checkingRule = gridPosRuleOver[gridPosRuleOver.length-1]
      }

      EventBus.$emit('sendWhichRuleIsActive', this.checkingRule-1)

      var width = 208
      var height = 230

      var colorsDiverging = ["#002051","#002153","#002255","#002356","#002358","#002459","#00255a","#00255c","#00265d","#00275e","#00275f","#002860","#002961","#002962","#002a63","#002b64","#012b65","#022c65","#032d66","#042d67","#052e67","#052f68","#063069","#073069","#08316a","#09326a","#0b326a","#0c336b","#0d346b","#0e346b","#0f356c","#10366c","#12376c","#13376d","#14386d","#15396d","#17396d","#183a6d","#193b6d","#1a3b6d","#1c3c6e","#1d3d6e","#1e3e6e","#203e6e","#213f6e","#23406e","#24406e","#25416e","#27426e","#28436e","#29436e","#2b446e","#2c456e","#2e456e","#2f466e","#30476e","#32486e","#33486e","#34496e","#364a6e","#374a6e","#394b6e","#3a4c6e","#3b4d6e","#3d4d6e","#3e4e6e","#3f4f6e","#414f6e","#42506e","#43516d","#44526d","#46526d","#47536d","#48546d","#4a546d","#4b556d","#4c566d","#4d576d","#4e576e","#50586e","#51596e","#52596e","#535a6e","#545b6e","#565c6e","#575c6e","#585d6e","#595e6e","#5a5e6e","#5b5f6e","#5c606e","#5d616e","#5e616e","#60626e","#61636f","#62646f","#63646f","#64656f","#65666f","#66666f","#67676f","#686870","#696970","#6a6970","#6b6a70","#6c6b70","#6d6c70","#6d6c71","#6e6d71","#6f6e71","#706f71","#716f71","#727071","#737172","#747172","#757272","#767372","#767472","#777473","#787573","#797673","#7a7773","#7b7774","#7b7874","#7c7974","#7d7a74","#7e7a74","#7f7b75","#807c75","#807d75","#817d75","#827e75","#837f76","#848076","#858076","#858176","#868276","#878376","#888477","#898477","#898577","#8a8677","#8b8777","#8c8777","#8d8877","#8e8978","#8e8a78","#8f8a78","#908b78","#918c78","#928d78","#938e78","#938e78","#948f78","#959078","#969178","#979278","#989278","#999378","#9a9478","#9b9578","#9b9678","#9c9678","#9d9778","#9e9878","#9f9978","#a09a78","#a19a78","#a29b78","#a39c78","#a49d78","#a59e77","#a69e77","#a79f77","#a8a077","#a9a177","#aaa276","#aba376","#aca376","#ada476","#aea575","#afa675","#b0a775","#b2a874","#b3a874","#b4a974","#b5aa73","#b6ab73","#b7ac72","#b8ad72","#baae72","#bbae71","#bcaf71","#bdb070","#beb170","#bfb26f","#c1b36f","#c2b46e","#c3b56d","#c4b56d","#c5b66c","#c7b76c","#c8b86b","#c9b96a","#caba6a","#ccbb69","#cdbc68","#cebc68","#cfbd67","#d1be66","#d2bf66","#d3c065","#d4c164","#d6c263","#d7c363","#d8c462","#d9c561","#dbc660","#dcc660","#ddc75f","#dec85e","#e0c95d","#e1ca5c","#e2cb5c","#e3cc5b","#e4cd5a","#e6ce59","#e7cf58","#e8d058","#e9d157","#ead256","#ebd355","#ecd454","#edd453","#eed553","#f0d652","#f1d751","#f1d850","#f2d950","#f3da4f","#f4db4e","#f5dc4d","#f6dd4d","#f7de4c","#f8df4b","#f8e04b","#f9e14a","#fae249","#fae349","#fbe448","#fbe548","#fce647","#fce746","#fde846","#fde946","#fdea45"]

      var colorsScaleProbaMul = d3v5.scaleQuantize()
          .domain([minimumProbaMul, maximumProbaMul])
          .range(colorsDiverging);

      var colors = []
      var colorsProbaMul = []
      var currentIndex
      for (let index = 0; index < gridPosRuleOver.length; index++) {
        if (gridPosRuleOver[index] == this.checkingRule) {
          currentIndex = index
          colors.push('rgb(227,26,28)')
          colorsProbaMul.push(colorsScaleProbaMul(probamulalphaListRuleOver[index]))
        } else {
          colors.push('rgb(0,0,255)')
          colorsProbaMul.push(colorsScaleProbaMul(probamulalphaListRuleOver[index]))
        }
      }
      
      var trace1 = {
        x: purityListRuleOver,
        y: gridPosRuleOver,
        orientation: 'h',
        name:'Impurity',
        marker: {
          color: colors,
          width: 1
        },
        type: 'bar'
      };

      var trace2 = {
        x: probamulalphaListRuleOver,
        y: gridPosRuleOver,
        orientation: 'h',
        type: 'bar',
        showlegend: false,
        name:'WxP',
        marker: {
          color: colorsProbaMul,
          width: 1
        }
      };

      var trace3 = {
        x: [0],
        y: [1],
        orientation: 'h',
        type: 'bar',
        hoverinfo: 'skip',
        name:'Selection',
        marker: {
          color: 'rgb(227,26,28)',
          width: 0
        }
      };

      var data = [trace1, trace2, trace3];

      var layout = {
        yaxis: {
          yanchor: 'right',
          showticklabels: true,
          type: 'category',
          dtick: 1
        },
        // xaxis: {
        //   visible: false
        // },
        width: width,
        height: height,
        showlegend: true,
        legend: {"orientation": "h", x: -0.2, y:1.15},
        barmode: 'relative',
        margin: {
          l: 15,
          r: 9,
          b: 15,
          t: 0,
          pad: 0
        },
      };

      const config = {
        displayModeBar: false, // this is the line that hides the bar.
      };

      var myPlot = document.getElementById('horizontalBar')

      Plotly.newPlot(myPlot, data, layout, config);

      myPlot.on('plotly_click', function(data){
        EventBus.$emit('clickedANewImpureRuleFlag')
        EventBus.$emit('clickedANewImpureRule', parseFloat(data.points[0].label))
        EventBus.$emit('SendHighlightingZero') 
        EventBus.$emit('updateFirstTimeGL') 
        EventBus.$emit('hoveringTestSample', -3)        
      });

      var svg = d3.select("#histogram");
      svg.selectAll("*").remove();

      var w = 615,
      h = 230,
      p = [5, 25, 20, 5],
        
      //fisheye distortion scale
      x = this.fisheyeOrdinal().rangeRoundBands([0, w - p[1] - p[3]]).distortion(0),

      y = d3.scale.linear().range([0, h - p[0] - p[2]]),
      z = d3.scale.ordinal().range(["#a6dba0", "#c2a5cf"])
      
      var valuesHist0 = this.problematic[38]
      var valuesHist1 = this.problematic[39]
      var threshold = this.problematic[40]
      var bins = this.problematic[41]
      var limiter = threshold[currentIndex] * (w - p[1] - p[3])

      var svg = d3.select("#histogram").append("svg:svg")
      .attr("width", w)
      .attr("height", h)
      .append("svg:g")
      .attr("transform", "translate(" + p[3] + "," + (h - p[2]) + ")");

      var crimea = []
      for (let index = 0; index < bins.length; index++) {
        var tempDic = { intervals: (bins[index]), firstClass: valuesHist0[currentIndex][index], secondClass: valuesHist1[currentIndex][index] }
        crimea.push(tempDic)
      }
      
      // Transpose the data into layers by cause.
      var causes = d3.layout.stack()(["firstClass", "secondClass"].map(function(cause) {
      return crimea.map(function(d) {
      return {x: d.intervals, y: +d[cause]};
      });
      }));
      
      // Compute the x-domain (by date) and y-domain (by top).
      x.domain(causes[0].map(function(d) { return d.x; }));
      y.domain([0, d3.max(causes[causes.length - 1], function(d) { return d.y0 + d.y; })]);
      
      // Add a group for each cause.
      var cause = svg.selectAll("g.cause")
      .data(causes)
      .enter().append("svg:g")
      .attr("class", "cause")
      .style("fill", function(d, i) { return z(i); })
      .style("stroke", function(d, i) { return d3.rgb(z(i)).darker(); });
      
      // Add a rect for each date.
      var rect = cause.selectAll("rect")
      .data(Object)
      .enter().append("svg:rect")
      .attr("x", function(d) { return x(d.x); })
      .attr("y", function(d) { return -y(d.y0) - y(d.y); })
      .attr("height", function(d) { return y(d.y); })
      .attr("width", function(d) {return x.rangeBand(d.x);});
      
      // Add a label per date.
      if (bins.length == 20) {
        var label = svg.selectAll("text")
        .data(x.domain())
        .enter().append("svg:text")
        .attr("x", function(d) { return x(d) + x.rangeBand(d) / 2; })
        .attr("y", 6)
        .attr("text-anchor", "middle")
        .attr("dy", ".71em")
        // .style("font-size", "10px")
        .text(function(d, i) { 
          return d;
        });
      } else {
        var label = svg.selectAll("text")
        .data(x.domain())
        .enter().append("svg:text")
        .attr("x", function(d) { return x(d) + x.rangeBand(d) / 2; })
        .attr("y", 6)
        .attr("text-anchor", "middle")
        .attr("dy", ".71em")
        // .style("font-size", "10px")
        .text(function(d, i) { 
          if ((i%2) != 0) { 
            return d;
          }
          else {
            return '';
          }
        });
      }

      
      // Add y-axis rules.
      var rule = svg.selectAll("g.rule")
      .data(y.ticks(6))
      .enter().append("svg:g")
      .attr("class", "rule")
      .attr("transform", function(d) { return "translate(0," + -y(d) + ")"; });
      
      rule.append("svg:line")
      .attr("x2", w - p[1] - p[3])
      .style("stroke", function(d) { return d ? "#D3D3D3" : "#D3D3D3"; })
      .style("stroke-opacity", function(d) { return d ? .3 : null; });
      
      rule.append("svg:text")
      .attr("x", w - p[1] - p[3] + 6)
      .attr("dy", ".35em")
      .text(d3.format(",d"));

      x.focus(limiter);

      //redraw the bars
      rect
      .attr("x", function(d) { return x(d.x); })
      .attr("y", function(d) { return -y(d.y0) - y(d.y); })
      .attr("width", function(d) {return x.rangeBand(d.x);});
    
      //redraw the text
      label.attr("x", function(d) { return x(d) + x.rangeBand(d) / 2; });

      var drag = d3.behavior.drag()
                .on('dragstart', null)
                .on('drag', function(d){
                  // move circle
                  var dx = d3.event.dx;
                  var xNew = parseFloat(d3.select(this).attr('x1'))+ dx;

                  line.attr("x1",xNew)
                      .attr("y1",(205 * (-1)))
                      .attr("x2",xNew)
                      .attr("y2",15);
                  x.focus(xNew);
                  //redraw the bars
                  rect
                  .attr("x", function(d) { return x(d.x); })
                  .attr("y", function(d) { return -y(d.y0) - y(d.y); })
                  .attr("width", function(d) {return x.rangeBand(d.x);});

                  var thresholdNew = xNew / (w - p[1] - p[3])

                  EventBus.$emit('sendNewUpdatedThreshold', thresholdNew)
                  //redraw the text
                  label.attr("x", function(d) { return x(d) + x.rangeBand(d) / 2; });
                  // console.log(d)
                  }).on('dragend', function(){
                    EventBus.$emit('callUpdateOfUMAP')
                }); 

      // var line = svg.append('g')
      //   .attr('transform', 'translate(0, 0)')

      var lineStable = svg.append('line')
        .attr("x1", limiter)
        .attr("y1", (205 * (-1)))
        .attr("x2", limiter)
        .attr("y2", 15)
        .attr("stroke-width", 1.5)
        .attr("stroke", "black")
        .attr("class", 'dashed');

      var line = svg.append('line')
        .attr("x1", limiter)
        .attr("y1", (205 * (-1)))
        .attr("x2", limiter)
        .attr("y2", 15)
        .attr("stroke-width", 3)
        .attr("stroke", "black")
        .call(drag);

      // line.append('text')
      //   .text('threshold')
      //   .attr('x', limiter+3)
      //   .attr('y', -2);

      //respond to the mouse and distort where necessary
      // svg.on("mousemove", function() {
      //     var mouse = d3.mouse(this);
          
      //     //refocus the distortion
      //     x.focus(mouse[0]);
      //     //redraw the bars
      //     rect
      //     .attr("x", function(d) { return x(d.x); })
      //     .attr("y", function(d) { return -y(d.y0) - y(d.y); })
      //     .attr("width", function(d) {return x.rangeBand(d.x);});
          
      //     //redraw the text
      //     label.attr("x", function(d) { return x(d) + x.rangeBand(d) / 2; });


      //     var dateLine = svg.append('g')
      //       .attr('transform', 'translate(0, 0)')
      //     dateLine.append('line')
      //       .attr("x1", mouse[0])
      //       .attr("y1", (h * (-1)))
      //       .attr("x2", mouse[0])
      //       .attr("y2", 100)
      //       .attr("stroke-width", 2)
      //       .attr("stroke", "black");

      //     dateLine.append('text')
      //       .text('threshold')
      //       .attr('x', mouse[0])
      //       .attr('y', 0);
      // });
      EventBus.$emit('sendWhichRuleIsActiveLocally', currentIndex)
      EventBus.$emit('currentIndexForUMAP', currentIndex)
    }
  },
  mounted () {
    EventBus.$on('emittedEventCallingInfo', data => {
      this.surrogateData = data })
    EventBus.$on('clickedANewImpureRuleFlag', data => {
      this.flagClicked = false })
    EventBus.$on('clickedANewImpureRule', data => {
      this.checkingRule = data })
    EventBus.$on('clickedANewImpureRule', this.problemIdentifier)
    EventBus.$on('emittedEventCallingSurrogateData', data => {
      this.problematic = data })
    EventBus.$on('emittedEventCallingSurrogateData', this.problemIdentifier)
  },
}
</script>
<style>
#histogram {
  font: 11px sans-serif;
  shape-rendering: crispEdges;
}

.whole {
  display: flex;
  min-width: 820px !important;
  max-width: 820px !important;
  width: 820px !important;
}

.left {
  flex: 0 0 25%;
}

.right {
    flex: 1;
}

.dashed {
  stroke-dasharray: 5,3;
}
</style>
