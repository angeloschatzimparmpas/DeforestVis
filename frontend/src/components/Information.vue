<template>
<div>
  <div id="lollipop"></div>
  <svg id="LegendsScatterPlots"></svg>
  <div id="container" class="outer">
    <div id="dotplot"></div>
    <div id="statisticsProbaOne" class="same"></div>
    <div id="statisticsAlphaOne" class="same"></div>
    <div id="statisticsDecOne" class="same"></div>
    <div id="gridVisualizationOne" class= "same"></div>
  </div>
  <svg id="LegendModeUnique"></svg>
</div>
</template>

<script>
import * as colorbr from 'colorbrewer'
import { EventBus } from '../main.js'
import * as d3Base from 'd3'
import * as Plotly from 'plotly.js'
import $ from 'jquery'; // <-to import jquery
import 'bootstrap-toggle/css/bootstrap-toggle.css'
import 'bootstrap-slider/dist/css/bootstrap-slider.css'
// attach all d3 plugins to the d3 library
const d3v5 = Object.assign(d3Base)

const colorbrewer = Object.assign(colorbr)
export default {
  name: 'Information',
  data () {
    return {
      defaultDataSet: 'HeartC', // default value for the first data set
      dataset: 'Data Set:',
      surrogateData: 0,
      mode: 'performant',
      groupRule: [],
      classesNumber: 8
    }
  },
  methods: {
    selectDataSet () {   
      const fileName = document.getElementById('selectFile')
      this.defaultDataSet = fileName.options[fileName.selectedIndex].value
      this.defaultDataSet = this.defaultDataSet.split('.')[0]

      this.dataset = "Data set"
      d3.select("#data").select("input").remove(); // Remove the selection field.
      EventBus.$emit('SendToServerDataSetConfirmation', this.defaultDataSet)
    },
    surrogateFun () { 

      var svg = d3v5.select("#lollipop");
      svg.selectAll("*").remove();
      var svgDot = d3.select("#dotplot");
      svgDot.selectAll("*").remove();
      var svgLegendTop = d3.select("#LegendsScatterPlots");
      svgLegendTop.selectAll("*").remove();
      var svgLegendBottom = d3.select("#LegendModeUnique");
      svgLegendBottom.selectAll("*").remove();
      svgLegendBottom = d3v5.select("#LegendModeUnique")

      var modeLocal = this.mode

      var ranking = this.surrogateData[2]
      var parameters = JSON.parse(this.surrogateData[3]).param_n_estimators
      var metrics = JSON.parse(this.surrogateData[4])
      var decisions = JSON.parse(JSON.parse(this.surrogateData[5]))
      var statistics = JSON.parse(JSON.parse(this.surrogateData[6]))
      var information = JSON.parse(JSON.parse(this.surrogateData[7]))
      var rounding = JSON.parse(JSON.parse(this.surrogateData[9]))
      var statisticsRanking = Object.values(statistics['commonality_ranking'])

      var probamulalpha = Object.values(statistics['probamulalpha'])

      var maximumProbaMul = Math.max(...probamulalpha)
      var minimumProbaMul = Math.min(...probamulalpha)

      var colorsDiverging = ["#002051","#002153","#002255","#002356","#002358","#002459","#00255a","#00255c","#00265d","#00275e","#00275f","#002860","#002961","#002962","#002a63","#002b64","#012b65","#022c65","#032d66","#042d67","#052e67","#052f68","#063069","#073069","#08316a","#09326a","#0b326a","#0c336b","#0d346b","#0e346b","#0f356c","#10366c","#12376c","#13376d","#14386d","#15396d","#17396d","#183a6d","#193b6d","#1a3b6d","#1c3c6e","#1d3d6e","#1e3e6e","#203e6e","#213f6e","#23406e","#24406e","#25416e","#27426e","#28436e","#29436e","#2b446e","#2c456e","#2e456e","#2f466e","#30476e","#32486e","#33486e","#34496e","#364a6e","#374a6e","#394b6e","#3a4c6e","#3b4d6e","#3d4d6e","#3e4e6e","#3f4f6e","#414f6e","#42506e","#43516d","#44526d","#46526d","#47536d","#48546d","#4a546d","#4b556d","#4c566d","#4d576d","#4e576e","#50586e","#51596e","#52596e","#535a6e","#545b6e","#565c6e","#575c6e","#585d6e","#595e6e","#5a5e6e","#5b5f6e","#5c606e","#5d616e","#5e616e","#60626e","#61636f","#62646f","#63646f","#64656f","#65666f","#66666f","#67676f","#686870","#696970","#6a6970","#6b6a70","#6c6b70","#6d6c70","#6d6c71","#6e6d71","#6f6e71","#706f71","#716f71","#727071","#737172","#747172","#757272","#767372","#767472","#777473","#787573","#797673","#7a7773","#7b7774","#7b7874","#7c7974","#7d7a74","#7e7a74","#7f7b75","#807c75","#807d75","#817d75","#827e75","#837f76","#848076","#858076","#858176","#868276","#878376","#888477","#898477","#898577","#8a8677","#8b8777","#8c8777","#8d8877","#8e8978","#8e8a78","#8f8a78","#908b78","#918c78","#928d78","#938e78","#938e78","#948f78","#959078","#969178","#979278","#989278","#999378","#9a9478","#9b9578","#9b9678","#9c9678","#9d9778","#9e9878","#9f9978","#a09a78","#a19a78","#a29b78","#a39c78","#a49d78","#a59e77","#a69e77","#a79f77","#a8a077","#a9a177","#aaa276","#aba376","#aca376","#ada476","#aea575","#afa675","#b0a775","#b2a874","#b3a874","#b4a974","#b5aa73","#b6ab73","#b7ac72","#b8ad72","#baae72","#bbae71","#bcaf71","#bdb070","#beb170","#bfb26f","#c1b36f","#c2b46e","#c3b56d","#c4b56d","#c5b66c","#c7b76c","#c8b86b","#c9b96a","#caba6a","#ccbb69","#cdbc68","#cebc68","#cfbd67","#d1be66","#d2bf66","#d3c065","#d4c164","#d6c263","#d7c363","#d8c462","#d9c561","#dbc660","#dcc660","#ddc75f","#dec85e","#e0c95d","#e1ca5c","#e2cb5c","#e3cc5b","#e4cd5a","#e6ce59","#e7cf58","#e8d058","#e9d157","#ead256","#ebd355","#ecd454","#edd453","#eed553","#f0d652","#f1d751","#f1d850","#f2d950","#f3da4f","#f4db4e","#f5dc4d","#f6dd4d","#f7de4c","#f8df4b","#f8e04b","#f9e14a","#fae249","#fae349","#fbe448","#fbe548","#fce647","#fce746","#fde846","#fde946","#fdea45"]

      var colors = d3v5.scaleQuantize()
          .domain([minimumProbaMul, maximumProbaMul])
          .range(colorsDiverging);

      var data = new Array()

      var collectOrder = []
      var collectOrderClean = []
      for (let i = 0; i < ranking.length; i++) {
        collectOrder.push((i+1).toString())
        collectOrderClean.push(i+1)
      }

      var collectNames = []
      var collectValues = []
      for (let i = 0; i < Object.values(rounding).length; i++) {
        var looping = i + 1
        collectNames.push('value'+looping)
        var gatherEach = []
        for (let j = 0; j < Object.values(Object.values(rounding)[i]).length; j++) {
          gatherEach.push(Object.values(rounding)[i][j].toString())
        }
        collectValues.push(gatherEach)
      }
      
        var colorScale = d3v5.scaleLinear()
          .domain([0, collectNames.length-1])
          .range(['#f0f0f0','#000000'])

      // select the svg area
      svgLegendTop = d3v5.select("#LegendsScatterPlots")

      svgLegendTop.append("text").attr("x", 0).attr("y", 0).text(function(d){ return "Rounding Digit (Decimal)";}).attr('transform', 'translate(43,160)rotate(-90)').style("font-size", "13px").attr("alignment-baseline","middle")

      for (var index = 0; index < Object.keys(rounding).length; index++) {
        svgLegendTop.append("circle").attr("cx", function (d) { return 8}).attr("cy", function (d) { return ((index*25)+15);}).attr("r", 4).style("fill", function (d) { return colorScale(index);})
        svgLegendTop.append("text").attr("x", function (d) { return 18}).attr("y", function (d) { return ((index*25)+15);}).text(function(d){ return (index+1);}).style("font-size", "13px").attr("alignment-baseline","middle")
      }

      if (modeLocal == 'performant') {

        svgLegendBottom.append("text").attr("x", 0).attr("y", 0).text(function(d){ return "Weight x Probability (WxP)";}).attr('transform', 'translate(43,165)rotate(-90)').style("font-size", "13px").attr("alignment-baseline","middle")

        var legendFullHeight = 180;
        var legendFullWidth = 40;

        var legendMargin = { top: 5, bottom: 10, left: 0, right: 30 };

        // use same margins as main plot
        var legendWidth = legendFullWidth - legendMargin.left - legendMargin.right;
        var legendHeight = legendFullHeight - legendMargin.top - legendMargin.bottom;
        
        svgLegendBottom = d3.select('#LegendModeUnique')
            .attr('width', legendFullWidth)
            .attr('height', legendFullHeight)
            .append('g')
            .attr('transform', 'translate(' + legendMargin.left + ',' +
            legendMargin.top + ')');

        updateColourScale(colorsDiverging);

        // update the colour scale, restyle the plot points and legend
        function updateColourScale(scale) {

                // append gradient bar
                var gradient = svgLegendBottom.append('defs')
                    .append('linearGradient')
                    .attr('id', 'gradient')
                    .attr('x1', '0%') // bottom
                    .attr('y1', '100%')
                    .attr('x2', '0%') // to top
                    .attr('y2', '0%')
                    .attr('spreadMethod', 'pad');

                // programatically generate the gradient for the legend
                // this creates an array of [pct, colour] pairs as stop
                // values for legend
                var pct = linspace(0, 100, scale.length).map(function(d) {
                    return Math.round(d) + '%';
                });

                var colourPct = d3.zip(pct, scale);

                colourPct.forEach(function(d) {
                    gradient.append('stop')
                        .attr('offset', d[0])
                        .attr('stop-color', d[1])
                        .attr('stop-opacity', 1);
                });

                svgLegendBottom.append('rect')
                    .attr('x1', 0)
                    .attr('y1', 0)
                    .attr('width', legendWidth)
                    .attr('height', legendHeight)
                    .style('fill', 'url(#gradient)');

                // create a scale and axis for the legend
                var legendScale = d3.scale.linear()
                    .domain([Math.floor(minimumProbaMul), Math.ceil(maximumProbaMul)])
                    .range([legendHeight, 0]);

                var legendAxis = d3.svg.axis()
                    .scale(legendScale)
                    .orient("right")
                    .tickValues([Math.floor(minimumProbaMul),Math.round((Math.floor(minimumProbaMul)+Math.ceil(maximumProbaMul))/2),Math.ceil(maximumProbaMul)])
                    .tickFormat(d3.format("d"));

                    svgLegendBottom.append("g")
                    .attr("class", "legend axis")
                    .attr("transform", "translate(" + legendWidth + ", 0)")
                    .call(legendAxis);
            }

            function linspace(start, end, n) {
                var out = [];
                var delta = (end - start) / (n - 1);

                var i = 0;
                while(i < (n - 1)) {
                    out.push(start + (i * delta));
                    i++;
                }

                out.push(end);
                return out;
            }
      } else {
        // select the svg area
        svgLegendBottom = d3v5.select("#LegendModeUnique")

        svgLegendBottom.append("rect")
          .attr("x", 3)
          .attr("y", 115) 
          .attr("width", 8)
          .attr("height", 8)
          .style("fill", "#0000FF")

        svgLegendBottom.append("text").attr("x", 0).attr("y", 0).text(function(d){ return "Unique Rule";}).attr('transform', 'translate(8,105)rotate(-90)').style("font-size", "13px").attr("alignment-baseline","middle")
        
        svgLegendBottom.append("rect")
          .attr("x", 21)
          .attr("y", 115) 
          .attr("width", 8)
          .attr("height", 8)
          .style("fill", "#bdbdbd")
        
        svgLegendBottom.append("text").attr("x", 0).attr("y", 0).text(function(d){ return "Original Rule";}).attr('transform', 'translate(26,105)rotate(-90)').style("font-size", "13px").attr("alignment-baseline","middle")
        
        svgLegendBottom.append("rect")
          .attr("x", 39)
          .attr("y", 115) 
          .attr("width", 8)
          .attr("height", 8)
          .style("fill", "#ffa500")

        svgLegendBottom.append("text").attr("x", 0).attr("y", 0).text(function(d){ return "Duplicated Rule";}).attr('transform', 'translate(44,105)rotate(-90)').style("font-size", "13px").attr("alignment-baseline","middle")
      }

      const max = Math.max(...[].concat(...collectValues));

      for (var i = 0; i < collectValues.length; i++) {
          for (let index = 0; index < collectValues[i].length; index++) {
            if (parseFloat(collectValues[i][index]) == max) {
              var indexMax = index
              break;
            }
          }
      }

      const min = Math.min(...[].concat(...collectValues));

      for (let i = 0; i < collectOrder.length; i++) {
        var obj = {}
        obj['group'] = collectOrder[i]
        obj['apples'] = parameters[i]
        obj['day'] = collectOrderClean[i]
        for (let j = 0; j < collectNames.length; j++) {
          obj[collectNames[j]] = collectValues[j][i]
        }
        
        data[i] = obj
      }
      
      // set the dimensions and margins of the graph
      var margin = {top: 5, right: 30, bottom: 25, left: 30},
          width = 1220 - margin.left - margin.right,
          height = 200 - margin.top - margin.bottom;

      // append the svg object to the body of the page
      svg = d3v5.select("#lollipop")
        .append("svg")
          .attr("width", width + margin.left + margin.right)
          .attr("height", height + margin.top + margin.bottom)
        .append("g")
          .attr("transform",
                "translate(" + margin.left + "," + margin.top + ")");

      // Add Y axis
        var y = d3v5.scaleLinear()
          .domain([min, max])
          .range([ height, 0]);
        // svg.append("g")
        //   .call(d3v5.axisLeft(y))

        svg.append("g")
          .attr("class", "yAxis")
          .call(d3v5.axisLeft(y))
          .append("text")
          .attr("class", "label")
          .attr("transform", "rotate(-90)")
          .attr("y", 6)
          .attr("dy", ".71em")
          .attr("fill", "black")
          .style("text-anchor", "end")
          .style("font-size", "16px")
          .text("Fidelity");

        // X axis
        var x = d3v5.scaleBand()
          .range([ 0, width ])
          .domain(data.map(function(d) { return d.group; }))
          .padding(1);

        svg.append("g")
          .attr("transform", "translate(0," + height + ")")
          .attr("class", "xAxis")
          .call(d3v5.axisBottom(x))
          .append("text")
          .attr("class", "label")
          .attr("x", width)
          .attr("y", -6)
          .style("text-anchor", "end")
          .style("font-size", "16px")
          .text("Complexity Index");

        svg.select('.yAxis').selectAll('text')
          .attr('font-weight', 100)

        svg.select('.xAxis').selectAll('text')
          .attr('id', function(d) { return ('ComInd'+d)})
          .attr('fill', function(d, i) { 
            if (i == indexMax) 
              return '#e31a1c'
            else 
              return 'black'
          })
          .attr('font-weight', function(d, i) { 
            if (i == indexMax) 
              return 900
            else 
              return 100
          })
          .on("click", function(d,i){
            for (let index = 1; index <= 50; index++) {
              d3v5.select('#ComInd'+index).attr("fill", "black");
              d3v5.select('#ComInd'+index).attr("font-weight", "100");
            }
            d3v5.select('#ComInd'+d).attr("fill", "#e31a1c");
            d3v5.select('#ComInd'+d).attr("font-weight", "900");
            EventBus.$emit('updateRuleModel', d)
          });

      function make_y_axis() {        
          return d3.svg.axis()
              .scale(y)
              .orient("left")
              .ticks(10)
      }

    svg.append("g")         
        .attr("class", "grid")
        .call(make_y_axis()
            .tickSize(-width, 0, 0)
            .tickFormat("")
        )

      // Lines
      svg.selectAll("myline")
        .data(data)
        .enter()
        .append("line")
          .attr("x1", function(d) { return x(d.group); })
          .attr("x2", function(d) { return x(d.group); })
          .attr("y1", function(d) { return y(d.value1); })
          .attr("y2", function(d) { return y(d[collectNames[collectNames.length-1]]); })
          .attr("stroke", "#bdbdbd")
          .attr("stroke-width", "1px")

      if (collectNames.length == 1) {
            svg.selectAll("mycircle")
              .data(data)
              .enter()
              .append("circle")
                .attr("cx", function(d) { return x(d.group); })
                .attr("cy", function(d) { return y(d[collectNames[0]]); })
                .attr("r", "2.5")
                .style("fill", function(d) { return '#000000'})
      } else {
        for (let j = 0; j < collectNames.length; j++) {
            svg.selectAll("mycircle")
              .data(data)
              .enter()
              .append("circle")
                .attr("cx", function(d) { return x(d.group); })
                .attr("cy", function(d) { return y(d[collectNames[j]]); })
                .attr("r", "2.5")
                .style("fill", function(d) { return colorScale(j)})
        }
      }


      // // Dot Plotting

    var x = d3.scale.ordinal()
        .rangePoints([22.7450980392, width]);

    var y = d3.scale.linear()
        .range([0, height]);

    var xAxis = d3.svg.axis()
        .scale(x)
        .orient("top");

    var yAxis = d3.svg.axis()
        .scale(y)
        .orient("left");

    margin = {top: 5, right: 50, bottom: 25, left: 32},
        width = 1220 - margin.left - margin.right,
        height = 200 - margin.top - margin.bottom;

    var svgDot = d3.select("#dotplot")
      .append("svg")
        .attr("width", width + margin.left + 2)
        .attr("height", height + margin.top + 10)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    x.domain(data.map(d => d.day));
    y.domain([1, parameters[49]]);

    // svgDot.append("g")
    //     .attr("class", "x axis")
    //     .call(xAxis)
    //     .append("text")
    //     .attr("class", "label")
    //     .attr("x", width)
    //     .attr("y", -6)
    //     .style("text-anchor", "end")
    //     .text("ID");

    svgDot.append("g")
        .attr("class", "y axis")
        .call(yAxis)
        .append("text")
        .attr("class", "label")
        .attr("transform", "rotate(-90)")
        .attr("y", 6)
        .attr("dy", ".71em")
        .style("text-anchor", "end")
        .style("font-size", "16px")
        .text("Trees");

      svgDot.append("g")
      .append("text")
      .attr('id', 'probabilityID')
      .attr("transform", "rotate(-90)")
      .attr("x", -55)
      .attr("y", 6)
      .attr("dy", ".90em")
      .style("text-anchor", "end")
      .style("font-size", "13px")
      .style("visibility", "hidden")
      .text("Probability (%)");

      svgDot.append("g")
      .append("text")
      .attr('id', 'weightID')
      .attr("x", 165)
      .attr("y", 45)
      .attr("dy", ".90em")
      .style("text-anchor", "end")
      .style("font-size", "13px")
      .style("visibility", "hidden")
      .text("Weight");

      svgDot.append("g")
      .append("text")
      .attr('id', 'decisionID')
      .attr("x", 200)
      .attr("y", 145)
      .attr("dy", ".90em")
      .style("text-anchor", "end")
      .style("font-size", "13px")
      .style("visibility", "hidden")
      .text("Decision Threshold");

    svgDot.selectAll('text')
      .attr('font-weight', 100)

    function make_y_axis_dot() {        
      return d3.svg.axis()
          .scale(y)
          .orient("left")
          .ticks(10)
    }

    svgDot.append("g")         
      .attr("class", "grid")
      .call(make_y_axis_dot()
          .tickSize(-width, 0, 0)
          .tickFormat("")
      )

    var groups = svgDot.selectAll(".groups")
        .data(data)
        .enter()
        .append("g")
        .attr("transform", function(d) {
            return "translate(" + 22.7*(d.day) + ".0)";
        });

    var foundSoFar = []
    var addAllModels = -2
    var sameModelRule = [] 
    var foundSoFarCopyX1 = []
    var addAllModelsLoop = -2
    var addAllModelsCopyX1 = -2
    var sameModelRuleCopyX1 = [] 
    var foundSoFarCopyX2 = []
    var addAllModelsCopyX2 = -2
    var sameModelRuleCopyX2 = [] 
    var count = -1

    var dots = groups.selectAll("line")
        .data(function(d) {
            return d3.range(1, +d.apples + 1)
        })
        .enter().append("line")
        .attr("id", function(d) { 
          if (d == 1) {
            count = count + 1
          }
          return ('i'+count+'j'+d)
        })
        .attr("class", function(d, i) {
          addAllModelsLoop = addAllModelsLoop + 2
          return ('cl'+statisticsRanking[addAllModelsLoop])
        })
        // .attr("r", 1)
        .attr("x1", function(d, i) {
          addAllModelsCopyX1 = addAllModelsCopyX1 + 2
            if (i == 0) {
              sameModelRuleCopyX1 = []
            }
            if (foundSoFarCopyX1.includes(statisticsRanking[addAllModelsCopyX1])) {
              if (sameModelRuleCopyX1.includes(statisticsRanking[addAllModelsCopyX1])) {
                return -2
              }
              else {
              sameModelRuleCopyX1.push(statisticsRanking[addAllModelsCopyX1])
                return -4
              }
            } else {
              foundSoFarCopyX1.push(statisticsRanking[addAllModelsCopyX1])
              return -6
            }
        })
        .attr("x2", function(d, i) {
          addAllModelsCopyX2 = addAllModelsCopyX2 + 2
            if (i == 0) {
              sameModelRuleCopyX2 = []
            }
            if (foundSoFarCopyX2.includes(statisticsRanking[addAllModelsCopyX2])) {
              if (sameModelRuleCopyX2.includes(statisticsRanking[addAllModelsCopyX2])) {
                return 2
              }
              else {
                sameModelRuleCopyX2.push(statisticsRanking[addAllModelsCopyX2])
                return 4
              }
            } else {
              foundSoFarCopyX2.push(statisticsRanking[addAllModelsCopyX2])
              return 6
            }
        })
        .attr("y1", function(d) {
          return y(d)
        })
        .attr("y2", function(d) {
            return y(d)
        })
        .attr("stroke", function(d, i) {
          addAllModels = addAllModels + 2
          if (modeLocal == 'performant') {
            return colors(probamulalpha[addAllModels])
          } else {
            if (i == 0) {
              sameModelRule = []
            }
            if (foundSoFar.includes(statisticsRanking[addAllModels])) {
              if (sameModelRule.includes(statisticsRanking[addAllModels])) {
                return '#ffa500'
              }
              else {
                sameModelRule.push(statisticsRanking[addAllModels])
                return '#bdbdbd'
              }
            } else {
              foundSoFar.push(statisticsRanking[addAllModels])
              return '#0000FF'
            }
          }
        })
        .attr("stroke-width", "1.5px")
        .style("stroke-dasharray", "0")
        .on('click', function(d, i, j) { 
          var probabilityID = document.getElementById("probabilityID");
          var weightID = document.getElementById("weightID");
          var decisionID = document.getElementById("decisionID");
          probabilityID.style.visibility = "visible";
          weightID.style.visibility = "visible";
          decisionID.style.visibility = "visible";
          var specificLine = d3v5.select('#i'+j+'j'+(i+1))
          var exactClass = specificLine.attr("class")
          $("."+exactClass).css( "stroke-dasharray", "1");
          $("."+exactClass).css( "stroke-width", "3px");
          specificLine.attr("stroke", "#E31A1C")
          EventBus.$emit('HoveringOverARule', [j,(i+1)])
        })
        .on("dblclick", function (){
          probabilityID.style.visibility = "hidden";
          weightID.style.visibility = "hidden";
          decisionID.style.visibility = "hidden";
          EventBus.$emit('NothingToSee', [])
        }); 
    },
    AddRule () {

      var getRule = Object.values(this.groupRule)

      var target_names = JSON.parse(this.surrogateData[0])
      var statistics = JSON.parse(JSON.parse(this.surrogateData[6]))
      var counterModels = statistics.counterModels
      var alpha = statistics.alpha
      var proba = statistics.proba
      var valueLimit = statistics.valueLimit
      var featureName = statistics.feature
      var predicted_value = statistics.predicted_value
      var locations = JSON.parse(JSON.parse(this.surrogateData[8]))
      var y_train = this.surrogateData[22]

      var keysColl = [];
      var filtered = Object.values(counterModels).filter((e, i) => {
        if (e === (getRule[0]+1)) {
          keysColl.push(i);
        }
      });
      var keepIndex = [0,0]
      for (let index = 0; index < keysColl.length; index++) {
        if (((getRule[1]-1)*2) == index) {
          keepIndex[0] = (keysColl[index])
          keepIndex[1] = (keysColl[index]) + 1
        }
      }

      var probabilities = []
      var probabilities2 = []
      var selectedModelList = []
      var colorEach1 = []
      var colorEach2 = []
      var colorsCompleteList = []
      var alphaModelSingle = []
      var splitModelDec = []
      var colorModelDec = []
      var gatherClass0 = []
      var countLeftClass0 = 0
      var countLeftClass1 = 0
      var countRightClass0 = 0
      var countRightClass1 = 0
      var gatherClass1 = []
      var activeFeature = ''
      for (let index = 0; index < keepIndex.length; index++) {
        if (index == 0) {
          alphaModelSingle.push(Object.values(alpha)[keepIndex[index]])
          for (var i = 0; i < y_train.length; i++) {
            if (Object.values(Object.values(locations)[i])[keepIndex[index]/2] == 1) {
              if (y_train[i] == 0) {
                countLeftClass0 = countLeftClass0 + 1
              } else {
                countLeftClass1 = countLeftClass1 + 1
              }
            } else {
              if (y_train[i] == 0) {
                countRightClass0 = countRightClass0 + 1
              } else {
                countRightClass1 = countRightClass1 + 1
              }            
            }
          } 
        }

        splitModelDec.push(Object.values(valueLimit)[keepIndex[index]])
        splitModelDec.push(1 - Object.values(valueLimit)[keepIndex[index]])
        probabilities.push(Object.values(proba)[keepIndex[index]][0])
        probabilities2.push((100 - Object.values(proba)[keepIndex[index]][0]))
        activeFeature = Object.values(featureName)[keepIndex[index]]
        if (Object.values(predicted_value)[keepIndex[index]] == 0) {
          colorEach1.push('rgb(153, 112, 171)')
          colorModelDec.push('rgb(153, 112, 171)')
          colorModelDec.push('rgb(90, 174, 97)')
          colorEach2.push('rgb(90, 174, 97)')
        } else {
          colorEach1.push('rgb(90, 174, 97)')
          colorModelDec.push('rgb(90, 174, 97)')
          colorModelDec.push('rgb(153, 112, 171)')
          colorEach2.push('rgb(153, 112, 171)')
        }
      }

      console.log(activeFeature)

      gatherClass1.push(countLeftClass0)
      gatherClass0.push(countLeftClass1)
      gatherClass1.push(countRightClass0)
      gatherClass0.push(countRightClass1)

      selectedModelList.push(probabilities)
      selectedModelList.push(probabilities2)
      colorsCompleteList.push(colorEach1)
      colorsCompleteList.push(colorEach2)

      var geometry = this.surrogateData[11]

      var depth = geometry[1]
      var size = geometry[2]

      var cellSize = 4

      var barChartGap = (size * cellSize) / 108

      const config = {
        displayModeBar: false, // this is the line that hides the bar.
      };

      Plotly.purge('statisticsProbaOne')

      var dataStacked = []

      for (let ins = 0; ins < selectedModelList.length; ins++) {
      dataStacked.push({
        x: [0,1],
        width: [0.08, 0.08],
        y: selectedModelList[ins],
        marker: {
        color: colorsCompleteList[ins],
        },
        xaxis: 'x',
        yaxis: 'y',
        type: 'bar'
      })
      }

      var layoutStacked = {
        showlegend: false,
        plot_bgcolor: "rgba(0,0,0,0)",
        paper_bgcolor: "rgba(0,0,0,0)",
        yaxis: {
            'visible': false,
          },
        xaxis: {
            autorange: false,
            showgrid: false,
            showline: false,
            showticklabels: false,
            range: [0, 1],
            type: 'linear',
            title: activeFeature
          },
        barmode: 'stack',
        bargap: barChartGap,
        margin: {
          t: 13,
          r: 10,
          l: 20,
          b: 50
        },
      };

      Plotly.newPlot('statisticsProbaOne', dataStacked, layoutStacked, config);

      Plotly.purge('statisticsAlphaOne')

      var alphaModel = this.surrogateData[17]
      
      var maxAlpha = 0
      alphaModel.forEach(element => {
        if (element.length != 0)
          if (element[0] >= maxAlpha)
            maxAlpha = element[0]
      });
      maxAlpha = maxAlpha / 2
      var dataBar = []
      var temp = []
      temp.push(alphaModelSingle[0] / 2)
      temp.push((alphaModelSingle[0] / 2) * (-1))
          dataBar.push({
            x: temp,
            y: [1, 1],
            width: [0.08, 0.08],
            marker: {
              color: 'rgba(255, 165, 0, 1)',
              // line: {
              //   color: 'black',
              //   width: 1
              // }
            },
            orientation: 'h',
            xaxis: 'x',
            yaxis: 'y',
            type: 'bar'
          })

      var layoutBar = {
        showlegend: false,
        barmode: 'relative',
        plot_bgcolor: "rgba(0,0,0,0)",
        paper_bgcolor: "rgba(0,0,0,0)",
        yaxis: {
            autorange: false,
            showgrid: false,
            showline: false,
            showticklabels: false,
            range: [0, 1],
            type: 'linear',
          },
        xaxis: {
            autorange: false,
            showgrid: false,
            showline: false,
            showticklabels: false,
            range: [-maxAlpha, maxAlpha],
            type: 'linear'
          },
        margin: {
          t: 15,
          r: 10,
          l: 20,
          b: 50
        },
      };

      Plotly.newPlot('statisticsAlphaOne', dataBar, layoutBar, config);  
      
      Plotly.purge('statisticsDecOne')

      var geometry = this.surrogateData[11]

      var dataStackDec = []
      dataStackDec.push({
        x: splitModelDec,
        y: [0, 0],
        width: [0.08, 0.08],
        marker: {
          color: colorModelDec,
          // line: {
          //   color: 'black',
          //   width: 1
          // }
        },
        orientation: 'h',
        xaxis: 'x',
        yaxis: 'y',
        type: 'bar'
      })

      var layoutStackDec = {
        showlegend: false,
        plot_bgcolor: "rgba(0,0,0,0)",
        paper_bgcolor: "rgba(0,0,0,0)",
        yaxis: {
            autorange: false,
            showgrid: false,
            showline: false,
            range: [0, 1],
            tickmode: 'array',
            tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
            ticktext: ['0', '20', '40', '60', '80', '100'],
            type: 'linear',
          },
        xaxis: {
            autorange: false,
            showgrid: false,
            showline: false,
            range: [0, 1],
            type: 'linear'
          },
        barmode: 'stack',
        margin: {
          t: 16,
          r: 10,
          l: 20,
          b: 50
        },
      };

      Plotly.newPlot('statisticsDecOne', dataStackDec, layoutStackDec, config);

      Plotly.purge('gridVisualizationOne')

      var trace1 = {
          x: ['Left Subtree', 'Right Subtree'],
          y: gatherClass0,
          marker: {
            color: 'rgb(90, 174, 97)',
              // line: {
              //   color: 'black',
              //   width: 1
              // }
          },
          showlegend:false,
          hovertemplate: 'Instances</b>: %{y}',
          name: target_names[1],
          type: 'bar'
        };

        var trace2 = {
          x: ['Left Subtree', 'Right Subtree'],
          y: gatherClass1,
          marker: {
          color: 'rgb(153, 112, 171)',
            // line: {
            //   color: 'black',
            //   width: 1
            // }
          },
          hovertemplate: 'Instances: %{y}',
          showlegend:false,
          name: target_names[0],
          type: 'bar'
        };

        if (colorModelDec[0] == "rgb(153, 112, 171)") {
          var data = [trace2, trace1];
        } else {
          var data = [trace1, trace2];
        }

        var layout = {
          width: 220,
          height: 130,
          plot_bgcolor: "rgba(0,0,0,0)",
          paper_bgcolor: "rgba(0,0,0,0)",
          barmode: 'group',
          showlegend:false,
          yaxis: {
            'visible': false,
          },
          margin: {
            t: 18,
            r: 0,
            l: 27,
            b: 50
          },
        };

        Plotly.newPlot('gridVisualizationOne', data, layout, config);

    }, 
    RemoveRule () {
      Plotly.purge('statisticsProbaOne')
      Plotly.purge('statisticsAlphaOne')
      Plotly.purge('statisticsDecOne')
      Plotly.purge('gridVisualizationOne')
    }
  },
  mounted () {
    EventBus.$on('emittedEventCallingInfo', data => {
      this.surrogateData = data })
    EventBus.$on('emittedEventCallingInfo', this.surrogateFun)
    EventBus.$on('SendModeChange', data => {
      this.mode = data })
    EventBus.$on('SendModeChange', this.surrogateFun)
    EventBus.$on('HoveringOverARule', data => {
      this.groupRule = data })
    EventBus.$on('HoveringOverARule', this.AddRule)
    EventBus.$on('NothingToSee', this.RemoveRule)
    EventBus.$on('NothingToSee', this.surrogateFun)
  },
}
</script>

<style>

#lollipop {
    float:left; 
    width:96%;
    height:200px;
}
#LegendsScatterPlots{
    float:right;
    width:4%;
    height:180px;
}

#container {
    float:left; 
    width:96%;
    height:180px;
    position: relative;
}

#LegendModeUnique{
    float:right;
    width:4%;
    height:180px;
}

pre {
    	display: none;
    }

    .axis path,
    .axis line {
      fill: none;
      stroke: #000000;
      shape-rendering: crispEdges;
    }

    .dot {
      stroke: none;
    }
    
    .tick {
      font-size: 12px !important;
    }

    .grid .tick {
      stroke: lightgrey;
      stroke-opacity: 0.3;
      shape-rendering: crispEdges;
    }
    .grid path {
      stroke-width: 0;
    }

    #dotplot{
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
    }
    #statisticsProbaOne,
    #statisticsAlphaOne,
    #statisticsDecOne,
    #gridVisualizationOne {
      width: 20%;
      height: 75%;
      position: absolute;
      z-index: 10;
      transform: translate(52px, 52px);
    }

    .outer {
      display: grid;
      grid-template: 1fr / 1fr;
    }
    .outer > * {
      grid-column: 1 / 1;
      grid-row: 1 / 1;
    }
    .outer .same {
      z-index: 10;
    }

</style>