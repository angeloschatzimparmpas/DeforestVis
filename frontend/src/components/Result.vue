<template>
  <div>
    <div id="stacks" class="two"></div>
    <svg id="stacksLegend"></svg>
  </div>
</template>

<script>
import { EventBus } from '../main.js'
import * as Plotly from 'plotly.js'
import $ from 'jquery'; // <-to import jquery

export default {
  name: 'Result',
  data () {
    return {
      Results: 0,
      sorting: -1,
      firstTime: -1,
    }
  },
  methods: {
    outcome () {
      
      var target_names = JSON.parse(this.Results[0])
      var colorsDiverging = ["#4b0856","#4d0a58","#4f0c5a","#510d5c","#530f5f","#551161","#571263","#591465","#5b1667","#5d1869","#5f1a6b","#611c6d","#631d70","#651f72","#672174","#692376","#6b2578","#6d277a","#6e297c","#702b7d","#722e7f","#743081","#753283","#773485","#793787","#7a3989","#7c3b8a","#7e3e8c","#7f408e","#814390","#824591","#844893","#854a95","#874d96","#884f98","#8a5299","#8b549b","#8d579d","#8e599e","#905ca0","#915ea1","#9361a3","#9463a4","#9666a6","#9768a7","#996ba9","#9a6daa","#9b70ac","#9d72ad","#9f74af","#a077b0","#a279b2","#a37bb3","#a57db5","#a680b6","#a882b7","#a984b9","#ab86ba","#ac88bc","#ae8bbd","#af8dbe","#b18fc0","#b391c1","#b493c2","#b695c4","#b797c5","#b999c6","#ba9bc8","#bc9dc9","#bd9fca","#bfa1cb","#c1a3cd","#c2a5ce","#c4a7cf","#c5a9d0","#c7abd1","#c8add2","#caafd3","#cbb1d5","#cdb2d6","#ceb4d7","#cfb6d8","#d1b8d9","#d2bada","#d4bcdb","#d5bddc","#d6bfdd","#d8c1de","#d9c3df","#dac5e0","#dbc6e0","#ddc8e1","#decae2","#dfcbe3","#e0cde4","#e1cfe5","#e2d0e6","#e4d2e6","#e5d4e7","#e6d5e8","#e6d7e9","#e7d8e9","#e8daea","#e9dbeb","#eaddeb","#ebdeec","#ebe0ec","#ece1ed","#ede2ed","#ede3ee","#eee5ee","#eee6ef","#efe7ef","#efe8ef","#efe9ef","#f0eaf0","#f0ebf0","#f0ecf0","#f0edf0","#f0eeef","#f0eeef","#f0efef","#eff0ef","#eff0ee","#eff1ee","#eef1ed","#eef2ed","#edf2ec","#edf2eb","#ecf2ea","#ebf2e9","#ebf3e8","#eaf3e7","#e9f3e6","#e8f2e5","#e7f2e4","#e6f2e3","#e5f2e1","#e4f2e0","#e2f2df","#e1f1dd","#e0f1dc","#def1da","#ddf0d9","#dcf0d7","#daefd6","#d9efd4","#d7eed2","#d6eed1","#d4edcf","#d2edcd","#d1eccb","#cfebc9","#cdebc8","#cbeac6","#cae9c4","#c8e9c2","#c6e8c0","#c4e7be","#c2e6bc","#c0e5ba","#bee4b8","#bce4b6","#bae3b4","#b8e2b2","#b6e1b0","#b3e0ae","#b1dfac","#afdeaa","#addca8","#aadba6","#a8daa4","#a6d9a1","#a3d89f","#a1d69d","#9ed59b","#9bd498","#99d296","#96d194","#94cf91","#91ce8f","#8ecc8d","#8ccb8b","#89c988","#86c886","#83c684","#80c481","#7ec37f","#7bc17d","#78bf7a","#75bd78","#72bc76","#70ba74","#6db871","#6ab66f","#67b46d","#64b26b","#62b069","#5fae67","#5cad65","#59ab62","#57a960","#54a75e","#51a55d","#4fa35b","#4ca159","#4a9f57","#479d55","#459b53","#429951","#409650","#3d944e","#3b924c","#39904b","#368e49","#348c47","#328a46","#308844","#2e8643","#2c8441","#298240","#28803e","#267e3d","#247b3b","#22793a","#207739","#1e7537","#1d7336","#1b7135","#1a6f33","#186d32","#176b31","#156930","#14672e","#12652d","#11632c","#10612b","#0f5f2a","#0d5d28","#0c5a27","#0b5826","#0a5625","#095424","#085223","#065022","#054e21"]

      var svgLegendStack = d3.select("#stacksLegend");
      svgLegendStack.selectAll("*").remove();

      svgLegendStack.append("text").attr("x", 0).attr("y", 0).text(function(d){ return target_names[0];}).attr('transform', 'translate(43,360)rotate(-90)').style("font-size", "13px").attr("alignment-baseline","middle")
      svgLegendStack.append("text").attr("x", 0).attr("y", 0).text(function(d){ return target_names[1];}).attr('transform', 'translate(43,75)rotate(-90)').style("font-size", "13px").attr("alignment-baseline","middle")
      svgLegendStack.append("text").attr("x", 0).attr("y", 0).text(function(d){ return "Influence Level per Class";}).attr('transform', 'translate(43,275)rotate(-90)').style("font-size", "13px").attr("alignment-baseline","middle")

      var legendFullHeight = 380;
      var legendFullWidth = 50;

      var legendMargin = { top: 25, bottom: 10, left: 0, right: 28 };

      // use same margins as main plot
      var legendWidth = legendFullWidth - legendMargin.left - legendMargin.right;
      var legendHeight = legendFullHeight - legendMargin.top - legendMargin.bottom;

      svgLegendStack = d3.select('#stacksLegend')
          .attr('width', legendFullWidth)
          .attr('height', legendFullHeight)
          .append('g')
          .attr('transform', 'translate(' + legendMargin.left + ',' +
          legendMargin.top + ')');

      updateColourScale(colorsDiverging);

      // update the colour scale, restyle the plot points and legend
      function updateColourScale(scale) {

        // append gradient bar
        var gradient = svgLegendStack.append('defs')
            .append('linearGradient')
            .attr('id', 'gradientStack')
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

        svgLegendStack.append('rect')
            .attr('x1', 0)
            .attr('y1', 0)
            .attr('width', legendWidth)
            .attr('height', legendHeight)
            .style('fill', 'url(#gradientStack)');

        // create a scale and axis for the legend
        var legendScale = d3.scale.linear()
            .domain([-1, 1])
            .range([legendHeight, 0]);

        var legendAxis = d3.svg.axis()
            .scale(legendScale)
            .orient("right")
            .tickValues([-1, 0, 1])
            .tickFormat(d3.format("d"));

            svgLegendStack.append("g")
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

      const config = {
        displayModeBar: false, // this is the line that hides the bar.
      };
      
      //Plotly.purge('bars')
      
      var orderedFeatures = this.Results[13]
      var gatherStacksList = []
      var gatherStacks = this.Results[30]
      var gatherColorsStackList = []
      var gatherColorsStack = this.Results[31]
      var gatherOpacitiesStackList = []
      var gatherOpacitiesStack = this.Results[32]
      var gatherStacksHover = this.Results[44]
      var maxPerFeatureList = []
      var sortingLoc = this.sorting

      // bar charts
      var GTTestCol = []
      var PredTestCol = []
      var index_list = []
      var diffTest = []
      var rectWidths = []
      var rectWidthsRemaining = []
      var warn = []
      var GTTestCol = [...this.Results[25]]
      var PredTestCol = [...this.Results[26]]
      var diffTest = [...this.Results[27]]
      var index_list = [...this.Results[28]]
      var rectWidths = [...this.Results[29]]
      var warn = [...this.Results[33]]
      var rectWidthsRemaining = [...this.Results[43]]

      var maxDiff = Math.max(...diffTest)

      let scaleDiff = d3.scale.linear()
        .domain([0, maxDiff])
        .range([0, 8]);

      var diffTestScaled = []
      var diffTestSubtract = []
      for (let index = 0; index < diffTest.length; index++) {
        diffTestScaled.push(scaleDiff(diffTest[index]))
      }

      var maxSubstract = Math.max(...diffTestScaled)
      
      for (let index = 0; index < diffTestScaled.length; index++) {
        diffTestSubtract.push((maxSubstract - diffTestScaled[index]) + 2)
      }

      if (sortingLoc != -1 ) {
        var len = gatherStacks[sortingLoc].length
        var misClassSlice = gatherStacks[sortingLoc].slice(0,this.firstTime);
        var corClassSlice = gatherStacks[sortingLoc].slice(this.firstTime,len)
        var gatherStacksSortedNested = []
        var gatherColorsStackSortedNested = []
        var gatherOpacitiesStackSortedNested = []
        var misClassIndices = new Array(misClassSlice.length);
        var corClassIndices = new Array(corClassSlice.length);
        for (var i = 0; i < misClassSlice.length; ++i) misClassIndices[i] = i;
        misClassIndices.sort(function (a, b) { return misClassSlice[a] > misClassSlice[b] ? -1 : misClassSlice[a] < misClassSlice[b] ? 1 : 0; });
        for (var i = 0; i < corClassSlice.length; ++i) corClassIndices[i] = i;
        corClassIndices.sort(function (a, b) { return corClassSlice[a] > corClassSlice[b] ? -1 : corClassSlice[a] < corClassSlice[b] ? 1 : 0; });
        corClassIndices.forEach((element, index) => {
          corClassIndices[index] = element + misClassSlice.length;
        });
        var indices = misClassIndices.concat(corClassIndices);
        var GTTestColSorted = new Array(len)
        var PredTestColSorted = new Array(len)
        var diffTestSorted = new Array(len)
        var diffTestSortedSubtract = new Array(len)
        for (let index = 0; index < indices.length; index++) {
          GTTestColSorted[index] = GTTestCol[indices[index]];
          PredTestColSorted[index] = PredTestCol[indices[index]];
          diffTestSorted[index] = diffTestScaled[indices[index]]
          diffTestSortedSubtract[index] = diffTestSubtract[indices[index]]
        }
        GTTestCol = [...GTTestColSorted]
        PredTestCol = [...PredTestColSorted]        
        diffTestScaled = [...diffTestSorted]
        diffTestSubtract = [...diffTestSortedSubtract]

        for (let index = 0; index < gatherStacks.length; index++) {
          var gatherStacksSorted = new Array(len)
          var gatherColorsStackSorted = new Array(len)
          var gatherOpacitiesStackSorted = new Array(len)
          for (let i = 0; i < gatherStacks[index].length; i++) {
            gatherStacksSorted[i] = gatherStacks[index][indices[i]]
            gatherColorsStackSorted[i] = gatherColorsStack[index][indices[i]]
            gatherOpacitiesStackSorted[i] = gatherOpacitiesStack[index][indices[i]]
          }
          gatherStacksSortedNested.push(gatherStacksSorted)
          gatherColorsStackSortedNested.push(gatherColorsStackSorted)
          gatherOpacitiesStackSortedNested.push(gatherOpacitiesStackSorted)
        }
        gatherStacks = [...gatherStacksSortedNested]
        gatherColorsStack = [...gatherColorsStackSortedNested]
        gatherOpacitiesStack = [...gatherOpacitiesStackSortedNested]
      }

      index_list.forEach((num, index) => {
        index_list[index] = num + 1;
        if (this.firstTime == -1) {
          if((GTTestCol[index] == PredTestCol[index]) && (GTTestCol[index-1] != PredTestCol[index-1])) {
            this.firstTime = index
            rectWidths.splice(index, 0, 4);
            rectWidthsRemaining.splice(index, 0, 0.5)
            diffTestScaled.splice(index, 0, 0);
            diffTestSubtract.splice(index, 0, 0)
            GTTestCol.splice(index, 0, 'rgb(255, 255, 255)');
            PredTestCol.splice(index, 0, 'rgb(255, 255, 255)');
            warn.splice(index, 0, 'rgb(255, 255, 255)');
            rectWidths.splice(index+1, 0, 4);
            rectWidthsRemaining.splice(index+1, 0, 0.5)
            diffTestScaled.splice(index+1, 0, 0);
            diffTestSubtract.splice(index+1, 0, 0)
            GTTestCol.splice(index+1, 0, 'rgb(255, 255, 255)');
            PredTestCol.splice(index+1, 0, 'rgb(255, 255, 255)');
            warn.splice(index+1, 0, 'rgb(255, 255, 255)');
          } 
        } else {
          if(index == this.firstTime) {
            this.firstTime = index
            rectWidths.splice(index, 0, 4);
            rectWidthsRemaining.splice(index, 0, 0.5)
            diffTestScaled.splice(index, 0, 0);
            diffTestSubtract.splice(index, 0, 0)
            GTTestCol.splice(index, 0, 'rgb(255, 255, 255)');
            PredTestCol.splice(index, 0, 'rgb(255, 255, 255)');
            warn.splice(index, 0, 'rgb(255, 255, 255)');
            rectWidths.splice(index+1, 0, 4);
            rectWidthsRemaining.splice(index+1, 0, 0.5)
            diffTestScaled.splice(index+1, 0, 0);
            diffTestSubtract.splice(index+1, 0, 0)
            GTTestCol.splice(index+1, 0, 'rgb(255, 255, 255)');
            PredTestCol.splice(index+1, 0, 'rgb(255, 255, 255)');
            warn.splice(index+1, 0, 'rgb(255, 255, 255)');
            }
          }
      });
      
      rectWidths.splice(diffTestScaled.length, 0, 4);
      rectWidthsRemaining.splice(diffTestScaled.length, 0, 0.5)
      diffTestScaled.splice(diffTestScaled.length, 0, 0);
      diffTestSubtract.splice(diffTestScaled.length, 0, 0)
      GTTestCol.splice(diffTestScaled.length, 0, 'rgb(255, 255, 255)');
      PredTestCol.splice(diffTestScaled.length, 0, 'rgb(255, 255, 255)');
      warn.splice(diffTestScaled.length, 0, 'rgb(255, 255, 255)');
      rectWidths.splice(diffTestScaled.length, 0, 4);
      rectWidthsRemaining.splice(diffTestScaled.length, 0, 0.5)
      diffTestScaled.splice(diffTestScaled.length, 0, 0);
      diffTestSubtract.splice(diffTestScaled.length, 0, 0)
      GTTestCol.splice(diffTestScaled.length, 0, 'rgb(255, 255, 255)');
      PredTestCol.splice(diffTestScaled.length, 0, 'rgb(255, 255, 255)');
      warn.splice(diffTestScaled.length, 0, 'rgb(255, 255, 255)');
      rectWidths.splice(diffTestScaled.length, 0, 4);
      rectWidthsRemaining.splice(diffTestScaled.length, 0, 0.5)
      diffTestScaled.splice(diffTestScaled.length, 0, 0);
      diffTestSubtract.splice(diffTestScaled.length, 0, 0)
      GTTestCol.splice(diffTestScaled.length, 0, 'rgb(255, 255, 255)');
      PredTestCol.splice(diffTestScaled.length, 0, 'rgb(255, 255, 255)');
      warn.splice(diffTestScaled.length, 0, 'rgb(255, 255, 255)');

      index_list = []
      for (let index = 0; index < diffTestScaled.length; index++) {
        index_list.push(index+1)
      }

      var gatherHoverGT = []
      GTTestCol.forEach(element => {
        if (element == 'rgb(90, 174, 97)') {
          gatherHoverGT.push(target_names[1])
        } else if ('rgb(153, 112, 171)') {
          gatherHoverGT.push(target_names[0])
        } else {
          gatherHoverGT.push('')
        }
      });

      var gatherHoverPred = []
      PredTestCol.forEach(element => {
        if (element == 'rgb(90, 174, 97)') {
          gatherHoverPred.push(target_names[1])
        } else if ('rgb(153, 112, 171)') {
          gatherHoverPred.push(target_names[0])
        } else {
          gatherHoverPred.push('')
        }
      });

      var trace1 = {
        name: 'GT',
        showlegend: false,
        x: rectWidths,
        y: index_list.reverse(),
        orientation: 'h',
        hovertemplate: '%{text}',
        text: gatherHoverGT,
        marker: {
          color: GTTestCol,
          line: {
            color: warn,
            width: 0.5
          }
        },
        xref: 'x',
        yref: 'y',
        type: 'bar'
      };

      var trace2 = {
          x: rectWidthsRemaining,
          y: index_list.reverse(),
          orientation: 'h',
          name: '',
          hoverinfo: 'skip',
          showlegend: false,
          type: 'bar',
          marker: {
              color: 'rgb(255,255,255)',
              opacity: 1,
              line: {
                color: 'rgb(255,255,255)',
                width: 0.5
              }
          },
          xref: 'x',
          yref: 'y',
        }

      var trace3 = {
        name: 'Pred',
        x: rectWidths,
        showlegend: false,
        y: index_list.reverse(),
        orientation: 'h',
        hovertemplate: '%{text}',
        text: gatherHoverPred,
        marker: {
          color: PredTestCol,
          line: {
            color: warn,
            width: 0.5
          }
        },
        xref: 'x',
        yref: 'y',
        type: 'bar'
      };

      var trace4 = {
          x: rectWidthsRemaining,
          y: index_list.reverse(),
          orientation: 'h',
          name: '',
          hoverinfo: 'skip',
          showlegend: false,
          type: 'bar',
          marker: {
              color: 'rgb(255,255,255)',
              opacity: 1,
              line: {
                color: 'rgb(255,255,255)',
                width: 0.5
              }
          },
          xref: 'x',
          yref: 'y',
        }

      for (let index = 0; index < diffTest.length; index++) {
        diffTest[index] = (Math.round(diffTest[index] * 100) / 100).toFixed(2);
      }
    
      var trace5 = {
        name: 'Δ(WxP)',
        x: diffTestScaled,
        showlegend: false,
        hovertemplate: '%{text}',
        text: diffTest,
        y: index_list.reverse(),
        orientation: 'h',
        marker: {
          color: 'rgba(123,62,28, 0.75)',
          line: {
            color: 'rgb(255,255,255)',
            width: 0.5
          }
        },
        xref: 'x',
        yref: 'y',
        type: 'bar'
      };

      var trace6 = {
          x: diffTestSubtract,
          y: index_list.reverse(),
          orientation: 'h',
          name: '',
          hoverinfo: 'skip',
          showlegend: false,
          type: 'bar',
          marker: {
              color: 'rgb(255,255,255)',
              opacity: 1,
              line: {
                color: 'rgb(255,255,255)',
                width: 0.5
              }
          },
          xref: 'x',
          yref: 'y',
        }

      Plotly.purge('stacks')

      // stacked bar charts

      for (let index = 0; index < gatherStacks.length; index++) {
        var maxPerFeature = Math.max(...gatherStacks[index]) + 1
        //var maxPerFeature = Math.max(...gatherStacks[index])
        maxPerFeatureList.push(maxPerFeature)
      }
      var gatherDifferenceList = []
      for (let index = 0; index < gatherStacks.length; index++) {
        // let scaleStacks = d3.scale.linear()
        //     .domain([0, maxPerFeatureList[index]])
        //     .range([0, 70]);
        var gatherDifference = []
        var gatherStacksInject = []
        var gatherColorsStackInject = []
        var gatherOpacitiesStackInject = []
        for (let item = 0; item < gatherStacks[index].length; item++) {
          if (item == gatherStacks[index].length) {
            gatherDifference.push(0)
            gatherDifference.push(0)
            gatherDifference.push(0)
            gatherStacksInject.push(0)
            gatherStacksInject.push(0)
            gatherStacksInject.push(0)
            gatherColorsStackInject.push('rgb(255,255,255)')
            gatherColorsStackInject.push('rgb(255,255,255)')
            gatherColorsStackInject.push('rgb(255,255,255)')
            gatherOpacitiesStackInject.push(0)
            gatherOpacitiesStackInject.push(0)       
            gatherOpacitiesStackInject.push(0)       
          }
          if(this.firstTime == item) { 
            gatherDifference.push(0)
            gatherDifference.push(0)
            gatherStacksInject.push(0)
            gatherStacksInject.push(0)
            gatherColorsStackInject.push('rgb(255,255,255)')
            gatherColorsStackInject.push('rgb(255,255,255)')
            gatherOpacitiesStackInject.push(0)
            gatherOpacitiesStackInject.push(0)
          }
          gatherStacksInject.push(gatherStacks[index][item])
          gatherColorsStackInject.push(gatherColorsStack[index][item])
          gatherOpacitiesStackInject.push(gatherOpacitiesStack[index][item])
          gatherDifference.push(maxPerFeatureList[index] - gatherStacks[index][item])
        }
        gatherStacksList.push(gatherStacksInject)
        gatherColorsStackList.push(gatherColorsStackInject)
        gatherOpacitiesStackList.push(gatherOpacitiesStackInject)
        gatherDifferenceList.push(gatherDifference)
      }

      var width = 1235
      var height = 380

      var dataStacks = []

      dataStacks.push(trace1, trace2, trace3, trace4, trace5, trace6)
      
      for (let index = 0; index < orderedFeatures.length; index++) {
          dataStacks.push({
            x: gatherStacksList[index],
            y: index_list.reverse(),
            orientation: 'h',
            showlegend: false,
            hovertemplate: '%{text}',
            text: gatherStacksHover[index],
            name: orderedFeatures[index],
            marker: {
              color: gatherColorsStackList[index],
              opacity: gatherOpacitiesStackList[index],
              line: {
                color: 'rgb(255,255,255)',
                width: 0.5
              }
            },
            type: 'bar'
          })
        dataStacks.push({
          x: gatherDifferenceList[index],
          y: index_list.reverse(),
          orientation: 'h',
          hoverinfo: 'skip',
          name: '',
          showlegend: false,
          type: 'bar',
          marker: {
              color: 'rgb(255,255,255)',
              opacity: 1,
              line: {
                color: 'rgb(255,255,255)',
                width: 0.5
              }
          }
        })
      }
        var annotationsGatherer = []
        annotationsGatherer.push({
          x: (2),
          y: (index_list.length),
          xref: 'x',
          yref: 'y',
          captureevents: true,
          text: 'GT',
          showarrow: false,
          font: {
              family: 'sans-serif',
              size: 16,
              color: '#000000'
            },
        })
        annotationsGatherer.push({
          x: (6.5),
          y: (index_list.length),
          xref: 'x',
          yref: 'y',
          captureevents: true,
          text: 'Pred',
          showarrow: false,
          font: {
              family: 'sans-serif',
              size: 16,
              color: '#000000'
            },
        })
        if (sortingLoc == -1) {
            annotationsGatherer.push({
            x: (13.5),
            y: (index_list.length),
            xref: 'x',
            yref: 'y',
            captureevents: true,
            text: '<b>Δ(WxP)</b>',
            showarrow: false,
            font: {
                family: 'sans-serif',
                size: 16,
                color: '#000000'
              },
          })      
        } else {
          annotationsGatherer.push({
            x: (13.5),
            y: (index_list.length),
            xref: 'x',
            yref: 'y',
            captureevents: true,
            text: 'Δ(WxP)',
            showarrow: false,
            font: {
                family: 'sans-serif',
                size: 16,
                color: '#000000'
              },
          })        
        }


      for (let index = 0; index < orderedFeatures.length; index++) {
        var stackEachLayerOfAnnotation = 19
        for (let loop = 0; loop <= index; loop++) {
          if (loop == index) {
            stackEachLayerOfAnnotation = stackEachLayerOfAnnotation + maxPerFeatureList[loop] / 2.4
          } else {
            stackEachLayerOfAnnotation = stackEachLayerOfAnnotation + maxPerFeatureList[loop]
          }
        }
        if (sortingLoc == -1) {
          annotationsGatherer.push({
              x: (stackEachLayerOfAnnotation),
              y: (index_list.length),
              xref: 'x',
              yref: 'y',
              captureevents: true,
              text: orderedFeatures[index],
              showarrow: false,
              font: {
                  family: 'sans-serif',
                  size: 16,
                  color: '#000000'
                },
            })
        } else {
          if (sortingLoc == index) {
            annotationsGatherer.push({
              x: (stackEachLayerOfAnnotation),
              y: (index_list.length),
              xref: 'x',
              yref: 'y',
              captureevents: true,
              text: '<b>'+orderedFeatures[index]+'</b>',
              showarrow: false,
              font: {
                  family: 'sans-serif',
                  size: 16,
                  color: '#000000'
                },
            })
          } else {
            annotationsGatherer.push({
              x: (stackEachLayerOfAnnotation),
              y: (index_list.length),
              xref: 'x',
              yref: 'y',
              captureevents: true,
              text: orderedFeatures[index],
              showarrow: false,
              font: {
                  family: 'sans-serif',
                  size: 16,
                  color: '#000000'
                },
            })
          }
        }

      }

      var layoutStacks = {
        plot_bgcolor: "rgba(0,0,0,0)",
        paper_bgcolor: "rgba(0,0,0,0)",
        annotations: annotationsGatherer,
        // yaxis: {
        //   visible: false,
        // },
        showlegend: false,
        xaxis: {
          autorange: true,
          visible: false,
          },
        //hovermode: 'closest',
        // showlegend: false,
        width: width,
        height: height,
        barmode: 'stack',
        margin: {
          l: 0,
          r: 0,
          b: 0,
          t: 0,
          pad: 0
        },
      };

     var plotDiv = document.getElementById('stacks')

      Plotly.newPlot(plotDiv, dataStacks, layoutStacks, config);
      
      plotDiv.on('plotly_clickannotation', function(data){
        if (data.index == 0 || data.index == 1 || data.index == 2) {
          EventBus.$emit('sortingCall', -1)
        } else {
          EventBus.$emit('sortingCall', (data.index-3))
        }
      });

      var SplittingPoint = this.firstTime

      plotDiv.on('plotly_hover', function(data){
        var activeTestID = data.points[0].pointNumber
        if (((activeTestID) == SplittingPoint) || ((activeTestID) == (SplittingPoint+1)) || (activeTestID == index_list.length) || (activeTestID == index_list.length-1) || (activeTestID == index_list.length-2) || (activeTestID == index_list.length-3)) {
          EventBus.$emit('hoveringTestSample', -1)
        } else {
          if ((activeTestID) < SplittingPoint) {

          } else {
            activeTestID = activeTestID - 2
          }
          EventBus.$emit('hoveringTestSample', activeTestID)
        }
      });

      plotDiv.on('plotly_unhover', function(data){
        EventBus.$emit('hoveringTestSample', -1)
      });

    }
  },
  mounted () {
    EventBus.$on('resetFirstTime', data => { this.firstTime = -1})
    EventBus.$on('sortingCall', data => {
      this.sorting = data })
    EventBus.$on('sortingCall', this.outcome)
    EventBus.$on('emittedEventCallingSurrogateData', data => {
      this.Results = data })
    EventBus.$on('emittedEventCallingSurrogateData', this.outcome)
  },
}
</script>

<style>
#stacks {
  float:left; 
  width:96%;
  height:380px;
  z-index: 0 !important;
}

#stacksLegend {
  float:right;
  width:4%;
  height:380px;
  z-index: 5 !important;
}
</style>