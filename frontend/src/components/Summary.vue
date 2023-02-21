<template>
<div>
  <div id="summary"></div>
  <div class="outer">
    <div id="statisticsProba" class="same"></div>
    <div id="statisticsAlpha" class="same"></div>
    <div id="statisticsDec" class="same"></div>
    <div id="statisticsDecHighlighting" class="above"></div>
    <div id="gridVisualization" class= "same"></div>
  </div>
  <svg id="LegendBehaviorSummary"></svg>
</div>
</template>

<script>
import { EventBus } from '../main.js'
import * as d3Base from 'd3'
import * as Plotly from 'plotly.js'
import $ from 'jquery'; // <-to import jquery
import 'bootstrap-toggle/css/bootstrap-toggle.css'
import 'bootstrap-slider/dist/css/bootstrap-slider.css'
// attach all d3 plugins to the d3 library
const d3v5 = Object.assign(d3Base)

export default {
  name: 'Summary',
  data () {
    return {
      firstTimeGl: true,
      gridData: 0,
      highlightFeat: -1,
      highlightInit: -1,
      highlightLast: -1,
      hoveredTestIns: -2,
      hoveredTrainIns: -2,
      checkingRule: -1,
    }
  },
  methods: {
    summarize () {

      var widthAll = 1654
      var heightAll = 200
      var target_names = JSON.parse(this.gridData[0])
      var svgLegendBehaviorSummary = d3.select("#LegendBehaviorSummary");
      svgLegendBehaviorSummary.selectAll("*").remove();
      svgLegendBehaviorSummary = d3v5.select("#LegendBehaviorSummary")

      const initialStep = 60

      svgLegendBehaviorSummary.append("rect")
        .attr("x", function(d) {return (widthAll/5)*0+initialStep;})
        .attr("y", 3) 
        .attr("width", 8)
        .attr("height", 8)
        .style("fill", "#ffa500")

        svgLegendBehaviorSummary.append("text").attr("x", function(d) {return (widthAll/5)*0+15+initialStep;}).attr("y", 8.5).text(function(d){ return "Weight";}).style("font-size", "13px").attr("alignment-baseline","middle")

        svgLegendBehaviorSummary.append("rect")
        .attr("x", function(d) {return (widthAll/5)*1+initialStep;})
        .attr("y", 3) 
        .attr("width", 8)
        .attr("height", 8)
        .style("fill", "#a6dba0")

        svgLegendBehaviorSummary.append("text").attr("x", function(d) {return (widthAll/5)*1+15+initialStep;}).attr("y", 8.5).text(function(d){ return "Left Subtree ("+ target_names[1] +")";}).style("font-size", "13px").attr("alignment-baseline","middle")

        svgLegendBehaviorSummary.append("rect")
        .attr("x", function(d) {return (widthAll/5)*2+initialStep;})
        .attr("y", 3) 
        .attr("width", 8)
        .attr("height", 8)
        .style("fill", "#c2a5cf")

        svgLegendBehaviorSummary.append("text").attr("x", function(d) {return (widthAll/5)*2+15+initialStep;}).attr("y", 8.5).text(function(d){ return "Left Subtree ("+ target_names[0] +")";}).style("font-size", "13px").attr("alignment-baseline","middle")

        svgLegendBehaviorSummary.append("rect")
        .attr("x", function(d) {return (widthAll/5)*3+initialStep;})
        .attr("y", 3) 
        .attr("width", 8)
        .attr("height", 8)
        .style("fill", "#1b7837")

        svgLegendBehaviorSummary.append("text").attr("x", function(d) {return (widthAll/5)*3+15+initialStep;}).attr("y", 8.5).text(function(d){ return "Right Subtree ("+ target_names[1] +")";}).style("font-size", "13px").attr("alignment-baseline","middle")

        svgLegendBehaviorSummary.append("rect")
        .attr("x", function(d) {return (widthAll/5)*4+initialStep;})
        .attr("y", 3) 
        .attr("width", 8)
        .attr("height", 8)
        .style("fill", "#762a83")

        svgLegendBehaviorSummary.append("text").attr("x", function(d) {return (widthAll/5)*4+15+initialStep;}).attr("y", 8.5).text(function(d){ return "Right Subtree ("+ target_names[0] +")";}).style("font-size", "13px").attr("alignment-baseline","middle")

      const config = {
        displayModeBar: false, // this is the line that hides the bar.
      };

      Plotly.purge('summary')
    
      var perFeatureUnique = this.gridData[12]
      var orderedFeatures = this.gridData[13]
      var keepMaxValueList = this.gridData[14]
      var TestIDs = this.gridData[42]
      var X_trainRounded = JSON.parse(this.gridData[21])
      var X_testRounded = JSON.parse(this.gridData[23])

      var data = []
      var xAxisComplete = []
      var widthsComplete = []
      var colorsComplete = []
      var powerComplete = []
      var xAxisNames = []
      for (let index = 0; index < perFeatureUnique.length; index++) {
        var xAxis = []
        var widths = []
        var colors = []
        var power = []
        for (let loop = 0; loop < perFeatureUnique[index].length-1; loop++) {
          // xAxis.push(perFeatureUnique[index][loop].toString()+'-'+perFeatureUnique[index][loop+1].toString())
          // xAxis.push(perFeatureUnique[index][loop].toString()+'-'+perFeatureUnique[index][loop+1].toString())
          if (loop < 3) {
            xAxisNames.push(perFeatureUnique[index][loop+1].toString())
          }
          //xAxisNames.push(perFeatureUnique[index][loop+1].toString())
          xAxis.push(((perFeatureUnique[index][loop]+perFeatureUnique[index][loop+1]) / 2))
          xAxis.push(((perFeatureUnique[index][loop]+perFeatureUnique[index][loop+1]) / 2))
          widths.push((perFeatureUnique[index][loop+1] - perFeatureUnique[index][loop]))
          widths.push((perFeatureUnique[index][loop+1] - perFeatureUnique[index][loop]))
          if (keepMaxValueList[index][loop][0] > keepMaxValueList[index][loop][1]) {
            colors.push('rgb(194, 165, 207)')
            colors.push('rgb(166, 219, 160)')
            power.push(keepMaxValueList[index][loop][0])
            power.push(keepMaxValueList[index][loop][1] * (-1))
          } else if (keepMaxValueList[index][loop][0] == keepMaxValueList[index][loop][1]) {
            colors.push('rgb(255, 255, 255)')
            colors.push('rgb(255, 255, 255)')
            power.push(keepMaxValueList[index][loop][0])
            power.push(keepMaxValueList[index][loop][0])
          } else {
            colors.push('rgb(166, 219, 160)')
            colors.push('rgb(194, 165, 207)')
            power.push(keepMaxValueList[index][loop][1])
            power.push(keepMaxValueList[index][loop][0] * (-1))
          }
        }
        xAxisComplete.push(xAxis)
        widthsComplete.push(widths)
        colorsComplete.push(colors)
        powerComplete.push(power)
      }

      const max = Math.max(...[].concat(...powerComplete));

      const min = Math.min(...[].concat(...powerComplete));

      var diffScale = 165

      for (let index = 0; index < orderedFeatures.length; index++) { 
        if (index == 0) {
          data.push({
            x: xAxisComplete[index],
            y: powerComplete[index],
            width: widthsComplete[index],
            name: orderedFeatures[index],
            marker: {
              color: colorsComplete[index],
              line: {
                color: 'black',
                width: 1
              }
            },
            hoverinfo:"y",
            xaxis: 'x',
            yaxis: 'y',
            type: 'bar'
          })
        } else {
          data.push({
            x: xAxisComplete[index],
            y: powerComplete[index],
            width: widthsComplete[index],
            name: orderedFeatures[index],
            marker: {
              color: colorsComplete[index],
              line: {
                color: 'black',
                width: 0.5
              }
            },
            hoverinfo:"y",
            bargap: 0,
            barmode: 'relative',
            xaxis: 'x'+(index+1),
            yaxis: 'y'+(index+1),
            type: 'bar'
          })
        }

      }

      var groupAnnotations = []
      var axesTogether = {};
      for (let index = 0; index < orderedFeatures.length; index++) {
        if (index == 0) {
          groupAnnotations.push(
            {
              x: 0.5,
              y: max,
              xref: 'x',
              yref: 'y',
              text: orderedFeatures[index],
              showarrow: false,
              font: {
                  family: 'sans-serif',
                  size: 16,
                  color: '#000000'
                },
            })
            axesTogether['yaxis'] = {
              autorange: false,
              range: [min, max],
              type: 'linear'
            }
          axesTogether['xaxis'] = {        
            showgrid: false,
            showline: false,
            autorange: false,
            range: [0, 1]
          }
        } else {
          groupAnnotations.push(
            {
              x: 0.5,
              y: max,
              xref: 'x'+(index+1),
              yref: 'y'+(index+1),
              text: orderedFeatures[index],
              showarrow: false,
              font: {
                  family: 'sans-serif',
                  size: 16,
                  color: '#000000'
                },
            })
          axesTogether['yaxis'+(index+1)] = {
            autorange: false,
            range: [min, max],
            type: 'linear'
          }
          axesTogether['xaxis'+(index+1)] = {        
            showgrid: false,
            showline: false,
            autorange: false,
            range: [0, 1]
          }
        }
        if (this.hoveredTestIns == -1 || this.hoveredTestIns == -2 || this.hoveredTestIns == -3) {
        } else {
            if (index == 0) {
             groupAnnotations.push({
                x: Object.values(X_testRounded[orderedFeatures[index]])[TestIDs[this.hoveredTestIns]],
                y: max,
                xref: 'x',
                yref: 'y',
                text: '',
                showarrow: true,
                arrowcolor: '#e31a1c',
                arrowhead: 0,
                ax: 0,
                ay: diffScale
              }) 
            } else {
              groupAnnotations.push({
                x: Object.values(X_testRounded[orderedFeatures[index]])[TestIDs[this.hoveredTestIns]],
                y: max,
                xref: 'x'+(index+1),
                yref: 'y'+(index+1),
                text: '',
                showarrow: true,
                arrowcolor: '#e31a1c',
                arrowhead: 0,
                ax: 0,
                ay: diffScale
              })        
            }
        }

        if (this.hoveredTrainIns == -2) {
        } else {
            if (index == 0) {
             groupAnnotations.push({
                x: Object.values(X_trainRounded[orderedFeatures[index]])[this.hoveredTrainIns],
                y: max,
                xref: 'x',
                yref: 'y',
                text: '',
                showarrow: true,
                arrowcolor: '#e31a1c',
                arrowhead: 0,
                ax: 0,
                ay: diffScale
              }) 
            } else {
              groupAnnotations.push({
                x: Object.values(X_trainRounded[orderedFeatures[index]])[this.hoveredTrainIns],
                y: max,
                xref: 'x'+(index+1),
                yref: 'y'+(index+1),
                text: '',
                showarrow: true,
                arrowcolor: '#e31a1c',
                arrowhead: 0,
                ax: 0,
                ay: diffScale
              })        
            }
        }

      }

      var layout = {
        grid: {rows: 1, columns: orderedFeatures.length, pattern: 'independent'},
        width: widthAll,
        height: heightAll,
        showlegend: false,
        
        annotations: groupAnnotations,
        hovermode: 'closest',
        margin: {
          t: 6,
          r: 15,
          l: 30,
          b: 30
        },
      }

      for (let index = 0; index < Object.keys(axesTogether).length; index++) {
        var eachObjectKey = Object.keys(axesTogether)[index]
        var eachObjectValue = Object.values(axesTogether)[index]
        layout[eachObjectKey] = eachObjectValue
      }

      var graphDiv = document.getElementById('summary')
      Plotly.newPlot(graphDiv, data, layout, config);

      var getTextsX = d3.select('#summary').selectAll('.xaxislayer-above text')
      getTextsX.each(function(d, i) {
        d3.select(this).text(xAxisNames[i]);
      });

      graphDiv.on('plotly_hover', function(data){
        var pointsSel = data.points[0]
        var gatherInitialStep = 0
        var gatherLastStep = 0
        for (let index = 0; index < pointsSel.data.width.length; index++) {
          if (index % 2 != 0) {
            if (index < pointsSel.pointIndex) {
              gatherInitialStep = gatherInitialStep + pointsSel.data.width[index]
            }
          }
        }
        gatherLastStep = gatherInitialStep + pointsSel.data.width[pointsSel.pointIndex]
        EventBus.$emit('SendHighlightingFeat', pointsSel.curveNumber)
        EventBus.$emit('SendHighlightingInit', gatherInitialStep)
        EventBus.$emit('SendHighlightingLast', gatherLastStep)
      })
      .on('plotly_unhover', function(data){
        EventBus.$emit('SendHighlightingZero')
      });

      if (this.hoveredTestIns == -2) {
        // $("#gridVisualization").empty();
        // var canvas = d3v5.select("#gridVisualization");
        // canvas.selectAll("*").remove();
        $("#gridVisualization").empty();
        var canvas = d3v5.select("#gridVisualization");
        canvas.selectAll("*").remove();
        this.TopLeftRight()
        this.callUpdatingFun()
        // $("#gridVisualization").empty();
        // var canvas = d3v5.select("#gridVisualization");
        // canvas.selectAll("*").remove();
        this.gridFun()
      } 
      else if (this.hoveredTestIns == -3) {
      // $("#gridVisualization").empty();
      // var canvas = d3v5.select("#gridVisualization");
      // canvas.selectAll("*").remove();
      //   $("#gridVisualization").empty();
      //   var canvas = d3v5.select("#gridVisualization");
      //   canvas.selectAll("*").remove();
        $("#gridVisualization").empty();
        var canvas = d3v5.select("#gridVisualization");
        canvas.selectAll("*").remove();
        this.callUpdatingFun()
        this.TopLeftRight()
        this.gridFun()
      } else {
      }
      
    },
    TopLeftRight () {

      var geometry = this.gridData[11]

      var depth = geometry[1]
      var size = geometry[2]

      //var cellSize = Math.floor((width - wid * groupSpacing) / size) - cellSpacing;
      var cellSize = 4

      Plotly.purge('statisticsProba')

      var selectedModel = this.gridData[15]
      var colorsComplete = this.gridData[16]
      var barChartGap = (size * cellSize) / 108
      var orderedFeatures = this.gridData[13]
      var dataStacked = []

      var axesTogetherStacked = {};

      for (let index = 0; index < selectedModel.length; index++) {
        if (index == 0){
          for (let ins = 0; ins < selectedModel[index].length; ins++) {
            var isEmpty = Object.keys(selectedModel[index][ins]).length === 0;
            axesTogetherStacked['yaxis'] = {
                'visible': false,
              }
            if(isEmpty){
              axesTogetherStacked['xaxis'] = {
                'visible': false,
              }
            } else {
              axesTogetherStacked['xaxis'] = 
              {        
                autorange: false,
                showgrid: false,
                showline: false,
                showticklabels: false,
                range: [0, 1],
                type: 'linear'
              }
            }
            dataStacked.push({
              x: [0,1],
              y: selectedModel[index][ins],
              width: [0.08, 0.08],
              marker: {
              color: colorsComplete[index][ins],
                // line: {
                //   color: 'black',
                //   width: 1
                // }
              },
              xaxis: 'x',
              yaxis: 'y',
              type: 'bar'
            })
          }
        } else {
          for (let ins = 0; ins < selectedModel[index].length; ins++) {
            var isEmpty = Object.keys(selectedModel[index][ins]).length === 0;
            axesTogetherStacked['yaxis'+(index+1)] = {
                'visible': false,
              }
            if(isEmpty){
              axesTogetherStacked['xaxis'+(index+1)] = {
                'visible': false,
              }
            } else {
              axesTogetherStacked['xaxis'+(index+1)] = 
              {        
                autorange: false,
                showgrid: false,
                showline: false,
                showticklabels: false,
                range: [0, 1],
                type: 'linear'
              }
            }
            dataStacked.push({
              x: [0,1],
              y: selectedModel[index][ins],
              width: [0.08, 0.08],
              marker: {
              color: colorsComplete[index][ins],
                // line: {
                //   color: 'black',
                //   width: 1
                // }
              },
              xaxis: 'x'+(index+1),
              yaxis: 'y'+(index+1),
              type: 'bar'
            })
          }
        }
      }

      var layoutStacked = {
        showlegend: false,
        plot_bgcolor: "rgba(0,0,0,0)",
        paper_bgcolor: "rgba(0,0,0,0)",
        grid: {rows: depth, columns: orderedFeatures.length, xgap: 0.18, ygap: 0.09, pattern: 'independent'},
        barmode: 'stack',
        height: 657,
        bargap: barChartGap,
        margin: {
          t: 13,
          r: 10,
          l: 25,
          b: 50
        },
      }

      for (let index = 0; index < Object.keys(axesTogetherStacked).length; index++) {
        var eachObjectKeyStacked = Object.keys(axesTogetherStacked)[index]
        var eachObjectValueStacked = Object.values(axesTogetherStacked)[index]
        layoutStacked[eachObjectKeyStacked] = eachObjectValueStacked
      }

      Plotly.newPlot('statisticsProba', dataStacked, layoutStacked);

      Plotly.purge('statisticsAlpha')

      var axesTogetherAlpha = {};

      var alphaModel = this.gridData[17]
      var maxAlpha = 0
      alphaModel.forEach(element => {
        if (element.length != 0)
          if (element[0] >= maxAlpha)
            maxAlpha = element[0]
      });
      maxAlpha = maxAlpha / 2
       var dataBar = []
       for (let index = 0; index < alphaModel.length; index++) {
        var temp = []
        temp.push(alphaModel[index] / 2)
        temp.push((alphaModel[index] / 2) * (-1))
        if (index == 0){
            if (temp[0] === 0 && temp[1] === -0) {
              axesTogetherAlpha['yaxis'] = {
                'visible': false,
              }
              axesTogetherAlpha['xaxis'] = {
                'visible': false,
              }
            } else {
              axesTogetherAlpha['yaxis'] = 
              {        
                autorange: false,
                showgrid: false,
                showline: false,
                showticklabels: false,
                range: [0, 1],
                type: 'linear'
              }
              axesTogetherAlpha['xaxis'] = 
              {        
                autorange: false,
                showgrid: false,
                showline: false,
                showticklabels: false,
                range: [-maxAlpha, maxAlpha],
                type: 'linear'
              }
            }
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
        } else {
          if (temp[0] === 0 && temp[1] === -0) {
              axesTogetherAlpha['yaxis'+(index+1)] = {
                'visible': false,
              }
              axesTogetherAlpha['xaxis'+(index+1)] = {
                'visible': false,
              }
            } else {
              axesTogetherAlpha['yaxis'+(index+1)] = 
              {        
                autorange: false,
                showgrid: false,
                showline: false,
                showticklabels: false,
                range: [0, 1],
                type: 'linear'
              }
              axesTogetherAlpha['xaxis'+(index+1)] = 
              {        
                autorange: false,
                showgrid: false,
                showline: false,
                showticklabels: false,
                range: [-maxAlpha, maxAlpha],
                type: 'linear'
              }
            }
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
              xaxis: 'x'+(index+1),
              yaxis: 'y'+(index+1),
              type: 'bar'
            })
        }
      }

      var layoutBar = {
        showlegend: false,
        barmode: 'relative',
        plot_bgcolor: "rgba(0,0,0,0)",
        paper_bgcolor: "rgba(0,0,0,0)",
        height: 657,
        grid: {rows: depth, columns: orderedFeatures.length, xgap: 0.18, ygap: 0.1, pattern: 'independent'},
        //barmode: 'stack',
        //bargap: barChartGap,
        margin: {
          t: 15,
          r: 10,
          l: 25,
          b: 50
        },
      };

      for (let index = 0; index < Object.keys(axesTogetherAlpha).length; index++) {
        var eachObjectKeyBar = Object.keys(axesTogetherAlpha)[index]
        var eachObjectValueBar = Object.values(axesTogetherAlpha)[index]
        layoutBar[eachObjectKeyBar] = eachObjectValueBar
      }

      Plotly.newPlot('statisticsAlpha', dataBar, layoutBar);    

      Plotly.purge('statisticsDec')

      var geometry = this.gridData[11]

      var depth = geometry[1]

      var orderedFeatures = this.gridData[13]
      var splitModelDec = this.gridData[18]
      var colorModelDec = this.gridData[19]
      var annotationsAll = []

      var layoutStackDec = {
        showlegend: false,
        plot_bgcolor: "rgba(0,0,0,0)",
        paper_bgcolor: "rgba(0,0,0,0)",
        grid: {rows: depth, columns: orderedFeatures.length, xgap: 0.18, ygap: 0.1, pattern: 'independent'},
        barmode: 'stack',
        height: 657,
        margin: {
          t: 16,
          r: 10,
          l: 25,
          b: 50
        },
      };

      var dataStackDec = []

       for (let index = 0; index < splitModelDec.length; index++) {
          if (index == 0){
                annotationsAll.push({
                  x: 0.5,
                  y: 0.5,
                  xref: 'x',
                  yref: 'y',
                  text: '1',
                  showarrow: false,
                  font: {
                      family: 'sans-serif',
                      size: 64,
                      color: '#000000'
                    },
                })
                dataStackDec.push({
                  x: splitModelDec[index],
                  y: [0, 0],
                  width: [0.08, 0.08],
                  marker: {
                    color: colorModelDec[index],
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
                if (splitModelDec[index].length == 0) {
                  layoutStackDec['xaxis'] = {
                    visible: false
                  }
                layoutStackDec['yaxis'] = {
                  visible: false
                  }   
                } else {
                  if (index == this.checkingRule) {
                    layoutStackDec['xaxis'] = {
                      autorange: false,
                      mirror:true,
                      showgrid: false,
                      showline:true,
                      linecolor: '#e31a1c',
                      linewidth: 3,
                      range: [0, 1],
                      type: 'linear'
                      }
                    layoutStackDec['yaxis'] = {
                      autorange: false,
                      mirror:true,
                      showgrid: false,
                      showline:true,
                      linecolor: '#e31a1c',
                      linewidth: 3,
                      range: [0, 1],
                      tickmode: 'array',
                      tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
                      ticktext: ['0', '20', '40', '60', '80', '100'],
                      type: 'linear'
                    }               
                  } else {
                    layoutStackDec['xaxis'] = {
                      autorange: false,
                      showgrid: false,
                      showline: false,
                      range: [0, 1],
                      type: 'linear'
                    }
                    layoutStackDec['yaxis'] = {
                      autorange: false,
                      showgrid: false,
                      showline: false,
                      range: [0, 1],
                      tickmode: 'array',
                      tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
                      ticktext: ['0', '20', '40', '60', '80', '100'],
                      type: 'linear'
                    }
                  }
                }   
          } else {
                dataStackDec.push({
                  x: splitModelDec[index],
                  y: [0, 0],
                  width: [0.08, 0.08],
                  marker: {
                    color: colorModelDec[index],
                    // line: {
                    //   color: 'black',
                    //   width: 1
                    // }
                  },
                  orientation: 'h',
                  xaxis: 'x'+(index+1),
                  yaxis: 'y'+(index+1),
                  type: 'bar'
                })
                if (splitModelDec[index].length == 0) {
                  layoutStackDec['xaxis'+(index+1)] = {
                    visible: false
                  }
                  layoutStackDec['yaxis'+(index+1)] = {
                    visible: false
                  }           
                } else {
                  annotationsAll.push({
                    x: 0.5,
                    y: 0.5,
                    xref: 'x'+(index+1),
                    yref: 'y'+(index+1),
                    text: ''+(index+1)+'',
                    showarrow: false,
                    font: {
                        family: 'sans-serif',
                        size: 64,
                        color: '#000000'
                      },
                  })
                  if (index == this.checkingRule) {
                    layoutStackDec['xaxis'+(index+1)] = {
                      autorange: false,
                      mirror:true,
                      showgrid: false,
                      showline:true,
                      linecolor: '#e31a1c',
                      linewidth: 3,
                      range: [0, 1],
                      type: 'linear'
                      }
                    layoutStackDec['yaxis'+(index+1)] = {
                      autorange: false,
                      mirror:true,
                      showgrid: false,
                      showline:true,
                      linecolor: '#e31a1c',
                      linewidth: 3,
                      range: [0, 1],
                      tickmode: 'array',
                      tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
                      ticktext: ['0', '20', '40', '60', '80', '100'],
                      type: 'linear'
                    }
                  } else {
                    layoutStackDec['xaxis'+(index+1)] = {
                      autorange: false,
                      showgrid: false,
                      showline: false,
                      range: [0, 1],
                      type: 'linear'
                    }
                    layoutStackDec['yaxis'+(index+1)] = {
                      autorange: false,
                      showgrid: false,
                      showline: false,
                      range: [0, 1],
                      tickmode: 'array',
                      tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
                      ticktext: ['0', '20', '40', '60', '80', '100'],
                      type: 'linear'
                    }
                  }
                }
          }
        }
      
      layoutStackDec['annotations'] = annotationsAll

      Plotly.newPlot('statisticsDec', dataStackDec, layoutStackDec);  

    },
    callUpdatingFun () {   

      if (!this.firstTimeGl) {
        var canvas = d3v5.select("#gridVisualization");
        canvas.selectAll("*").remove();
      } 

      Plotly.purge('statisticsDecHighlighting')

      var geometry = this.gridData[11]

      var depth = geometry[1]

      var orderedFeatures = this.gridData[13]
      var orderedFeaturesLength = orderedFeatures.length
      var splitModelDec = this.gridData[18]

      var dataStackDecHighlighting = []
      var gatherPositionsX = []
      var gatherColors = []
      var gatherOutlines = []
      var gatherY = []
      var gatherWidth = []
      var gatherEachPossibleRule = []

      var perFeatureUnique = this.gridData[12]
        if (this.highlightFeat != -1) {
          for (let index = 0; index < perFeatureUnique[this.highlightFeat].length; index++) {
            if (index == (perFeatureUnique[this.highlightFeat].length - 1)) {

            } else {
              gatherEachPossibleRule.push(this.highlightFeat+(orderedFeaturesLength*index))
              if (perFeatureUnique[this.highlightFeat][index] == this.highlightInit && perFeatureUnique[this.highlightFeat][index+1] == this.highlightLast) {
                gatherPositionsX.push(perFeatureUnique[this.highlightFeat][index+1]-perFeatureUnique[this.highlightFeat][index])
                gatherY.push(0.02)
                gatherWidth.push(0.03)
                gatherColors.push('rgba(255,255,255,0)')
                gatherOutlines.push('rgba(227,26,28,1)')
              } else {
                gatherPositionsX.push(perFeatureUnique[this.highlightFeat][index+1]-perFeatureUnique[this.highlightFeat][index])
                gatherY.push(0.02)
                gatherWidth.push(0.03)
                gatherColors.push('rgba(255,255,255,0)')
                gatherOutlines.push('rgba(255,255,255,0)')
              }
            }
        }
      }

      // if (this.highlightInit == 0) {
      //   gatherPositionsX = [this.highlightLast,(1-this.highlightLast)]
      //   gatherY = [0.02,0.02]
      //   gatherWidth = [0.03, 0.03]
      //   gatherColors = ['rgba(255,255,255,0)','rgba(255,255,255,0)']
      //   gatherOutlines = ['rgba(227,26,28,1)','rgba(255,255,255,0)']
      // } else if (this.highlightLast == 1) {
      //   gatherPositionsX = [this.highlightInit,(1-this.highlightInit)]
      //   gatherY = [0.02,0.02]
      //   gatherWidth = [0.03, 0.03]
      //   gatherColors = ['rgba(255,255,255,0)','rgba(255,255,255,0)']
      //   gatherOutlines = ['rgba(255,255,255,0)','rgba(227,26,28,1)']
      // } else {
      //   gatherPositionsX = [this.highlightInit,(this.highlightLast-this.highlightInit),(1-this.highlightLast)]
      //   gatherY = [0.02,0.02,0.02]
      //   gatherWidth = [0.03, 0.03, 0.03]
      //   gatherColors = ['rgba(255,255,255,0)','rgba(255,255,255,0)','rgba(255,255,255,0)']
      //   gatherOutlines = ['rgba(255,255,255,0)','rgba(227,26,28,1)','rgba(255,255,255,0)']
      // }

      var layoutStackDecHighlighting = {
        showlegend: false,
        plot_bgcolor: "rgba(0,0,0,0)",
        paper_bgcolor: "rgba(0,0,0,0)",
        grid: {rows: depth, columns: orderedFeatures.length, xgap: 0.18, ygap: 0.1, pattern: 'independent'},
        barmode: 'stack',
        //height: 450,
        //bargap: barChartGap,
        margin: {
          t: 16,
          r: 10,
          l: 25,
          b: 50
        },
      };

       for (let index = 0; index < splitModelDec.length; index++) {
          if(this.highlightInit == -1) {
              if (index == 0){
                if (splitModelDec[index].length == 0) {
                  layoutStackDecHighlighting['xaxis'] = {
                    visible: false
                  }
                  layoutStackDecHighlighting['yaxis'] = {
                  visible: false
                  }   
                } else {
                  if (index == this.checkingRule) {
                    layoutStackDecHighlighting['xaxis'] = {
                      autorange: false,
                      mirror:true,
                      showgrid: false,
                      showline:true,
                      linecolor: '#e31a1c',
                      linewidth: 3,
                      range: [0, 1],
                      type: 'linear'
                      }
                      layoutStackDecHighlighting['yaxis'] = {
                      autorange: false,
                      mirror:true,
                      showgrid: false,
                      showline:true,
                      linecolor: '#e31a1c',
                      linewidth: 3,
                      range: [0, 1],
                      tickmode: 'array',
                      tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
                      ticktext: ['0', '20', '40', '60', '80', '100'],
                      type: 'linear'
                    }               
                  } else {
                    layoutStackDecHighlighting['xaxis'] = {
                      autorange: false,
                      showgrid: false,
                      showline: false,
                      range: [0, 1],
                      type: 'linear'
                    }
                    layoutStackDecHighlighting['yaxis'] = {
                      autorange: false,
                      showgrid: false,
                      showline: false,
                      range: [0, 1],
                      tickmode: 'array',
                      tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
                      ticktext: ['0', '20', '40', '60', '80', '100'],
                      type: 'linear'
                    }
                  }
                }   
                dataStackDecHighlighting.push({
                  x: [0, 0],
                  y: [0, 0],
                  width: [0, 0],
                  marker: {
                    color: ['rgba(255,255,255,0)','rgba(255,255,255,0)'],
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
          } else {
            if (splitModelDec[index].length == 0) {
              layoutStackDecHighlighting['xaxis'+(index+1)] = {
                    visible: false
                  }
                  layoutStackDecHighlighting['yaxis'+(index+1)] = {
                  visible: false
                  }   
                } else {
                  if (index == this.checkingRule) {
                    layoutStackDecHighlighting['xaxis'+(index+1)] = {
                      autorange: false,
                      mirror:true,
                      showgrid: false,
                      showline:true,
                      linecolor: '#e31a1c',
                      linewidth: 3,
                      range: [0, 1],
                      type: 'linear'
                      }
                      layoutStackDecHighlighting['yaxis'+(index+1)] = {
                      autorange: false,
                      mirror:true,
                      showgrid: false,
                      showline:true,
                      linecolor: '#e31a1c',
                      linewidth: 3,
                      range: [0, 1],
                      tickmode: 'array',
                      tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
                      ticktext: ['0', '20', '40', '60', '80', '100'],
                      type: 'linear'
                    }               
                  } else {
                    layoutStackDecHighlighting['xaxis'+(index+1)] = {
                      autorange: false,
                      showgrid: false,
                      showline: false,
                      range: [0, 1],
                      type: 'linear'
                    }
                    layoutStackDecHighlighting['yaxis'+(index+1)] = {
                      autorange: false,
                      showgrid: false,
                      showline: false,
                      range: [0, 1],
                      tickmode: 'array',
                      tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
                      ticktext: ['0', '20', '40', '60', '80', '100'],
                      type: 'linear'
                    }
                  }
                }   
                dataStackDecHighlighting.push({
                  x: [0, 0],
                  y: [0, 0],
                  width: [0, 0],
                  marker: {
                    color: ['rgba(255,255,255,0)','rgba(255,255,255,0)'],
                    // line: {
                    //   color: 'black',
                    //   width: 1
                    // }
                  },
                  orientation: 'h',
                  xaxis: 'x'+(index+1),
                  yaxis: 'y'+(index+1),
                  type: 'bar'
                })
            }
          } else {
            //if (splitModelDec[index].length != 0) {
                if (gatherEachPossibleRule.includes(index)) {
                  if (splitModelDec[index].length == 0) {
                    if (index == 0) {
                    layoutStackDecHighlighting['xaxis'] = {
                      visible: false
                    }
                    layoutStackDecHighlighting['yaxis'] = {
                    visible: false
                    }   
                    dataStackDecHighlighting.push({
                        x: [0, 0],
                        y: [0, 0],
                        width: [0, 0],
                        marker: {
                          color: ['rgba(255,255,255,0)','rgba(255,255,255,0)'],
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
                    } else {
                      layoutStackDecHighlighting['xaxis'+(index+1)] = {
                      visible: false
                    }
                    layoutStackDecHighlighting['yaxis'+(index+1)] = {
                    visible: false
                    }   
                    dataStackDecHighlighting.push({
                        x: [0, 0],
                        y: [0, 0],
                        width: [0, 0],
                        marker: {
                          color: ['rgba(255,255,255,0)','rgba(255,255,255,0)'],
                          // line: {
                          //   color: 'black',
                          //   width: 1
                          // }
                        },
                        orientation: 'h',
                        xaxis: 'x'+(index+1),
                        yaxis: 'y'+(index+1),
                        type: 'bar'
                      })
                    }
                } else {
                  if (index == 0){
                    if (index == this.checkingRule) {
                      layoutStackDecHighlighting['xaxis'] = {
                        autorange: false,
                        mirror:true,
                        showgrid: false,
                        showline:true,
                        linecolor: '#e31a1c',
                        linewidth: 3,
                        range: [0, 1],
                        type: 'linear'
                        }
                        layoutStackDecHighlighting['yaxis'] = {
                        autorange: false,
                        mirror:true,
                        showgrid: false,
                        showline:true,
                        linecolor: '#e31a1c',
                        linewidth: 3,
                        range: [0, 1],
                        tickmode: 'array',
                        tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
                        ticktext: ['0', '20', '40', '60', '80', '100'],
                        type: 'linear'
                      }               
                    } else {
                      layoutStackDecHighlighting['xaxis'] = {
                        autorange: false,
                        showgrid: false,
                        showline: false,
                        range: [0, 1],
                        type: 'linear'
                      }
                      layoutStackDecHighlighting['yaxis'] = {
                        autorange: false,
                        showgrid: false,
                        showline: false,
                        range: [0, 1],
                        tickmode: 'array',
                        tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
                        ticktext: ['0', '20', '40', '60', '80', '100'],
                        type: 'linear'
                      }
                    }
                  } else {
                    if (index == this.checkingRule) {
                      layoutStackDecHighlighting['xaxis'+(index+1)] = {
                        autorange: false,
                        mirror:true,
                        showgrid: false,
                        showline:true,
                        linecolor: '#e31a1c',
                        linewidth: 3,
                        range: [0, 1],
                        type: 'linear'
                        }
                        layoutStackDecHighlighting['yaxis'+(index+1)] = {
                        autorange: false,
                        mirror:true,
                        showgrid: false,
                        showline:true,
                        linecolor: '#e31a1c',
                        linewidth: 3,
                        range: [0, 1],
                        tickmode: 'array',
                        tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
                        ticktext: ['0', '20', '40', '60', '80', '100'],
                        type: 'linear'
                      }               
                    } else {
                      layoutStackDecHighlighting['xaxis'+(index+1)] = {
                        autorange: false,
                        showgrid: false,
                        showline: false,
                        range: [0, 1],
                        type: 'linear'
                      }
                      layoutStackDecHighlighting['yaxis'+(index+1)] = {
                        autorange: false,
                        showgrid: false,
                        showline: false,
                        range: [0, 1],
                        tickmode: 'array',
                        tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
                        ticktext: ['0', '20', '40', '60', '80', '100'],
                        type: 'linear'
                      }
                    } 
                  }
                    //var gatherYDiv = gatherY.map(function(item) { return ((item*howManyTimes) + ((howManyTimes-1) * 0.05)) })
                    if (index == 0){
                      dataStackDecHighlighting.push({
                        x: gatherPositionsX,
                        y: gatherY,
                        width: gatherWidth,
                        marker: {
                          color: gatherColors,
                          line: {
                            color: gatherOutlines,
                            width: 3
                          }
                        },
                        orientation: 'h',
                        xaxis: 'x',
                        yaxis: 'y',
                        type: 'bar'
                      })
                  } else {
                        dataStackDecHighlighting.push({
                          x: gatherPositionsX,
                          y: gatherY,
                          width: gatherWidth,
                          marker: {
                            color: gatherColors,
                            line: {
                              color: gatherOutlines,
                              width: 3
                            }
                          },
                          orientation: 'h',
                          xaxis: 'x'+(index+1),
                          yaxis: 'y'+(index+1),
                          type: 'bar'
                        })
                    }
                  }   
                } else {
                  if (index == 0){
                    if (splitModelDec[index].length == 0) {
                  layoutStackDecHighlighting['xaxis'] = {
                    visible: false
                  }
                  layoutStackDecHighlighting['yaxis'] = {
                  visible: false
                  }   
                } else {
                  if (index == this.checkingRule) {
                    layoutStackDecHighlighting['xaxis'] = {
                      autorange: false,
                      mirror:true,
                      showgrid: false,
                      showline:true,
                      linecolor: '#e31a1c',
                      linewidth: 3,
                      range: [0, 1],
                      type: 'linear'
                      }
                      layoutStackDecHighlighting['yaxis'] = {
                      autorange: false,
                      mirror:true,
                      showgrid: false,
                      showline:true,
                      linecolor: '#e31a1c',
                      linewidth: 3,
                      range: [0, 1],
                      tickmode: 'array',
                      tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
                      ticktext: ['0', '20', '40', '60', '80', '100'],
                      type: 'linear'
                    }               
                  } else {
                    layoutStackDecHighlighting['xaxis'] = {
                      autorange: false,
                      showgrid: false,
                      showline: false,
                      range: [0, 1],
                      type: 'linear'
                    }
                    layoutStackDecHighlighting['yaxis'] = {
                      autorange: false,
                      showgrid: false,
                      showline: false,
                      range: [0, 1],
                      tickmode: 'array',
                      tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
                      ticktext: ['0', '20', '40', '60', '80', '100'],
                      type: 'linear'
                    }
                  }
                }   
                    dataStackDecHighlighting.push({
                      x: [0, 0],
                      y: [0, 0],
                      width: [0, 0],
                      marker: {
                        color: ['rgba(255,255,255,0)','rgba(255,255,255,0)'],
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
              } else {
                if (splitModelDec[index].length == 0) {
                  layoutStackDecHighlighting['xaxis'+(index+1)] = {
                    visible: false
                  }
                  layoutStackDecHighlighting['yaxis'+(index+1)] = {
                  visible: false
                  }   
                } else {
                  if (index == this.checkingRule) {
                    layoutStackDecHighlighting['xaxis'+(index+1)] = {
                      autorange: false,
                      mirror:true,
                      showgrid: false,
                      showline:true,
                      linecolor: '#e31a1c',
                      linewidth: 3,
                      range: [0, 1],
                      type: 'linear'
                      }
                      layoutStackDecHighlighting['yaxis'+(index+1)] = {
                      autorange: false,
                      mirror:true,
                      showgrid: false,
                      showline:true,
                      linecolor: '#e31a1c',
                      linewidth: 3,
                      range: [0, 1],
                      tickmode: 'array',
                      tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
                      ticktext: ['0', '20', '40', '60', '80', '100'],
                      type: 'linear'
                    }               
                  } else {
                    layoutStackDecHighlighting['xaxis'+(index+1)] = {
                      autorange: false,
                      showgrid: false,
                      showline: false,
                      range: [0, 1],
                      type: 'linear'
                    }
                    layoutStackDecHighlighting['yaxis'+(index+1)] = {
                      autorange: false,
                      showgrid: false,
                      showline: false,
                      range: [0, 1],
                      tickmode: 'array',
                      tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
                      ticktext: ['0', '20', '40', '60', '80', '100'],
                      type: 'linear'
                    }
                  }
                }   
                    dataStackDecHighlighting.push({
                      x: [0, 0],
                      y: [0, 0],
                      width: [0, 0],
                      marker: {
                        color: ['rgba(255,255,255,0)','rgba(255,255,255,0)'],
                        // line: {
                        //   color: 'black',
                        //   width: 1
                        // }
                      },
                      orientation: 'h',
                      xaxis: 'x'+(index+1),
                      yaxis: 'y'+(index+1),
                      type: 'bar'
                    })
                }
                }
            //}
          }
        }

      const config = {
        displayModeBar: false, // this is the line that hides the bar.
      };

      Plotly.newPlot('statisticsDecHighlighting', dataStackDecHighlighting, layoutStackDecHighlighting, config); 

      if (!this.firstTimeGl) {
       this.gridFun()
      }

      this.firstTimeGl = false

    },
    gridFun () {

      var hoveredTrainInsLoc = this.hoveredTrainIns

      var gridDecisions = JSON.parse(JSON.parse(this.gridData[10]))
      var geometry = this.gridData[11]

      var wid = geometry[0]
      var depth = geometry[1]
      var size = geometry[2]
      var gridSize = geometry[3]
      var combineDepthGridSize = depth * gridSize

      var margin = {top: 32, right: 0, bottom: 0, left: 40} //32 for breastC

      var width = 1654 - margin.left;
      var height = 647 - margin.top; 
      //var height = 2000 - margin.top; 

      // settings for a grid with 40 cells in a row and 2x5 cells in a group
      var groupSpacing = (width-gridSize)/(wid-1) + 6; // + 5 default
      var groupSpacingPoint = (width/(wid-1)) * 0.18 + 5;
      var cellSpacing = 2;
      var cellSize = Math.floor((width - ((wid-1) * groupSpacingPoint) - ((size-1) * cellSpacing * wid)) / (size*wid));
      if (cellSize == 8) {
        cellSize = 7.5
      }
      //var cellSize = 4

      var log = console.log.bind(console);
      var dir = console.dir.bind(console);
      var replace = function(string) { return string.replace(/[^a-z0-9]/gi,""); };


      // === Set up canvas === //

      var data = [];
      var colorScale;


      // var mainCanvas = d3.select('#container')
      //   .append('canvas')
      //   .classed('mainCanvas', true)
      //   .attr('width', width)
      //   .attr('height', height);

  // new -----------------------------------------------------

      var hiddenCanvas = d3v5.select('#gridVisualization')
        .append('canvas')
        .classed('hiddenCanvas', true)
        .attr('width', width + margin.left)
        .attr('height', height + margin.top) 

      var colourToNode = {}; // map to track the colour of nodes

      // function to create new colours for the picking

      var nextCol = 1;

  // new -----------------------------------------------------


      // === Load and prepare the data === //

      for (var i = 0; i < Object.values(gridDecisions.value).length; i++) {
        data.push({ value: Object.values(gridDecisions.value)[i], color: Object.values(gridDecisions.colorGT)[i], class: Object.values(gridDecisions.ID)[i]})
      }
      
      // === Bind data to custom elements === //

      var customBase = document.createElement('custom');
      var custom = d3v5.select(customBase); // this is our svg replacement

      // === First call === //

      databind(data); // ...then update the databind function
      
      var t = d3v5.timer(function(elapsed) {
        draw(hiddenCanvas, false); // <--- new insert arguments
        if (elapsed > 4000) t.stop();
      }); // start a timer that runs the draw function for 300 ms (this needs to be higher than the transition in the databind function)


      // === Bind and draw functions === //

      function databind(data) {

        colorScale = d3v5.scaleSequential(d3v5.interpolateSpectral).domain(d3v5.extent(data, function(d) { return d.value; }));
        
        var join = custom.selectAll('custom.rect')
          .data(data);

        var loop = -1
        var count = -1
        var enterSel = join.enter()
          .append('custom')
          .attr('class', function(d, i) {
            return 'rect'
          })
          .attr('y', function(d, i) {
            if (i % combineDepthGridSize == 0) {
              loop = loop + 1
            }
            i = i - (combineDepthGridSize*loop)
            var x0 = Math.floor(i / gridSize) % size, x1 = Math.floor(i % size);
            if (i % gridSize == 0) {
              count = count + 1
              return (groupSpacing - 25) * x0 + (cellSpacing + cellSize) * (x1 + x0 * 10); // - 5 default
            } else {
              if (count == -1) {
                return groupSpacing * x0 + (cellSpacing + cellSize) * (x1 + x0 * 10);
              } else {
                if (count == gridSize) {
                  count = -1
                }
                return (groupSpacing - 25) * x0 + (cellSpacing + cellSize) * (x1 + x0 * 10);
              }
            }
          })
          .attr('x', function(d, i) {
            var y0 = Math.floor(i / combineDepthGridSize), y1 = Math.floor(i % gridSize / size);
            return groupSpacing * y0 + (cellSpacing + cellSize) * (y1 + y0 * 10);

          })
          .attr('width', 0)
          .attr('height', 0);

        join
          .merge(enterSel)
          .transition()
          .attr('width', cellSize)
          .attr('height', cellSize)

          // new -----------------------------------------------------

          .attr('fillStyleHidden', function(d) { 
            if (d.class == hoveredTrainInsLoc) {
              return 'rgb(227, 26, 28)';
            } else {
              return d.color;
            }
				});

          // new -----------------------------------------------------

    


        var exitSel = join.exit()
          .transition()
          .attr('width', 0)
          .attr('height', 0)
          .remove();

      } // databind()

      var firstTime = true

      // === Draw canvas === //

      function draw(canvas, hidden) { // <---- new arguments

        // build context
        var context = canvas.node().getContext('2d');

        if (firstTime)
        context.translate(margin.left, margin.top)

        context.globalCompositeOperation = 'destination-over';

        firstTime = false

        // clear canvas
        context.clearRect(0, 0, width, height);

        
        // draw each individual custom element with their properties
        
        var elements = custom.selectAll('custom.rect') // this is the same as the join variable, but used here to draw
        
        elements.each(function(d,i) { // for each virtual/custom element...

          var node = d3v5.select(this);

          context.fillStyle = hidden ? node.attr('fillStyleHidden') : node.attr('fillStyleHidden'); // <--- new: node colour depends on the canvas we draw 
          context.fillRect(node.attr('x'), node.attr('y'), node.attr('width'), node.attr('height'))

        });

      } // draw()


      // === Listeners/handlers === //

  // new -----------------------------------------------------



      // d3.select('.mainCanvas').on('mousemove', function() {

      //   // draw the hiddenCanvas
      //   draw(hiddenCanvas, true); 

      //   // get mousePositions from the main canvas
      //   var mouseX = d3.event.layerX || d3.event.offsetX;
      //   var mouseY = d3.event.layerY || d3.event.offsetY;
        

      //   // get the toolbox for the hidden canvas  
      //   var hiddenCtx = hiddenCanvas.node().getContext('2d');

      //   // Now to pick the colours from where our mouse is then stringify it in a way our map-object can read it
      //   var col = hiddenCtx.getImageData(mouseX, mouseY, 1, 1).data;
      //   var colKey = 'rgb(' + col[0] + ',' + col[1] + ',' + col[2] + ')';
        
      //   // get the data from our map !
      //   var nodeData = colourToNode[colKey];
        
      //   log(nodeData);


      //   if (nodeData) {

      //     // Show the tooltip only when there is nodeData found by the mouse

      //     d3.select('#tooltip')
      //       .style('opacity', 0.8)
      //       .style('top', d3.event.pageY + 5 + 'px')
      //       .style('left', d3.event.pageX + 5 + 'px')
      //       .html(nodeData.value);

      //   } else {

      //     // Hide the tooltip when there our mouse doesn't find nodeData

      //     d3.select('#tooltip')
      //       .style('opacity', 0);

      //   }

      // }); // canvas listener/handler 
      // Plotly.purge('statisticsProba')
      // Plotly.purge('statisticsAlpha')
      // Plotly.purge('statisticsDec')
      // Plotly.purge('statisticsDecHighlighting')
    }
  },
  mounted () {
    EventBus.$on('sendWhichRuleIsActive', data => {
      this.checkingRule = data })
    EventBus.$on('hoveringTestSampleReset', data => {
      this.hoveredTestIns = -2 
      this.firstTimeGl = true
    })
    EventBus.$on('hoveringTrainingReset', data => {
      this.hoveredTrainIns = -2
      this.firstTimeGl = true
      $("#gridVisualization").empty();
      var canvas = d3v5.select("#gridVisualization");
      canvas.selectAll("*").remove();
      this.gridFun()
    })
    EventBus.$on('hoveringTrainingReset', this.summarize)
    EventBus.$on('hoveringTrainingSample', data => {
      this.hoveredTrainIns = data
      $("#gridVisualization").empty();
      var canvas = d3v5.select("#gridVisualization");
      canvas.selectAll("*").remove();
      this.gridFun()
    })
    EventBus.$on('hoveringTrainingSample', this.summarize)
    EventBus.$on('hoveringTestSample', data => {
      this.hoveredTestIns = data })
    EventBus.$on('hoveringTestSample', this.summarize)
    EventBus.$on('updateFirstTimeGL', data => {
      this.firstTimeGl = true })
    EventBus.$on('emittedEventCallingSurrogateData', data => {
      this.gridData = data })
    EventBus.$on('SendHighlightingZero', data => {
      this.highlightFeat = -1
      this.highlightInit = -1
      this.highlightLast = -1
    })
    EventBus.$on('SendHighlightingFeat', data => {
      this.highlightFeat = data })
    EventBus.$on('SendHighlightingInit', data => {
      this.highlightInit = data })
    EventBus.$on('SendHighlightingLast', data => {
      this.highlightLast = data })
    EventBus.$on('emittedEventCallingSurrogateDataLater', this.summarize)
    EventBus.$on('SendHighlightingLast', this.callUpdatingFun)
    EventBus.$on('SendHighlightingZero', this.callUpdatingFun)
  },
}
</script>

<style>
    #LegendBehaviorSummary {
      height: 15px;
      width: 100%;
    }

		body {
			font-family: 'Open Sans', sans-serif;
		}
		
		canvas {
			border:  1px dotted #ccc;
      z-index: 2;
		}
		
		#text-explain {
			display: inline-block;
			font-size: 0.75em;
			margin-bottom: 1em;
		}

		.alert {
			color: tomato;
		}


/* new (in comparison to code w/o interactivty at:) ---- */
/* (http://blockbuilder.org/larsvers/d187337850d58a444082841c739985ca) */

		div#tooltip {
		  position: absolute;        
			display: inline-block;
			padding: 10px;
			font-family: 'Open Sans' sans-serif;
			color: #000;
		  background-color: #fff;
			border: 1px solid #999;
			border-radius: 2px;
		  pointer-events: none;
			opacity: 0;
			z-index: 1;
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
      z-index: 2;
    }
    .outer .above {
      z-index: 3;
    }

</style>