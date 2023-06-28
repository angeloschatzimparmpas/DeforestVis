<template>
<div>
  <div id="Scatterplot"></div>
</div>
</template>

<script>
import { EventBus } from '../main.js'
import * as Plotly from 'plotly.js'

export default {
  name: 'Comparison',
  data () {
    return {
      FinalResultsGeneral: 0,
      updatedUMAP: 0,
      currentIndex: -1
    }
  },
  methods: {
    ScatterplotView() {

        Plotly.purge('Scatterplot')

        var colors = ['rgb(90, 174, 97)', 'rgb(153, 112, 171)']

        var UMAPData = this.FinalResultsGeneral[20]
        var order = Object.values(this.FinalResultsGeneral[13])
        var XTrainRounded = JSON.parse(this.FinalResultsGeneral[21])
        var keepRoundingLevel = this.FinalResultsGeneral[45]
        var yTrain = this.FinalResultsGeneral[22]
        var symbolUMAPBorder = this.FinalResultsGeneral[37]

        var symbolUMAPBorderList = symbolUMAPBorder[this.currentIndex]

        var colorsGathered = []
        for (let index = 0; index < yTrain.length; index++) {
          colorsGathered.push(colors[yTrain[index]])
        }

        var UMAPDataABx = []
        var UMAPDataABy = []
        var DataGeneral = []
        var InfoProcessing = []
        for (let index = 0; index < UMAPData[0].length; index++) {
          UMAPDataABx.push(UMAPData[0][index])
          UMAPDataABy.push(UMAPData[1][index])
          var gatherEachRounded = {}
          for (let k = 0; k < order.length; k++) {
            var position = Object.keys(XTrainRounded).indexOf(order[k])
            gatherEachRounded[order[k]] = Object.values(Object.values(XTrainRounded)[position])[index].toFixed((keepRoundingLevel-1))
          }
          InfoProcessing.push(JSON.stringify(gatherEachRounded).replace(/,/gi, '<br>'))
        }

        DataGeneral.push({
          type: 'scatter',
          mode: 'markers',
          x: UMAPDataABx,
          y: UMAPDataABy,
          hovertemplate: 
                "%{text}<br><br>" +
                "<extra></extra>",
          text: InfoProcessing,
          marker: {
            symbol: 'square',
            size: 10,
            color: symbolUMAPBorderList,
              line: {
                color: 'rgb(0,0,0)',
                width: 1
              }
          }
        })

        var width = 820
        var height = 570

        var layout = {
          xaxis: {
              visible: false,
              autorange: true
          },
          yaxis: {
              visible: false,
              autorange: true
          },
          font: { family: 'Helvetica', size: 14, color: '#000000' },
          autosize: false,
          showlegend: false,
          width: width,
          height: height,
          // dragmode: 'lasso',
          hovermode: "closest",
          hoverlabel: { bgcolor: "#FFF" },
          legend: {orientation: 'h', x: 0.15, y: 0},
          margin: {
            l: 0,
            r: 0,
            b: 0,
            t: 0,
            pad: 0
          },
        }
     
        var graphDiv = document.getElementById('Scatterplot')
        var config = {scrollZoom: true, displaylogo: false, showLink: false, showSendToCloud: false, modeBarButtonsToRemove: ['toImage', 'toggleSpikelines', 'autoScale2d', 'hoverClosestGl2d','hoverCompareCartesian','select2d','hoverClosestCartesian'], responsive: true}

        Plotly.newPlot(graphDiv, DataGeneral, layout, config)

        graphDiv.on('plotly_hover', function(data){
        var activeTrainID = data.points[0].pointNumber
        // if (activeTestID == 0 || activeTestID == 1 || activeTestID == 2 || ((activeTestID-3) == SplittingPoint) || ((activeTestID-3) == (SplittingPoint+1))) {
        //   EventBus.$emit('hoveringTestSample', -1)
        // } else {
        //   if ((activeTestID-3) < SplittingPoint) {
        //     activeTestID = activeTestID - 3
        //   } else {
        //     activeTestID = activeTestID - 5
        //   }
        EventBus.$emit('hoveringTrainingSample', activeTrainID)
        // }
      });

      graphDiv.on('plotly_unhover', function(data){
        EventBus.$emit('hoveringTrainingReset')
      });

    },
    ScatterplotViewUpdate() {

        Plotly.purge('Scatterplot')

        var colors = ['rgb(90, 174, 97)', 'rgb(153, 112, 171)']

        var UMAPData = this.updatedUMAP[3]
        console.log(UMAPData)
        var UMAPDataUpdated = this.updatedUMAP[0]
        console.log(UMAPDataUpdated)
        var order = Object.values(this.FinalResultsGeneral[13])
        var XTrainRounded = JSON.parse(this.FinalResultsGeneral[21])
        var keepRoundingLevel = this.FinalResultsGeneral[45]
        var yTrain = this.updatedUMAP[1]
        var symbolUMAPBorder = this.updatedUMAP[2]

        var colorsGathered = []
        for (let index = 0; index < yTrain.length; index++) {
          colorsGathered.push(colors[yTrain[index]])
        }

        var DataGeneral = []
        var InfoProcessing = []
        for (let index = 0; index < UMAPDataUpdated[0].length; index++) {
          var UMAPDataUpdatedx = []
          var UMAPDataUpdatedy = []
          var symbolUMAPBorderList = []
          var gatherWidth = []
          UMAPDataUpdatedx.push(UMAPDataUpdated[0][index])
          UMAPDataUpdatedy.push(UMAPDataUpdated[1][index])
          UMAPDataUpdatedx.push(UMAPData[0][index])
          UMAPDataUpdatedy.push(UMAPData[1][index])
          symbolUMAPBorderList.push(symbolUMAPBorder[index])
          gatherWidth.push(1)
          symbolUMAPBorderList.push('rgba(255,255,255,0)')
          gatherWidth.push(0)

          var gatherEachRounded = {}
          for (let k = 0; k < order.length; k++) {
            var position = Object.keys(XTrainRounded).indexOf(order[k])
            gatherEachRounded[order[k]] = Object.values(Object.values(XTrainRounded)[position])[index].toFixed((keepRoundingLevel-1))
          }
          InfoProcessing.push(JSON.stringify(gatherEachRounded).replace(/,/gi, '<br>'))

          DataGeneral.push({
          type: 'scatter',
          mode: 'lines+markers',
          x: UMAPDataUpdatedx,
          y: UMAPDataUpdatedy,
          hovertemplate: 
                "%{text}<br><br>" +
                "<extra></extra>",
          text: InfoProcessing,
          name: 'Decision Paths',
          marker: {
            symbol: 'square',
            size: 10,
            color: symbolUMAPBorderList,
            line: {
                color: 'rgb(0,0,0)',
                width: gatherWidth
              }
          },
          line: {
            color: 'rgba(211,211,211,0.3)', // 15 for breastC
            width: 1
          }
        })  
          // var gatherEachRounded = {}
          // for (let k = 0; k < order.length; k++) {
          //   var position = Object.keys(XTrainRounded).indexOf(order[k])
          //   gatherEachRounded[order[k]] = Object.values(Object.values(XTrainRounded)[position])[index].toFixed((keepRoundingLevel-1))
          // }
          // InfoProcessing.push(JSON.stringify(gatherEachRounded).replace(/,/gi, '<br>'))
        }    

        var width = 820
        var height = 576

        var layout = {
          xaxis: {
              visible: false,
              autorange: true
          },
          yaxis: {
              visible: false,
              autorange: true
          },
          font: { family: 'Helvetica', size: 14, color: '#000000' },
          autosize: false,
          showlegend: false,
          width: width,
          height: height,
          // dragmode: 'lasso',
          hovermode: "closest",
          hoverlabel: { bgcolor: "#FFF" },
          legend: {orientation: 'h', x: 0.15, y: 0},
          margin: {
            l: 0,
            r: 0,
            b: 0,
            t: 0,
            pad: 0
          },
        }
     
        var graphDiv = document.getElementById('Scatterplot')
        var config = {scrollZoom: true, displaylogo: false, showLink: false, showSendToCloud: false, modeBarButtonsToRemove: ['toImage', 'toggleSpikelines', 'autoScale2d', 'hoverClosestGl2d','hoverCompareCartesian','select2d','hoverClosestCartesian'], responsive: true}

        Plotly.newPlot(graphDiv, DataGeneral, layout, config)

        graphDiv.on('plotly_hover', function(data){
        var activeTrainID = data.points[0].pointNumber
        // if (activeTestID == 0 || activeTestID == 1 || activeTestID == 2 || ((activeTestID-3) == SplittingPoint) || ((activeTestID-3) == (SplittingPoint+1))) {
        //   EventBus.$emit('hoveringTestSample', -1)
        // } else {
        //   if ((activeTestID-3) < SplittingPoint) {
        //     activeTestID = activeTestID - 3
        //   } else {
        //     activeTestID = activeTestID - 5
        //   }
        EventBus.$emit('hoveringTrainingSample', activeTrainID)
        // }
      });

      graphDiv.on('plotly_unhover', function(data){
        EventBus.$emit('hoveringTrainingReset')
      });

    }
  },
  mounted () {
     EventBus.$on('emittedEventCallingUMAPFirst', data => {
      this.FinalResultsGeneral = data })
    EventBus.$on('emitUpdateUMAP', data => {
      this.updatedUMAP = data })
    EventBus.$on('emitUpdateUMAP', this.ScatterplotViewUpdate)
    EventBus.$on('currentIndexForUMAP', data => { this.currentIndex = data })
    EventBus.$on('currentIndexForUMAP', this.ScatterplotView)
  },
}
</script>