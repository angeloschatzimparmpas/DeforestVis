<template>
    <b-row style="padding-top: 7px; margin-top: -8px; margin-bottom:-8.5px">
      <b-col cols="4">
        <label id="data" for="param-dataset" data-toggle="tooltip" data-placement="right" title="Tip: use one of the data sets already provided or upload a new file.">{{ dataset }}</label>
        <select id="selectFile" @change="selectDataSet()">
            <option value="breastC.csv" selected>Breast Cancer</option>
            <option value="diabetesC.csv" >Indian Diabetes</option>
            <option value="HeartC.csv" >Heart Disease</option>
        </select>
      </b-col>
      <b-col cols="4">
        Surrogate Model Selection
      </b-col>
      <b-col cols="4">
        <label id="mode" for="param-dataset" data-toggle="tooltip" data-placement="right">{{ encoding }}</label>
        <select id="selectProfile" @change="changeMode()">
          <option value="performant">Performance</option>
          <option value="unique" >Uniqueness</option>
        </select>
      </b-col>
    </b-row>
</template>

<script>
import { EventBus } from '../main.js'
import * as d3Base from 'd3'

// attach all d3 plugins to the d3 library
const d3 = Object.assign(d3Base)

export default {
  name: 'DataSetSlider',
  data () {
    return {
      defaultDataSet: '', // default value for the first data set
      dataset: 'Data Set',
      encoding: 'Visual Encoding: ',
      modeLoc: ''
    }
  },
  methods: {
    changeMode () {
      const mode = document.getElementById('selectProfile')
      this.modeLoc = mode.options[mode.selectedIndex].value
      EventBus.$emit('SendModeChange', this.modeLoc)
    },
    selectDataSet () {   
      const fileName = document.getElementById('selectFile')
      this.defaultDataSet = fileName.options[fileName.selectedIndex].value
      this.defaultDataSet = this.defaultDataSet.split('.')[0]

      this.dataset = "Data set"
      d3.select("#data").select("input").remove(); // Remove the selection field.
      EventBus.$emit('SendToServerDataSetConfirmation', this.defaultDataSet)
    },
    reset () {
      EventBus.$emit('reset')
      EventBus.$emit('alternateFlagLock')
    },
  },
  mounted () {
  },
}
</script>
