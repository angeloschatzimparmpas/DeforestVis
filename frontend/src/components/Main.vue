<!-- Main Visualization View -->

<template>
<body>
    <b-container fluid class="bv-example-row">
      <b-row>
        <b-col cols="6">
          <mdb-card style="margin-right:-9px">
            <mdb-card-header color="primary-color" tag="h5" class="text-center" ><DataSetSlider/></mdb-card-header>
            <mdb-card-body>
              <mdb-card-text class="text-left" style="font-size: 18.5px; min-height: 380px">
                <Information/>
              </mdb-card-text>
            </mdb-card-body>
          </mdb-card>
        </b-col>
          <b-col cols="6">
           <mdb-card style="margin-left:-9px">
            <mdb-card-header color="primary-color" tag="h5" class="text-center" style="background-color: #C0C0C0;">Test Set Results</mdb-card-header>
            <mdb-card-body>
              <mdb-card-text class="text-left" style="font-size: 18.5px; min-height: 380px">
                <Result/>
              </mdb-card-text>
            </mdb-card-body>
          </mdb-card>
          </b-col>
      </b-row>
      <b-row style="margin-top: 10px">
        <b-col cols="8">
          <mdb-card>
            <mdb-card-header  style=" z-index: 1" color="primary-color" tag="h5" class="text-center">
              Behavioral Model Summarization
              </mdb-card-header>
            <mdb-card-body>
              <mdb-card-text class="text-left" style="font-size: 18.5px; min-height: 886px">
                <Summary/>
              </mdb-card-text>
            </mdb-card-body>
          </mdb-card>
          </b-col>
           <b-col cols="4">
          <mdb-card style="margin-left:-18px">
            <mdb-card-header color="primary-color" tag="h5" class="text-center"><ControlRule/></mdb-card-header>
            <mdb-card-body>
              <mdb-card-text class="text-left" style="font-size: 18.5px; min-height: 230px">
                <Overriding/>
              </mdb-card-text>
            </mdb-card-body>
          </mdb-card>
          <div style="margin-top: 10px">
          <mdb-card style="margin-left:-18px">
            <mdb-card-header color="primary-color" tag="h5" class="text-center">Decisions Comparison</mdb-card-header>
            <mdb-card-body>
              <mdb-card-text class="text-left" style="font-size: 18.5px; min-height: 576px">
                <Comparison/>
              </mdb-card-text>
            </mdb-card-body>
          </mdb-card>
          </div>
          </b-col>
      </b-row>
    </b-container>
  </body>
</template>

<script>

import Vue from 'vue'
import DataSetSlider from './DataSetSlider.vue'
import Information from './Information.vue'
import Result from './Result.vue'
import Summary from './Summary.vue'
import Overriding from './Overriding.vue'
import ControlRule from './ControlRule.vue'
import Comparison from './Comparison.vue'
import axios from 'axios'
import { loadProgressBar } from 'axios-progress-bar'
import 'axios-progress-bar/dist/nprogress.css'
import 'bootstrap-css-only/css/bootstrap.min.css'
import { mdbCard, mdbCardBody, mdbCardText, mdbCardHeader } from 'mdbvue'
import { EventBus } from '../main.js'
import $ from 'jquery'; // <-to import jquery
import 'bootstrap';
import * as d3Base from 'd3'

// attach all d3 plugins to the d3 library
const d3 = Object.assign(d3Base)

export default Vue.extend({
  name: 'Main',
  components: {
    DataSetSlider,
    Information,
    Result,
    Summary,
    Overriding,
    ControlRule,
    Comparison,
    mdbCard,
    mdbCardBody,
    mdbCardHeader,
    mdbCardText
  },
  data () {
    return {
      RetrieveValueFile: 'diabetesC', // this is for the default data set
      reset: false,
      surrogateModelData: 0,
      updateUMAP: 0,
      checkingRule: -1,
      localRule: -1,
      thresholdRule: -1,
      recomputeRuleModelVar: -1,
    }
  },
  methods: {
    fileNameSend () {
      const path = `http://127.0.0.1:5000/data/ServerRequest`
      const postData = {
        fileName: this.RetrieveValueFile,
      }
      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.post(path, postData, axiosConfig)
      .then(response => {
        console.log('File name was sent successfully!')
        this.getModelsPerformanceFromBackend(true)
      })
      .catch(error => {
        console.log(error)
      })
    },
    getModelsPerformanceFromBackend (flagLoc) {
      const path = `http://localhost:5000/data/PerformanceForEachModel`

      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.get(path, axiosConfig)
        .then(response => {
          this.surrogateModelData = response.data.surrogateModelData
          if (flagLoc) {
            EventBus.$emit('emittedEventCallingInfo', this.surrogateModelData)
          }
          EventBus.$emit('emittedEventCallingUMAPFirst', this.surrogateModelData)
          EventBus.$emit('emittedEventCallingSurrogateData', this.surrogateModelData)
          EventBus.$emit('emittedEventCallingSurrogateDataLater')
          console.log('Server successfully sent all computed information!')
        })
        .catch(error => {
          console.log(error)
        })
    },
    Reset () {
      const path = `http://127.0.0.1:5000/data/Reset`
      this.reset = true
      const postData = {
        ClassifiersList: this.reset
      }
      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.post(path, postData, axiosConfig)
        .then(response => {
          console.log('The server side was reset! Done.')
          this.reset = false
          EventBus.$emit('resetViews')
          this.fileNameSend()
        })
        .catch(error => {
          console.log(error)
        })
    },
    updateUMAPDueToNewThreshold () {
      const path = `http://127.0.0.1:5000/data/updateUMAP`
      const postData = {
        localRule: this.localRule,
        checkingRule: this.checkingRule,
        thresholdRule: this.thresholdRule
      }
      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.post(path, postData, axiosConfig)
        .then(response => {
          console.log('The server updates UMAP! Done.')
          this.updateUMAP = response.data.updateUMAP
          EventBus.$emit('emitUpdateUMAP', this.updateUMAP)   
        })
        .catch(error => {
          console.log(error)
        })
    },
    recomputeModel () {
      const path = `http://127.0.0.1:5000/data/callRecomputeModel`
      const postData = {
        localRule: this.localRule,
        checkingRule: this.checkingRule,
        thresholdRule: this.thresholdRule
      }
      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.post(path, postData, axiosConfig)
        .then(response => {
          EventBus.$emit('resetFirstTime')
          EventBus.$emit('hoveringTestSampleReset')
          EventBus.$emit('hoveringTrainingReset')
          this.getModelsPerformanceFromBackend(false)
          console.log('The server updates the entire model! Done.')
        })
        .catch(error => {
          console.log(error)
        })
      },
    recomputeRuleModel () {
      const path = `http://127.0.0.1:5000/data/callRecomputeRuleModel`
      const postData = {
        recomputeRuleModelVar: this.recomputeRuleModelVar,
      }
      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.post(path, postData, axiosConfig)
        .then(response => {
          EventBus.$emit('resetFirstTime')
          EventBus.$emit('hoveringTestSampleReset')
          EventBus.$emit('hoveringTrainingReset')
          this.getModelsPerformanceFromBackend(false)
          console.log('The server updates the entire model! Done.')
        })
        .catch(error => {
          console.log(error)
        })
    }
  },
  created () {

    // does the browser support the Navigation Timing API?
    if (window.performance) {
        console.info("window.performance is supported");
    }
    // do something based on the navigation type...
    if(performance.navigation.type === 1) {
        console.info("TYPE_RELOAD");
        this.Reset();
    }
  },
  mounted() {

    loadProgressBar()
    window.onbeforeunload = function(e) {
      return 'Dialog text here.'
    }
    $(window).on("unload", function(e) {
      alert('Handler for .unload() called.');
    })

    //Prevent double click to search for a word. 
    document.addEventListener('mousedown', function (event) {
      if (event.detail > 1) {
      event.preventDefault();
      }
    }, false);
    EventBus.$on('sendWhichRuleIsActiveLocally', data => { this.localRule = data })
    EventBus.$on('sendWhichRuleIsActive', data => { this.checkingRule = data })
    EventBus.$on('sendNewUpdatedThreshold', data => { this.thresholdRule = data })
    EventBus.$on('callUpdateOfUMAP', this.updateUMAPDueToNewThreshold)
    EventBus.$on('executeMainAgain', this.recomputeModel)
    EventBus.$on('updateRuleModel',  data => { this.recomputeRuleModelVar = data })
    EventBus.$on('updateRuleModel',  this.recomputeRuleModel)
  },
})
</script>

<style lang="scss">

#nprogress .bar {
background: red !important;
}

#nprogress .peg {
box-shadow: 0 0 10px red, 0 0 5px red !important;
}

#nprogress .spinner-icon {
border-top-color: red !important;
border-left-color: red !important;
}

body {
  font-family: 'Helvetica', 'Arial', sans-serif !important;
  left: 0px;
  right: 0px;
  top: 0px;
  bottom: 0px;
  margin-top: -4px !important;
  //overflow: hidden !important; // remove scrolling
}

.card-body {
   padding: 0.60rem !important;
}

hr {
  margin-top: 1rem;
  margin-bottom: 1rem;
  border: 0;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
}

@import './../assets/w3.css';
</style>