<template>
    <b-row style="padding-top: 7px; margin-top: -8px; margin-bottom:-5px">
      <b-col cols="4">
      </b-col>
      <b-col cols="3">
        Rule Overriding
      </b-col>
      <b-col cols="5">
        {{ action }}
        <span>
          <button btn-dark id="ApplyChange" v-on:click="ApplyChange">
          <font-awesome-icon icon="check" /> Threshold Change </button>
        </span>
      </b-col>
    </b-row>
</template>

<script>
import { EventBus } from '../main.js'
import * as Plotly from 'plotly.js'

export default {
  name: 'ControlRule',
  data () {
    return {
      action: 'Action: ',
    }
  },
  methods: {
    ApplyChange () {
      EventBus.$emit('executeMainAgain')
    },
  },
  mounted () {

  },
}
</script>