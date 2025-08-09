<script setup>
import { nextTick } from 'vue'
import InputPanel from '@/components/InputPanel.vue'
import QueuePanel from '@/components/QueuePanel.vue'
import ResultsGrid from '@/components/ResultsGrid.vue'
import { useCompare } from '@/composables/useCompare'

// 直接解構：拿到的就是 ref 本體，v-model 會正確改 .value
const {
  // state (refs)
  refUrl, listInput, chips, interval, keep,
  loading, running, tasks, results,

  // derived (computed)
  canStart, sortedResults, overallPercent, doneCount, totalCount,

  // methods
  addChips, removeChip, clearAll, submit, stopAll, cancelTask,

  // utils
  labelFor,
} = useCompare()
</script>

<template>
  <div class="container">
    <h1 class="title">影音比對系統</h1>
    <p class="subtitle">輸入參考影片與多支待比對影片，設定參數後開始分析；可即時查看進度與結果。</p>

    <div class="grid grid-2">
      <section class="card">
        <InputPanel
          v-model:refUrl="refUrl"
          v-model:listInput="listInput"
          v-model:chips="chips"
          v-model:interval="interval"
          v-model:keep="keep"
          :can-start="canStart"
          :loading="loading"
          :overall="{ done: doneCount, total: totalCount, percent: overallPercent }"
          :label-for="labelFor"
          @add-chips="addChips"
          @remove-chip="removeChip"
          @clear-all="clearAll"
          @submit="async () => { console.log('[App] submit fired'); await nextTick(); submit() }"
        />
      </section>

      <aside class="card queue-panel">
        <QueuePanel
          :tasks="tasks"
          :running="running"
          @cancel="cancelTask"
          @stop-all="stopAll"
        />
      </aside>
    </div>

    <section class="card" style="margin-top:16px">
      <h3>比對結果</h3>
      <ResultsGrid :results="sortedResults" :label-for="labelFor" />
    </section>
  </div>
</template>

<style scoped></style>
