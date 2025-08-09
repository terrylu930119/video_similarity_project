<script setup>
import { computed, unref } from 'vue'
import ProgressBar from './ProgressBar.vue'
import LogViewer from './LogViewer.vue'

const props = defineProps({
  tasks: { type:[Array,Object], required: true },
  running: { type:[Boolean,Object], default: false }
})
const emit = defineEmits(['cancel','stop-all'])

const tasksV = computed(() => unref(props.tasks) ?? [])
const isCancellable = (t) => {
  const s = (t.status || '').split(' ')[0]
  return ['佇列中','待處理','下載中','轉錄中','抽幀中','音訊比對','畫面比對','文本比對','比對中'].includes(s)
}
</script>

<template>
  <div class="row" style="justify-content:space-between;align-items:center">
    <h3 style="margin:0">任務佇列</h3>
    <div class="sticky-actions">
      <button class="btn" @click="emit('stop-all')" :disabled="!tasksV.length">停止全部</button>
    </div>
  </div>

  <div class="task-scroll">
    <div v-if="!tasksV.length" class="hint">尚無任務</div>

    <div v-for="t in tasksV" :key="t.id || t.url" class="queue-item">
      <div class="q-head">
        <div class="q-url" :title="t.url">{{ t.display || t.url }}</div>
        <div class="q-right tiny">{{ t.status }}</div>
      </div>

      <ProgressBar :value="t.progress || 0" small />

      <div class="q-actions">
        <button class="btn small" @click="t.showLog = !t.showLog">{{ t.showLog ? '隱藏 log' : '顯示 log' }}</button>
        <button class="btn small" :disabled="!isCancellable(t)" @click="emit('cancel', t)">取消</button>
      </div>

      <LogViewer v-if="t.showLog" :lines="t.log" />
    </div>
  </div>
</template>