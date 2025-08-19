<script setup>
import { computed, unref } from 'vue'
const props = defineProps({
  results: { type:[Array,Object], default: () => [] },
  labelFor: { type: Function, required: true }
})
const rows = computed(() => unref(props.results) ?? [])
const safeLabel = (fn, url) => { try { return fn(url || '') } catch { return '未知' } }
</script>

<template>
  <div class="results-grid" style="margin-top:12px">
    <div v-if="!rows.length" class="hint">尚未有完成結果</div>

    <div v-for="r in rows" :key="r.url || r.pair" class="result">
      <div class="meta">
        <a v-if="r.url" class="url" :href="r.url" target="_blank" rel="noopener">{{ safeLabel(labelFor, r.url) }}</a>
        <span v-else class="url">{{ r.pair || '（未知）' }}</span>
        <div class="score">{{ r.score ?? 0 }}%</div>
      </div>
      <div class="divider"></div>
      <div class="tiny">
        音訊 {{ r.audio ?? '0.00' }}｜畫面 {{ r.visual ?? '0.00' }}｜內容
        <template v-if="r.text_meaningful === false">
          — <span :title="r.text_status || '文本跳過'">（跳過）</span>
        </template>
        <template v-else>
          {{ r.text ?? '0.00' }}
        </template>
      </div>
    </div>
  </div>
</template>
