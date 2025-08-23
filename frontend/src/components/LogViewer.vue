<script setup>
// =============== 日誌查看器組件 ===============
// 功能：顯示任務執行過程的日誌資訊，支援自動滾動到最新內容

import { onMounted, ref, watch } from 'vue'

// ──────────────── 組件屬性定義 ────────────────
const props = defineProps({ 
  lines: { type: Array, default: () => [] }  // 日誌行陣列
})

// ──────────────── 響應式引用 ────────────────
const el = ref(null)  // 日誌容器的 DOM 引用

// ──────────────── 功能函數 ────────────────
// 滾動到日誌底部，顯示最新的日誌內容
function scrollBottom(){ 
  if (el.value) el.value.scrollTop = el.value.scrollHeight 
}

// ──────────────── 生命週期與監聽器 ────────────────
// 組件掛載後自動滾動到底部
onMounted(scrollBottom)

// 監聽日誌行數變化，當有新日誌時自動滾動到底部
watch(() => props.lines.length, scrollBottom)
</script>

<template>
  <!-- 日誌顯示容器：使用 pre 標籤保持格式，支援滾動 -->
  <pre class="log" ref="el" aria-live="polite">{{ (lines || []).join('\n') }}</pre>
</template>