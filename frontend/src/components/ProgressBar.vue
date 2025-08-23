<script setup>
// =============== 進度條組件 ===============
// 功能：顯示任務進度的視覺化組件，支援平滑動畫和不同尺寸

import { computed, unref, watch, onMounted, onUnmounted } from 'vue'
import { useCompare } from '@/composables/useCompare'

// ──────────────── 組件屬性定義 ────────────────
const props = defineProps({
  value: { type:[Number,Object], default: 0 },             // 進度值（0-100）
  small: { type:[Boolean,Object], default: false },        // 是否使用小尺寸樣式
  smooth: { type:[Boolean,Object], default: true },        // 是否啟用平滑動畫
  duration: { type:[Number,Object], default: 0.5 }         // 動畫持續時間（秒）
})

// ──────────────── 使用組合式函數 ────────────────
// 從 useCompare 中獲取進度條的顯示邏輯和更新方法
const { displayValue, updateProgressBar } = useCompare()

// ──────────────── 計算屬性 ────────────────
// 進度值處理：確保在 0-100 範圍內並四捨五入
const v = computed(() => {
  const n = Number(unref(props.value) ?? 0)
  return Math.max(0, Math.min(100, Math.round(n)))
})

// 樣式相關的計算屬性
const isSmall = computed(() => !!unref(props.small))       // 小尺寸樣式
const isSmooth = computed(() => !!unref(props.smooth))     // 平滑動畫開關
const animDuration = computed(() => Number(unref(props.duration))) // 動畫時長

// ──────────────── 監聽器與生命週期 ────────────────
// 監聽進度值變化，自動更新進度條顯示
watch(v, (newValue, oldValue) => {
  updateProgressBar(newValue, isSmooth.value, animDuration.value)
}, { immediate: true })

// 組件掛載時初始化進度條
onMounted(() => {
  updateProgressBar(v.value, isSmooth.value, animDuration.value)
})
</script>

<template>
  <!-- 進度條容器：根據 small 屬性決定高度 -->
  <div class="progress" :style="isSmall ? 'height:8px' : 'height:10px'">
    <!-- 進度條填充部分：寬度根據 displayValue 動態調整 -->
    <div class="bar" :style="{ width: displayValue + '%' }"></div>
  </div>
</template>

