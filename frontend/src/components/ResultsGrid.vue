<script setup>
// =============== 影片相似度比對結果網格顯示組件 ===============
// 功能：顯示影片相似度比對的詳細結果，包括分數、音訊、畫面、文本等維度

import { computed, unref } from 'vue'

// ──────────────── 組件屬性定義 ────────────────
const props = defineProps({
  results: { type:[Array,Object], default: () => [] },    // 比對結果陣列或物件
  labelFor: { type: Function, required: true }            // 用於生成URL標籤的函數
})

// ──────────────── 計算屬性 ────────────────
// 將結果轉換為陣列格式，確保資料結構一致
const rows = computed(() => unref(props.results) ?? [])

// 安全標籤生成函數：避免標籤生成錯誤導致組件崩潰
const safeLabel = (fn, url) => { 
  try { 
    return fn(url || '') 
  } catch { 
    return '未知' 
  } 
}

// ──────────────── 分數樣式分類 ────────────────
// 根據相似度分數決定顯示顏色類別，提供視覺化回饋
const getScoreClass = (score) => {
  const numScore = Number(score) || 0
  if (numScore < 50) return 'low'        // 低相似度：紅色
  if (numScore < 70) return 'medium'     // 中等相似度：黃色
  if (numScore < 85) return 'high'       // 高相似度：綠色
  return 'very-high'                      // 極高相似度：深綠色
}
</script>

<template>
  <!-- 結果網格容器 -->
  <div class="results-grid" style="margin-top:12px">
    <!-- 無結果時的提示訊息 -->
    <div v-if="!rows.length" class="hint">尚未有完成結果</div>

    <!-- 遍歷顯示每個比對結果 -->
    <div v-for="r in rows" :key="r.url || r.pair" class="result">
      <!-- 結果元資料區域：URL/配對名稱 + 相似度分數 -->
      <div class="meta">
        <!-- 如果有URL則顯示為可點擊連結，否則顯示配對名稱 -->
        <a v-if="r.url" class="url" :href="r.url" target="_blank" rel="noopener">
          {{ safeLabel(labelFor, r.url) }}
        </a>
        <span v-else class="url">{{ r.pair || '（未知）' }}</span>
        <!-- 相似度分數顯示，根據分數套用對應的樣式類別 -->
        <div class="score" :data-score="getScoreClass(r.score)">{{ r.score ?? 0 }}%</div>
      </div>
      
      <!-- 分隔線 -->
      <div class="divider"></div>
      
      <!-- 詳細分數資訊：音訊、畫面、文本三個維度 -->
      <div class="tiny">
        音訊 {{ r.audio ?? '0.00' }}｜畫面 {{ r.visual ?? '0.00' }}｜內容
        <!-- 文本比對狀態處理：如果文本被跳過則顯示跳過原因 -->
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
