<script setup>
// =============== 影片相似度比對系統 - 主應用程式組件 ===============
// 功能：整合所有子組件，管理整體狀態，提供完整的影片比對功能

import { nextTick } from 'vue'
import InputPanel from '@/components/InputPanel.vue'      // 輸入面板組件：影片輸入和參數設定
import QueuePanel from '@/components/QueuePanel.vue'      // 佇列面板組件：任務管理和進度追蹤
import ResultsGrid from '@/components/ResultsGrid.vue'    // 結果網格組件：比對結果顯示
import { useCompare } from '@/composables/useCompare'    // 核心邏輯組合式函數

// ──────────────── 使用核心邏輯組合式函數 ────────────────
// 直接解構：拿到的就是 ref 本體，v-model 會正確改 .value
const {
  // =============== 狀態變數（State） ===============
  refUrl, listInput, chips, interval, keep,    // 表單輸入狀態
  loading, running, tasks, results,            // 執行狀態和資料

  // =============== 計算屬性（Derived） ===============
  canStart, sortedResults, overallPercent, doneCount, totalCount,  // 派生狀態

  // =============== 方法函數（Methods） ===============
  addChips, removeChip, clearAll, submit, stopAll, cancelTask,    // 業務邏輯方法

  // =============== 工具函數（Utils） ===============
  labelFor,  // 影片標籤生成函數
} = useCompare()
</script>

<template>
  <!-- 主應用程式容器 -->
  <div class="container">
    <!-- 頁面標題區域 -->
    <h1 class="title">影音比對系統</h1>
    <p class="subtitle">輸入參考影片與多支待比對影片，設定參數後開始分析；可即時查看進度與結果。</p>

    <!-- 主要功能區域：雙欄佈局 -->
    <div class="grid grid-2">
      <!-- 左欄：輸入面板 -->
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
        <!-- 參考影片URL雙向綁定、影片列表輸入雙向綁定、影片標籤陣列雙向綁定、抽幀間隔雙向綁定、保留檔案選項雙向綁定、是否可以開始比對、載入狀態、整體進度資訊、影片標籤生成函數、添加影片標籤事件、移除影片標籤事件、清除所有資料事件、提交比對任務事件 -->
      </section>

      <!-- 右欄：任務佇列面板 -->
      <aside class="card queue-panel">
        <QueuePanel
          :tasks="tasks"
          :running="running"
          @cancel="cancelTask"
          @stop-all="stopAll"
        />
        <!-- 任務陣列、執行狀態、取消任務事件、停止所有任務事件 -->
      </aside>
    </div>

    <!-- 結果顯示區域：獨立卡片 -->
    <section class="card" style="margin-top:16px">
      <h3>比對結果</h3>
      <ResultsGrid 
        :results="sortedResults"
        :label-for="labelFor"
      />
      <!-- 排序後的比對結果、影片標籤生成函數 -->
    </section>
  </div>
</template>

<style scoped>
/* 此組件使用全域樣式，無需額外的scoped樣式 */
</style>
