<script setup>
// =============== 任務佇列管理面板組件 ===============
// 功能：顯示和管理影片比對任務的執行狀態，包括進度追蹤、日誌查看、任務控制等

import { computed, unref } from 'vue'
import ProgressBar from './ProgressBar.vue'  // 進度條組件
import LogViewer from './LogViewer.vue'     // 日誌查看器組件

// ──────────────── 組件屬性定義 ────────────────
const props = defineProps({
  tasks: { type:[Array,Object], required: true },    // 任務陣列，包含所有比對任務
  running: { type:[Boolean,Object], default: false } // 是否有任務正在執行
})

// ──────────────── 事件定義 ────────────────
const emit = defineEmits(['cancel','stop-all'])

// ──────────────── 計算屬性 ────────────────
// 將任務陣列解包，確保響應式更新
const tasksV = computed(() => unref(props.tasks) ?? [])

// ──────────────── 任務狀態判斷 ────────────────
// 判斷任務是否可以取消：只有在特定狀態下的任務才能被取消
const isCancellable = (t) => {
  const s = (t.status || '').split(' ')[0]
  return ['佇列中','待處理','下載中','轉錄中','字幕解析','抽幀中','音訊比對','畫面比對','文本比對','比對中'].includes(s)
}
</script>

<template>
  <!-- 佇列面板標題和控制按鈕 -->
  <div class="row" style="justify-content:space-between;align-items:center">
    <h3 style="margin:0">任務佇列</h3>
    <!-- 停止全部任務的按鈕，只有在有任務時才啟用 -->
    <div class="sticky-actions">
      <button class="btn" @click="emit('stop-all')" :disabled="!tasksV.length">停止全部</button>
    </div>
  </div>

  <!-- 任務列表滾動容器 -->
  <div class="task-scroll">
    <!-- 無任務時的提示訊息 -->
    <div v-if="!tasksV.length" class="hint">尚無任務</div>

    <!-- 遍歷顯示每個任務項目 -->
    <div v-for="t in tasksV" :key="t.id || t.url" class="queue-item">
      <!-- 任務標題區域：顯示URL和狀態資訊 -->
      <div class="q-head">
        <!-- 任務URL顯示，使用 display 或原始 URL -->
        <div class="q-url" :title="t.url">{{ t.display || t.url }}</div>
        <!-- 右側狀態資訊：任務狀態、文本來源、跳過標記等 -->
        <div class="q-right tiny">
          <span class="status">{{ t.status }}</span>
          <!-- 文本來源標記：字幕或自動轉錄 -->
          <span v-if="t.text_source" class="tag" :title="t.text_source === 'subtitle' ? '來自字幕' : '自動轉錄'">
            {{ t.text_source === 'subtitle' ? '字幕' : 'ASR' }}
          </span>
          <!-- 文本跳過警告標記 -->
          <span v-if="t.text_skipped" class="tag warn" :title="t.text_status || '文本跳過'">文本跳過</span>
        </div>
      </div>

      <!-- 任務進度條：顯示當前執行進度 -->
      <ProgressBar :value="t.progress || 0" small />

      <!-- 任務操作按鈕區域 -->
      <div class="q-actions">
        <!-- 日誌顯示切換按鈕 -->
        <button class="btn small" @click="t.showLog = !t.showLog">
          {{ t.showLog ? '隱藏 log' : '顯示 log' }}
        </button>
        <!-- 取消任務按鈕：只有在可取消狀態下才啟用 -->
        <button class="btn small" :disabled="!isCancellable(t)" @click="emit('cancel', t)">取消</button>
      </div>

      <!-- 日誌查看器：當 showLog 為 true 時顯示任務執行日誌 -->
      <LogViewer v-if="t.showLog" :lines="t.log" />
    </div>
  </div>
</template>