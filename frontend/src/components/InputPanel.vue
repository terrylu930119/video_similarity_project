<script setup>
// =============== 影片比對輸入面板組件 ===============
// 功能：提供影片比對任務的輸入介面，包括參考影片、目標影片列表、參數設定等

import { computed, unref } from 'vue'
import ChipList from './ChipList.vue'      // 影片標籤列表組件
import ProgressBar from './ProgressBar.vue' // 進度條組件

// ──────────────── 組件屬性定義 ────────────────
const props = defineProps({
  refUrl:    { type: [String, Object], default: '' },     // 參考影片的URL
  listInput: { type: [String, Object], default: '' },     // 要比對的影片列表輸入文字
  chips:     { type: [Array,  Object], default: () => [] }, // 已加入的影片標籤陣列
  interval:  { type: [String, Object], default: 'auto' }, // 抽幀間隔設定
  keep:      { type: [Boolean,Object], default: false },  // 是否保留中間檔案
  canStart:  { type: [Boolean,Object], default: false },  // 是否可以開始比對
  loading:   { type: [Boolean,Object], default: false },  // 是否正在處理中
  overall:   { type: Object, default: () => ({ done:0, total:0, percent:0 }) }, // 整體進度資訊
  labelFor:  { type: Function, required: true }            // 用於生成影片標籤的函數
})

// ──────────────── 事件定義 ────────────────
// v-model 對應事件必須使用 camelCase 命名
const emit = defineEmits([
  'update:refUrl','update:listInput','update:chips','update:interval','update:keep',
  'add-chips','remove-chip','clear-all','submit'
])

// ──────────────── 計算屬性 ────────────────
// 將可能是 ref 的值解包成純值，確保響應式更新
const refUrlV    = computed(() => unref(props.refUrl) ?? '')           // 參考影片URL
const listInputV = computed(() => unref(props.listInput) ?? '')        // 影片列表輸入
const chipsV     = computed(() => unref(props.chips) ?? [])            // 影片標籤陣列
const intervalV  = computed(() => unref(props.interval) ?? 'auto')     // 抽幀間隔
const keepV      = computed(() => !!unref(props.keep))                 // 保留檔案開關
const canStartV  = computed(() => !!unref(props.canStart))             // 可開始比對狀態
const loadingV   = computed(() => !!unref(props.loading))              // 載入狀態
const overallV   = computed(() => props.overall || { done:0,total:0,percent:0 }) // 整體進度
const chipsCount = computed(() => chipsV.value.length)                 // 已加入的影片數量
</script>

<template>
  <!-- 輸入面板標題 -->
  <h3>輸入與參數</h3>

  <!-- 參考影片輸入區域 -->
  <label>參考影片（Reference URL）</label>
  <input
    :value="refUrlV"
    @input="(e)=>{ console.log('[InputPanel] update:refUrl', e.target.value); emit('update:refUrl', e.target.value.trim()) }"
    type="text" placeholder="https://youtu.be/...." />

  <!-- 要比對的影片列表輸入區域 -->
  <label style="margin-top:14px">要比對的影片（多個）</label>
  <textarea
    :value="listInputV"
    @input="(e)=>{ console.log('[InputPanel] update:listInput', e.target.value); emit('update:listInput', e.target.value) }"
    placeholder="可多行貼上，逗號或空白自動分割。"></textarea>

  <!-- 影片加入控制區域 -->
  <div class="row" style="align-items:center;gap:10px">
    <button type="button" class="btn" @click="()=>{ console.log('[InputPanel] add-chips'); emit('add-chips') }">加入</button>
    <span class="hint">已加入 <strong>{{ chipsCount }}</strong> 支影片（自動去重、可刪除）</span>
  </div>

  <!-- 影片標籤列表顯示 -->
  <ChipList
    :items="chipsV"
    :label-for="props.labelFor"
    @remove="(i) => emit('remove-chip', i)"
  />

  <!-- 抽幀間隔設定區域 -->
  <div class="row" style="margin-top:14px">
    <div style="min-width:200px;flex:1">
      <label>幀間隔</label>
      <select :value="intervalV" @change="(e)=>{ console.log('[InputPanel] update:interval', e.target.value); emit('update:interval', e.target.value) }">
        <option value="auto">自動</option>
        <option value="0.5">0.5 秒</option>
        <option value="1">1 秒</option>
        <option value="2">2 秒</option>
        <option value="5">5 秒</option>
      </select>
    </div>
    <div style="flex:1"></div>
  </div>

  <!-- 底部控制區域：保留檔案選項 + 操作按鈕 -->
  <div class="row" style="justify-content:space-between;margin-top:10px">
    <!-- 保留中間檔案選項 -->
    <label class="checkbox">
      <input type="checkbox" :checked="keepV" @change="(e)=>{ console.log('[InputPanel] update:keep', e.target.checked); emit('update:keep', e.target.checked) }" />
      保留中間檔案
    </label>

    <!-- 操作按鈕組 -->
    <div class="btns">
      <button type="button" class="btn" @click="()=>{ console.log('[InputPanel] clear-all'); emit('clear-all') }">清除</button>
      <button
        type="button"
        class="btn primary"
        :disabled="!canStartV || loadingV"
        @click="()=>{ console.log('[InputPanel] submit'); emit('submit') }"
      >
        {{ loadingV ? '處理中…' : '開始比對' }}
      </button>
    </div>
  </div>

  <!-- 整體進度顯示區域 -->
  <div v-if="overallV.total" style="margin-top:16px">
    <div class="row" style="justify-content:space-between;align-items:center;margin-bottom:6px">
      <div class="tiny">整體進度｜已完成 {{ overallV.done }} / {{ overallV.total }}</div>
      <div class="tiny">{{ overallV.percent }}%</div>
    </div>
    <!-- 進度條顯示整體完成百分比 -->
    <ProgressBar :value="overallV.percent" />
  </div>
</template>
