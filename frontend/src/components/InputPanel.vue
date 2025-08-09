<script setup>
import { computed, unref } from 'vue'
import ChipList from './ChipList.vue'
import ProgressBar from './ProgressBar.vue'

const props = defineProps({
  refUrl:    { type: [String, Object], default: '' },
  listInput: { type: [String, Object], default: '' },
  chips:     { type: [Array,  Object], default: () => [] },
  interval:  { type: [String, Object], default: 'auto' },
  keep:      { type: [Boolean,Object], default: false },
  canStart:  { type: [Boolean,Object], default: false },
  loading:   { type: [Boolean,Object], default: false },
  overall:   { type: Object, default: () => ({ done:0, total:0, percent:0 }) },
  labelFor:  { type: Function, required: true }
})

// v-model 對應事件必須 camelCase
const emit = defineEmits([
  'update:refUrl','update:listInput','update:chips','update:interval','update:keep',
  'add-chips','remove-chip','clear-all','submit'
])

// 將可能是 ref 的值解包成純值
const refUrlV    = computed(() => unref(props.refUrl) ?? '')
const listInputV = computed(() => unref(props.listInput) ?? '')
const chipsV     = computed(() => unref(props.chips) ?? [])
const intervalV  = computed(() => unref(props.interval) ?? 'auto')
const keepV      = computed(() => !!unref(props.keep))
const canStartV  = computed(() => !!unref(props.canStart))
const loadingV   = computed(() => !!unref(props.loading))
const overallV   = computed(() => props.overall || { done:0,total:0,percent:0 })
const chipsCount = computed(() => chipsV.value.length)
</script>

<template>
  <h3>輸入與參數</h3>

  <label>參考影片（Reference URL）</label>
  <input
    :value="refUrlV"
    @input="(e)=>{ console.log('[InputPanel] update:refUrl', e.target.value); emit('update:refUrl', e.target.value.trim()) }"
    type="text" placeholder="https://youtu.be/...." />

  <label style="margin-top:14px">要比對的影片（多個）</label>
  <textarea
    :value="listInputV"
    @input="(e)=>{ console.log('[InputPanel] update:listInput', e.target.value); emit('update:listInput', e.target.value) }"
    placeholder="可多行貼上，逗號或空白自動分割。"></textarea>

  <div class="row" style="align-items:center;gap:10px">
    <button type="button" class="btn" @click="()=>{ console.log('[InputPanel] add-chips'); emit('add-chips') }">加入</button>
    <span class="hint">已加入 <strong>{{ chipsCount }}</strong> 支影片（自動去重、可刪除）</span>
  </div>

  <ChipList
    :items="chipsV"
    :label-for="props.labelFor"
    @remove="(i) => emit('remove-chip', i)"
  />

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

  <div class="row" style="justify-content:space-between;margin-top:10px">
    <label class="checkbox">
      <input type="checkbox" :checked="keepV" @change="(e)=>{ console.log('[InputPanel] update:keep', e.target.checked); emit('update:keep', e.target.checked) }" />
      保留中間檔案
    </label>

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

  <div v-if="overallV.total" style="margin-top:16px">
    <div class="row" style="justify-content:space-between;align-items:center;margin-bottom:6px">
      <div class="tiny">整體進度｜已完成 {{ overallV.done }} / {{ overallV.total }}</div>
      <div class="tiny">{{ overallV.percent }}%</div>
    </div>
    <ProgressBar :value="overallV.percent" />
  </div>
</template>
