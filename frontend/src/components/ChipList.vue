<script setup>
// =============== 影片標籤列表組件 ===============
// 功能：以標籤形式顯示已加入的影片列表，支援個別移除操作

import { computed, unref } from 'vue'

// ──────────────── 組件屬性定義 ────────────────
const props = defineProps({
  items: { type: [Array, Object], default: () => [] },    // 影片URL陣列
  labelFor: { type: Function, required: true }            // 用於生成影片標籤文字的函數
})

// ──────────────── 事件定義 ────────────────
const emit = defineEmits(['remove'])  // 移除影片標籤事件

// ──────────────── 計算屬性 ────────────────
// 將影片陣列解包，確保響應式更新
const list = computed(() => unref(props.items) ?? [])
</script>

<template>
  <!-- 影片標籤容器 -->
  <div class="chips" aria-label="chips">
    <!-- 遍歷顯示每個影片標籤 -->
    <div v-for="(url, i) in list" :key="url + i" class="chip">
      <!-- 影片標籤文字：使用 labelFor 函數生成可讀的標籤 -->
      <span class="chip-text">{{ labelFor(url) }}</span>
      <!-- 移除按鈕：點擊時觸發 remove 事件並傳遞索引 -->
      <span class="x" title="移除" @click="emit('remove', i)">×</span>
    </div>
  </div>
</template>
