<script setup>
import { computed, unref } from 'vue'
const props = defineProps({
  value: { type:[Number,Object], default: 0 },
  small: { type:[Boolean,Object], default: false }
})
const v = computed(() => {
  const n = Number(unref(props.value) ?? 0)
  return Math.max(0, Math.min(100, Math.round(n)))
})
const isSmall = computed(() => !!unref(props.small))
</script>

<template>
  <div class="progress" :style="isSmall ? 'height:8px' : 'height:10px'">
    <div class="bar" :style="{ width: v + '%' }"></div>
  </div>
</template>

