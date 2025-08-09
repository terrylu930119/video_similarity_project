<script setup>
import { computed, unref } from 'vue'
const props = defineProps({
  items: { type: [Array, Object], default: () => [] },
  labelFor: { type: Function, required: true }
})
const emit = defineEmits(['remove'])
const list = computed(() => unref(props.items) ?? [])
</script>

<template>
  <div class="chips" aria-label="chips">
    <div v-for="(url, i) in list" :key="url + i" class="chip">
      <span class="chip-text">{{ labelFor(url) }}</span>
      <span class="x" title="移除" @click="emit('remove', i)">×</span>
    </div>
  </div>
</template>
