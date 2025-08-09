<script setup>
import { onMounted, ref, watch } from 'vue'
const props = defineProps({ lines: { type: Array, default: () => [] } })
const el = ref(null)
function scrollBottom(){ if (el.value) el.value.scrollTop = el.value.scrollHeight }
onMounted(scrollBottom)
watch(() => props.lines.length, scrollBottom)
</script>

<template>
  <pre class="log" ref="el" aria-live="polite">{{ (lines || []).join('\n') }}</pre>
</template>