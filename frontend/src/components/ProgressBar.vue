<script setup>
import { computed, unref, ref, watch, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  value: { type:[Number,Object], default: 0 },
  small: { type:[Boolean,Object], default: false },
  smooth: { type:[Boolean,Object], default: true }, // 是否啟用平滑動畫
  duration: { type:[Number,Object], default: 0.5 }  // 動畫持續時間（秒）
})

const displayValue = ref(0)
let animationId = null

const v = computed(() => {
  const n = Number(unref(props.value) ?? 0)
  return Math.max(0, Math.min(100, Math.round(n)))
})

const isSmall = computed(() => !!unref(props.small))
const isSmooth = computed(() => !!unref(props.smooth))
const animDuration = computed(() => Number(unref(props.duration)) * 1000) // 轉換為毫秒

// 平滑進度更新函數
function animateProgress(targetValue, startValue, duration) {
  if (animationId) {
    cancelAnimationFrame(animationId)
  }
  
  const startTime = performance.now()
  
  function update(currentTime) {
    const elapsed = currentTime - startTime
    const progress = Math.min(elapsed / duration, 1)
    
    // 使用 ease-out 緩動函數
    const easeProgress = 1 - Math.pow(1 - progress, 3)
    const currentValue = startValue + (targetValue - startValue) * easeProgress
    
    displayValue.value = Math.round(currentValue)
    
    if (progress < 1) {
      animationId = requestAnimationFrame(update)
    } else {
      displayValue.value = targetValue
      animationId = null
    }
  }
  
  animationId = requestAnimationFrame(update)
}

// 監聽進度值變化
watch(v, (newValue, oldValue) => {
  if (isSmooth.value && Math.abs(newValue - oldValue) > 1) {
    // 進度變化較大時，使用平滑動畫
    animateProgress(newValue, oldValue, animDuration.value)
  } else {
    // 進度變化很小或不需要平滑時，直接更新
    displayValue.value = newValue
  }
}, { immediate: true })

// 組件卸載時清理動畫
onUnmounted(() => {
  if (animationId) {
    cancelAnimationFrame(animationId)
  }
})

// 組件掛載時初始化
onMounted(() => {
  displayValue.value = v.value
})
</script>

<template>
  <div class="progress" :style="isSmall ? 'height:8px' : 'height:10px'">
    <div class="bar" :style="{ width: displayValue + '%' }"></div>
  </div>
</template>

