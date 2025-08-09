import { createApp } from 'vue'
import App from './App.vue'
import '@/styles/variables.css'
import '@/styles/app.css'

// 最早期的心跳：看到這行代表 JS 已成功執行
console.log('[main] loaded')

createApp(App).mount('#app')
