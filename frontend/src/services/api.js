// =============== 影片比對系統 - API 服務模組 ===============
// 功能：提供與後端服務的 HTTP 通訊介面，包括影片比對、狀態查詢、任務取消等操作

import axios from 'axios'

// ──────────────── 環境配置 ────────────────
// 從環境變數獲取 API 基礎 URL，支援不同環境的配置
import.meta.env.VITE_API_BASE

// API 基礎 URL：優先使用環境變數，否則預設為本地開發環境
export const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

// ──────────────── HTTP 客戶端配置 ────────────────
// 創建 axios 實例，設定基礎配置
const http = axios.create({
    baseURL: API_BASE,                                    // 設定基礎 URL
    headers: { 'Content-Type': 'application/json' }       // 設定預設請求標頭
})

// ──────────────── API 端點函數 ────────────────
// 影片比對 API：提交比對任務到後端
export const compare = (payload) =>
    http.post('/api/compare', payload).then(r => r.data)

// 狀態查詢 API：查詢任務執行狀態和進度
export const status = (payload) =>
    http.post('/api/status', payload).then(r => r.data)

// 任務取消 API：取消正在執行的比對任務
export const cancel = (payload) =>
    http.post('/api/cancel', payload).then(r => r.data)
