// =============== 影片比對系統 - Server-Sent Events 服務模組 ===============
// 功能：提供與後端的實時通訊連接，用於接收任務進度更新、日誌資訊等實時資料

import { API_BASE } from './api'

// ──────────────── 事件源連接函數 ────────────────
// 建立 Server-Sent Events 連接，用於接收後端的實時資料推送
// 參數：path - 事件源的相對路徑（例如：'/api/events'）
// 返回：EventSource 實例，用於監聽實時事件
export function openEventSource(path) {
    return new EventSource(API_BASE + path)
}
