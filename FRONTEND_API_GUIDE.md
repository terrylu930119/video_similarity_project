# 🎬 影音比對系統 - 前端 API 整合指導

## 🚀 快速開始 
```javascript
// 提交比對任務
const response = await fetch('http://localhost:8000/api/compare', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    ref: 'https://youtu.be/example1',        // 參考影片 URL
    comp: ['https://youtu.be/example2'],     // 要比對的影片 URL 陣列
    interval: 'auto'                         // 抽幀間隔，預設 "auto"
  })
});
const result = await response.json();

// 監聽 SSE
const es = new EventSource('http://localhost:8000/api/events');
es.onmessage = (e) => console.log(JSON.parse(e.data));
```

## 📑 API 一覽表
| 路徑           | 方法  | 用途           |
| -------------- | ---- | ------------ |
| `/api/compare` | POST | 提交影片比對任務     |
| `/api/status`  | POST | 查詢任務進度       |
| `/api/cancel`  | POST | 取消比對任務       |
| `/api/events`  | GET  | 接收實時事件 (SSE) |


## 📡 API 詳解

### 1. POST `/api/compare`
#### 請求 JSON
```json
{
  "ref": "https://youtu.be/example1",        // 參考影片 URL (必填)
  "comp": ["https://youtu.be/example2"],     // 要比對的影片 URL 陣列 (必填)
  "interval": "auto",                        // 抽幀間隔："auto"|"0.5"|"1"|"2"|"5" (選填，預設"auto")
  "keep": false,                             // 是否保留中間檔案 (選填，預設false)
  "allow_self": false                        // 是否允許與自己比對 (選填，預設false)
}
```
#### 回應 JSON
```json
{
  "task_ids": [],                            // 任務ID陣列，初始為空，實際ID透過SSE hello事件提供
  "cmd": ["python","-m","cli.main","--ref","https://youtu.be/example1","--comp","https://youtu.be/example2"]  // 除錯用命令列陣列
}
```
- task_ids 初始可能為空，實際 ID 會透過 SSE hello 事件提供。
- **錯誤碼**  
  - 409: 已有任務 → UI 提示「已有比對任務進行中」
  - 422: 請求格式錯誤 → 提示使用者檢查參數


### 2. POST `/api/status`
#### 請求 JSON
```json
{
  "ref": "https://youtu.be/example1",        // 參考影片 URL (必填)
  "comp": ["https://youtu.be/example2"]      // 要比對的影片 URL 陣列 (必填)
}
```
#### 回應 JSON
```json
[
  {
    "url": "https://youtu.be/example1",      // 影片 URL
    "phase": "download",                     // 處理階段：queued|download|transcribe|extract|compare
    "percent": 30,                           // 進度百分比 (0-100)
    "cached_flags": {                        // 快取檔案標記
      "video": true,                         // 是否已下載影片檔
      "transcript": false,                   // 是否已取得轉錄文本檔
      "frames": false                        // 是否已抽取影格圖像
    }
  }
]
```


### 3. POST `/api/cancel`
#### 請求 JSON
```json
{ 
  "task_ids": ["abc123def4"]                 // 要取消的任務 ID 陣列 (必填)
}
```
#### 回應 JSON
```json
{ 
  "ok": true,                                // 操作是否成功
  "killed": true                             // 是否已終止任務程序
}
```


## 🔄 SSE 事件 (`GET /api/events`)
### hello
```json
{
  "type": "hello",                           // 事件類型
  "ref": {                                   // 參考影片資訊
    "task_id": "ref-1",                      // 參考影片任務ID
    "url": "https://youtu.be/example1"       // 參考影片URL
  },
  "targets": [{                              // 目標影片陣列
    "task_id": "abc123def4",                 // 目標影片任務ID
    "url": "https://youtu.be/example2"       // 目標影片URL
  }]
}
```
### progress
```json
{
  "type": "progress",                        // 事件類型
  "task_id": "abc123def4",                   // 任務ID
  "url": "https://youtu.be/example2",        // 影片URL
  "ref_url": "https://youtu.be/example1",    // 參考影片URL
  "phase": "download",                       // 處理階段：queued|download|transcribe|extract|compare
  "percent": 50,                             // 進度百分比 (0-100)
  "msg": "下載中..."                         // 可選的狀態訊息
}
```
### done
```json
{
  "type": "done",                            // 事件類型
  "task_id": "abc123def4",                   // 任務ID
  "url": "https://youtu.be/example2",        // 影片URL
  "ref_url": "https://youtu.be/example1",    // 參考影片URL
  "score": 87.2,                             // 整體相似度分數 (0-100)
  "visual": 80,                              // 畫面相似度分數 (0-100)
  "audio": 92,                               // 音訊相似度分數 (0-100)
  "text": 85,                                // 文本相似度分數 (0-100)
  "text_meaningful": true,                   // 文本內容是否有效
  "text_status": "ok"                        // 文本狀態描述
}
```
### canceled
```json
{
  "type": "canceled",                        // 事件類型
  "task_id": "abc123def4",                   // 任務ID
  "url": "https://youtu.be/example2",        // 影片URL
  "ref_url": "https://youtu.be/example1"     // 參考影片URL
}
```

## 📊 任務生命週期
```bash
queued → download → transcribe/subtitle → extract → audio/image/text → compare → done
                          ↓
                       canceled
```

## ✅ 前置驗證規則
- `comp` 陣列：1 ~ 10 筆
- `interval`:`"auto" | "0.5" | "1" | "2" | "5"`
- URL 必須為 YouTube / Bilibili / 支援網站

## 🛠️ 最佳實踐
- SSE 去重：以 (`task_id, type, phase`) 當 key，避免重複渲染
- 錯誤碼對應 UI：
  - 409 → 提醒等待/取消舊任務
  - 422 → 提示輸入有誤
  - 500 → 提供「重試」按鈕
- 斷線恢復：SSE 斷線後 → 先呼叫 `/api/status` → 再重連 SSE
- 生產安全性：建議加上 API key / JWT 與限定 CORS