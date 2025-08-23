# 🎬 影音比對系統 - 前端 API 整合指導

## 📋 目錄
- [API 概覽](#api-概覽)
- [認證與 CORS](#認證與-cors)
- [API 端點詳解](#api-端點詳解)
- [Server-Sent Events](#server-sent-events)
- [錯誤處理](#錯誤處理)
- [前端整合範例](#前端整合範例)
- [最佳實踐](#最佳實踐)

## 🌐 API 概覽

### **基礎資訊**
- **Base URL**: `http://localhost:8000` (開發環境)
- **API 版本**: `/api`
- **通訊協定**: HTTP/HTTPS + Server-Sent Events
- **資料格式**: JSON

### **支援的影片平台**
- ✅ YouTube (youtube.com, youtu.be)
- ✅ Bilibili (bilibili.com, b23.tv)
- ✅ 其他通用網站

## 🔐 認證與 CORS

### **CORS 設定**
後端已啟用 CORS，支援跨域請求：
```python
# 後端設定
allow_origins=["*"]
allow_credentials=True
allow_methods=["*"]
allow_headers=["*"]
```

### **認證需求**
目前 API 無需認證，但建議在生產環境中實作。

## 📡 API 端點詳解

### 1. **影片比對 API**

#### **POST** `/api/compare`
提交影片比對任務

**請求格式：**
```typescript
interface CompareRequest {
  ref: string;           // 參考影片 URL
  comp: string[];        // 要比對的影片 URL 陣列
  interval: string;      // 抽幀間隔 ("auto" | "0.5" | "1" | "2" | "5")
  keep: boolean;         // 是否保留中間檔案
  allow_self: boolean;   // 是否允許與自己比對
}
```

**回應格式：**
```typescript
interface CompareResponse {
  task_ids: Array<{
    url: string;         // 影片 URL
    task_id: string;     // 任務 ID
    ref_url: string;     // 參考影片 URL
  }>;
}
```

**使用範例：**
```javascript
const response = await fetch('/api/compare', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    ref: 'https://youtu.be/example1',
    comp: ['https://youtu.be/example2', 'https://youtu.be/example3'],
    interval: 'auto',
    keep: false
  })
});

const result = await response.json();
console.log('任務ID:', result.task_ids);
```

### 2. **狀態查詢 API**

#### **POST** `/api/status`
查詢任務執行狀態

**請求格式：**
```typescript
interface StatusRequest {
  ref: string;           // 參考影片 URL
  comp: string[];        // 要比對的影片 URL 陣列
}
```

**回應格式：**
```typescript
interface StatusItem {
  url: string;           // 影片 URL
  phase: string;         // 執行階段
  percent: number;       // 進度百分比 (0-100)
  cached_flags: {        // 快取標記
    [key: string]: boolean;
  };
}
```

**執行階段說明：**
- `queued`: 佇列中
- `download`: 下載中
- `transcribe`: 轉錄中
- `subtitle`: 字幕解析
- `extract`: 抽幀中
- `audio`: 音訊比對
- `image`: 畫面比對
- `text`: 文本比對
- `compare`: 比對中

### 3. **任務取消 API**

#### **POST** `/api/cancel`
取消正在執行的任務

**請求格式：**
```typescript
interface CancelRequest {
  task_ids: string[];    // 要取消的任務 ID 陣列
}
```

**回應格式：**
```typescript
interface CancelResponse {
  ok: boolean;           // 操作是否成功
  killed: bool;          // 是否已終止任務
}
```

## 🔄 Server-Sent Events

### **事件端點**
**GET** `/api/events`

### **事件類型**

#### 1. **進度更新事件**
```typescript
interface ProgressEvent {
  type: 'progress';
  task_id: string;
  url: string;
  phase: string;
  percent: number;
  phaseName?: string;    // 階段名稱（中文）
  msg?: string;          // 日誌訊息
  overallHint?: number;  // 整體進度提示
  textSource?: 'subtitle' | 'asr';  // 文本來源
  text_skipped?: boolean;           // 文本是否跳過
  text_status?: string;             // 文本狀態
}
```

#### 2. **日誌事件**
```typescript
interface LogEvent {
  type: 'log';
  task_id: string;
  url: string;
  msg: string;
}
```

#### 3. **任務完成事件**
```typescript
interface DoneEvent {
  type: 'done';
  task_id: string;
  url: string;
  ref_url: string;
  score: number;         // 相似度分數 (0-100)
  visual: number;        // 畫面相似度
  audio: number;         // 音訊相似度
  text: number;          // 文本相似度
  text_meaningful: boolean;  // 文本是否有效
  text_status: string;       // 文本狀態
  hot: string[];             // 熱門關鍵字
}
```

#### 4. **任務取消事件**
```typescript
interface CanceledEvent {
  type: 'canceled';
  task_id: string;
  url: string;
}
```

### **前端整合範例**
```javascript
// 建立 SSE 連接
const eventSource = new EventSource('/api/events');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'progress':
      // 更新進度條
      updateProgress(data.task_id, data.percent);
      // 更新狀態
      updateStatus(data.task_id, data.phase, data.phaseName);
      // 添加日誌
      if (data.msg) addLog(data.task_id, data.msg);
      break;
      
    case 'done':
      // 顯示結果
      showResult(data);
      break;
      
    case 'canceled':
      // 處理取消
      handleCancel(data.task_id);
      break;
  }
};

eventSource.onerror = (error) => {
  console.error('SSE 連接錯誤:', error);
  // 可以實作重連邏輯
};
```

## ❌ 錯誤處理

### **HTTP 狀態碼**
- `200`: 成功
- `409`: 衝突（例如：任務已存在）
- `500`: 伺服器內部錯誤

### **錯誤回應格式**
```typescript
interface ErrorResponse {
  detail: string;        // 錯誤詳細訊息
}
```

### **前端錯誤處理範例**
```javascript
try {
  const response = await fetch('/api/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestData)
  });
  
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `HTTP ${response.status}`);
  }
  
  const result = await response.json();
  return result;
  
} catch (error) {
  console.error('API 請求失敗:', error);
  // 顯示錯誤訊息給用戶
  showErrorMessage(error.message);
}
```

## 🎯 前端整合範例

### **完整的比對流程**
```javascript
class VideoCompareService {
  constructor() {
    this.eventSource = null;
    this.tasks = new Map();
  }
  
  // 開始比對
  async startCompare(refUrl, compUrls, options = {}) {
    try {
      // 1. 提交比對任務
      const response = await this.submitCompare(refUrl, compUrls, options);
      
      // 2. 建立 SSE 連接
      this.connectEvents();
      
      // 3. 初始化任務狀態
      response.task_ids.forEach(task => {
        this.tasks.set(task.task_id, {
          url: task.url,
          status: 'queued',
          progress: 0,
          logs: []
        });
      });
      
      return response;
      
    } catch (error) {
      console.error('開始比對失敗:', error);
      throw error;
    }
  }
  
  // 提交比對請求
  async submitCompare(refUrl, compUrls, options) {
    const response = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ref: refUrl,
        comp: compUrls,
        interval: options.interval || 'auto',
        keep: options.keep || false
      })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail);
    }
    
    return await response.json();
  }
  
  // 連接 SSE 事件
  connectEvents() {
    if (this.eventSource) {
      this.eventSource.close();
    }
    
    this.eventSource = new EventSource('/api/events');
    
    this.eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleEvent(data);
    };
    
    this.eventSource.onerror = (error) => {
      console.error('SSE 錯誤:', error);
      // 實作重連邏輯
      setTimeout(() => this.connectEvents(), 5000);
    };
  }
  
  // 處理事件
  handleEvent(data) {
    const task = this.tasks.get(data.task_id);
    if (!task) return;
    
    switch (data.type) {
      case 'progress':
        task.status = data.phase;
        task.progress = data.percent;
        if (data.msg) task.logs.push(data.msg);
        this.updateUI(task);
        break;
        
      case 'done':
        task.status = 'completed';
        task.progress = 100;
        task.result = data;
        this.showResult(task);
        break;
    }
  }
  
  // 取消任務
  async cancelTask(taskId) {
    try {
      const response = await fetch('/api/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_ids: [taskId] })
      });
      
      if (response.ok) {
        const task = this.tasks.get(taskId);
        if (task) {
          task.status = 'canceled';
          this.updateUI(task);
        }
      }
    } catch (error) {
      console.error('取消任務失敗:', error);
    }
  }
  
  // 清理資源
  destroy() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
    this.tasks.clear();
  }
}
```

## 🚀 最佳實踐

### **1. 錯誤處理**
- 實作完整的錯誤處理機制
- 提供用戶友善的錯誤訊息
- 實作重試機制

### **2. 狀態管理**
- 使用狀態管理工具（Vuex、Pinia 等）
- 保持前端狀態與後端同步
- 實作樂觀更新

### **3. 用戶體驗**
- 實作進度條和載入狀態
- 提供實時進度更新
- 支援任務取消和重試

### **4. 效能優化**
- 實作請求去重
- 使用防抖和節流
- 實作資料快取

### **5. 安全性**
- 驗證輸入資料
- 防止 XSS 攻擊
- 實作適當的錯誤訊息過濾

## 📚 參考資源

- [FastAPI 官方文件](https://fastapi.tiangolo.com/)
- [Server-Sent Events 規範](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [Fetch API 文件](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)

---

**注意事項：**
- 本文件基於後端 API 實作，如有變更請同步更新
- 建議在開發過程中實作完整的錯誤處理和用戶回饋
- 生產環境部署前請進行充分的測試 