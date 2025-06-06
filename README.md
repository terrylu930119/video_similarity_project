# 影片相似度比對系統

## 專案簡介

這是一個多模態影片相似度比對系統，能夠分析各種網站的影片的視覺、音頻和文本內容，計算不同影片之間的相似度。系統採用深度學習和傳統計算機視覺技術，提供全面的相似度評估。目前主要支援 YouTube 和 Bilibili 等主流視頻平台的影片比對。

## 功能特點

- **多模態分析**：
  - 視覺分析：使用 OpenCV 和 scikit-image 進行圖像處理和特徵提取
  - 音頻分析：使用 librosa 和 soundfile 進行音頻特徵提取
  - 文本分析：使用 Whisper 進行語音識別和文本相似度計算

- **影片下載與處理**：
  - 支援 YouTube、Bilibili 等多個主流視頻平台
  - 使用 yt-dlp 作為主要下載引擎
  - 支援自定義視頻解析度和幀提取間隔

- **性能優化**：
  - GPU 加速支援（使用 PyTorch）
  - 並行處理架構
  - 智能緩存機制避免重複處理

- **相似度計算**：
  - 多維度特徵融合
  - 可調整的權重配置
  - 詳細的相似度分析報告

## 技術架構

系統由以下主要模組組成：

- **main.py**：主程式入口，協調各模組工作
- **downloader.py**：影片下載模組，支援多平台
- **video_utils.py**：視頻處理工具，負責幀提取
- **audio_processor.py**：音頻處理模組，特徵提取
- **text_processor.py**：文本處理模組，語音識別
- **image_processor.py**：圖像處理模組，視覺特徵提取
- **similarity.py**：相似度計算核心模組
- **gpu_utils.py**：GPU 資源管理
- **logger.py**：日誌記錄系統
- **dependencies.py**：依賴項管理

## 安裝方法

### 系統要求

- Python 3.8+
- CUDA 支援（可選，用於 GPU 加速）
- FFmpeg（用於音頻和視頻處理）

### 安裝步驟

1. 克隆專案倉庫：
   ```bash
   git clone https://github.com/yourusername/video_similarity_project.git
   cd video_similarity_project
   ```

2. 創建並啟動虛擬環境：
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. 安裝依賴項：
   ```bash
   pip install -r requirements.txt
   ```

4. 安裝 FFmpeg：
   - Windows：使用 [FFmpeg 官方網站](https://ffmpeg.org/download.html) 下載並添加到 PATH
   - Linux：`sudo apt-get install ffmpeg`
   - Mac：`brew install ffmpeg`

## 使用方法

### 基本用法

1. 修改 `main.py` 中的參考影片和比對影片連結：
   ```python
   reference_link = "參考影片網址"
   comparison_links = [
       "需要被比對的影片1網址",
       "需要被比對的影片2網址",
       ....
   ]
   ```

2. 運行主程式：
   ```bash
   python main.py
   ```

### 進階配置

- **調整處理參數**：
  ```python
  time_interval = 1.0  # 幀提取間隔（秒）
  resolution = "720p"  # 視頻解析度
  use_silence_detection = True  # 是否使用靜音檢測
  use_source_separation = True  # 是否使用音源分離
  ```

- **調整相似度權重**：
  在 `similarity.py` 中修改權重配置：
  ```python
  weights = {
      'visual': 0.4,
      'audio': 0.3,
      'text': 0.3
  }
  ```

## 相似度計算方法

系統使用多層次的特徵提取和相似度計算：

1. **視覺相似度**：
   - 使用 OpenCV 進行圖像預處理
   - 應用深度學習模型提取視覺特徵
   - 計算圖像哈希和特徵向量相似度

2. **音頻相似度**：
   - 使用 librosa 提取音頻特徵
   - 分析頻譜特徵和節奏模式
   - 計算音頻特徵向量相似度

3. **文本相似度**：
   - 使用 Whisper 進行語音識別
   - 應用 sentence-transformers 計算文本相似度
   - 考慮文本的語義相似度

## 注意事項

- 首次運行會下載必要的模型文件（約 1GB）
- GPU 加速可顯著提升處理速度
- 系統會自動清理臨時文件
- 支援中斷處理和資源清理
- 建議使用穩定的網絡連接

## 已知限制

- 目前主要支援 YouTube 和 Bilibili
- 長視頻處理時間較長
- 需要較大的硬碟空間用於臨時文件
- 某些網站可能需要配置代理

## 未來計劃

- [ ] 支援更多視頻平台
- [ ] 優化處理速度
- [ ] 改進相似度算法
- [ ] 添加 Web 界面
- [ ] 支援批量處理

## 貢獻指南

歡迎提交問題報告和改進建議！請遵循以下步驟：

1. Fork 專案倉庫
2. 創建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 開啟 Pull Request

## 授權

本專案採用 MIT 授權 - 詳見 [LICENSE](LICENSE) 文件 