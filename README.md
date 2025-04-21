# 影片相似度比對系統

## 專案簡介

這是一個多模態影片相似度比對系統，能夠分析 YouTube 影片的視覺、音頻和文本內容，計算不同影片之間的相似度。系統採用深度學習和傳統計算機視覺技術，提供全面的相似度評估。

## 功能特點

- **多模態分析**：同時分析影片的視覺、音頻和文本內容
- **YouTube 影片下載**：直接從 YouTube 下載影片進行分析
- **GPU 加速**：支援 GPU 加速處理，提高分析效率
- **智能文本處理**：自動提取字幕或進行語音轉錄
- **綜合相似度評分**：根據不同模態的權重計算整體相似度
- **詳細結果報告**：提供各模態的相似度分數和綜合評分

## 技術架構

系統由以下主要模組組成：

- **downloader.py**：負責從 YouTube 下載影片
- **video_utils.py**：處理影片幀提取和基本資訊獲取
- **audio_processor.py**：處理音頻提取和相似度計算
- **text_processor.py**：處理文本轉錄和相似度計算
- **image_processor.py**：處理圖像特徵提取和相似度計算
- **similarity.py**：整合各模態相似度，計算綜合評分
- **gpu_utils.py**：管理 GPU 資源和加速
- **logger.py**：提供日誌記錄功能
- **dependencies.py**：管理專案依賴項

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

4. 安裝 FFmpeg（如果尚未安裝）：
   - Windows：使用 [FFmpeg 官方網站](https://ffmpeg.org/download.html) 下載並添加到 PATH
   - Linux：`sudo apt-get install ffmpeg`
   - Mac：`brew install ffmpeg`

## 使用方法

### 基本用法

1. 修改 `main.py` 中的參考影片和比對影片連結：
   ```python
   reference_link = "https://www.youtube.com/watch?v=YOUR_REFERENCE_VIDEO"
   comparison_links = [
       "https://www.youtube.com/watch?v=VIDEO_TO_COMPARE_1",
       "https://www.youtube.com/watch?v=VIDEO_TO_COMPARE_2"
   ]
   ```

2. 運行主程式：
   ```bash
   python main.py
   ```

3. 查看結果：程式將顯示各影片與參考影片的相似度評分。

### 進階配置

- **調整幀提取間隔**：修改 `time_interval` 參數（預設為 2.0 秒）
- **調整視頻解析度**：修改 `resolution` 參數（預設為 "480p"）
- **調整相似度權重**：在 `calculate_overall_similarity` 函數中修改 `weights` 參數

## 相似度計算方法

系統使用多種技術計算不同模態的相似度：

- **視覺相似度**：使用感知哈希（pHash）和深度學習特徵提取
- **音頻相似度**：使用頻譜特徵和節奏分析
- **文本相似度**：使用自然語言處理和文本嵌入

最終相似度是這些模態相似度的加權平均，預設權重為：
- 視覺：40%
- 音頻：30%
- 文本：30%

## 注意事項

- 首次運行時會下載必要的模型，可能需要一些時間
- 處理高解析度或長視頻可能需要較長時間
- 如果沒有 GPU，處理速度會較慢
- 系統會自動清理臨時文件，但可以選擇保留下載的視頻

## 貢獻指南

歡迎提交問題報告和改進建議！請遵循以下步驟：

1. Fork 專案倉庫
2. 創建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 開啟一個 Pull Request

## 授權

本專案採用 MIT 授權 - 詳見 [LICENSE](LICENSE) 文件 