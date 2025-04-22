import os
import time
from logger import logger
import re
import yt_dlp

def is_valid_youtube_url(url: str) -> bool:
    """檢查 URL 是否為有效的 YouTube 連結"""
    youtube_regex = (
        r'(https?://)?(www\.)?'
        r'(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|shorts/|.+\?v=)?([^&]{11})')
    return bool(re.match(youtube_regex, url))

def extract_video_id(url: str) -> str:
    """從 YouTube URL 中提取影片 ID 和播放清單索引"""
    # 提取影片 ID
    video_id = None
    
    # 處理 Shorts URL
    shorts_match = re.search(r'shorts/([^/?]+)', url)
    if shorts_match:
        video_id = shorts_match.group(1)[:11]
    else:
        # 處理標準 YouTube URL
        v_match = re.search(r'[?&]v=([^&]+)', url)
        if v_match:
            video_id = v_match.group(1)[:11]
        else:
            path_match = re.search(r'(?:embed/|v/|.+\?v=)?([^&]{11})', url)
            if path_match:
                video_id = path_match.group(1)
    
    # 提取播放清單索引
    index_match = re.search(r'index=(\d+)', url)
    playlist_index = index_match.group(1) if index_match else ""
    
    # 組合檔案名稱：videoId_index
    return f"{video_id}_{playlist_index}" if playlist_index else video_id

def download_youtube(url: str, output_dir: str, resolution: str = "480p", max_retries: int = 3) -> str:
    """
    使用 yt-dlp 下載 YouTube 影片
    
    參數:
        url: YouTube 影片完整 URL
        output_dir: 輸出目錄
        resolution: 影片解析度，預設為 "480p"
        max_retries: 最大重試次數
    
    返回:
        下載的影片路徑
    """
    # 驗證 URL 格式
    if not is_valid_youtube_url(url):
        raise ValueError(f"無效的 YouTube URL: {url}")
    
    # 提取影片 ID（包含播放清單索引）
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"無法從 URL 中提取影片 ID: {url}")
    
    # 建立輸出目錄
    try:
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"沒有輸出目錄的寫入權限: {output_dir}")
    except Exception as e:
        logger.error(f"建立或檢查輸出目錄時出錯: {str(e)}")
        raise
    
    # 建立輸出檔案路徑
    output_path = os.path.join(output_dir, f"{video_id}.mp4")
    
    # 檢查文件是否已存在且大小正常
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > 0:
            logger.info(f"影片已存在且大小正常 ({file_size} bytes): {output_path}")
            return output_path
        else:
            logger.warning(f"發現空檔案，將重新下載: {output_path}")
            os.remove(output_path)
    
    # 設定 yt-dlp 選項
    ydl_opts = {
        'format': f'bestvideo[height<={resolution[:-1]}][ext=mp4]+bestaudio[ext=m4a]/best[height<={resolution[:-1]}][ext=mp4]/best[ext=mp4]/best',  # 增加更多格式選項
        'outtmpl': output_path,
        'quiet': False,
        'no_warnings': True,
        'retries': max_retries,
        'noplaylist': True,  # 禁用播放清單下載
        'extract_flat': False,  # 不提取播放清單資訊
        'writeautomaticsub': False,  # 禁止下載自動生成的字幕
        'writesubtitles': True,     # 下載手動上傳的字幕
        'subtitlesformat': 'vtt',
        'subtitleslangs': ['all', '-live_chat'],  # 排除 live chat
        'compat_opts': ['no-live-chat'],  # 禁用 live chat
        'skip_download': False,
        'writedescription': False,
        'writeinfojson': False,
        'writeannotations': False,
        'writethumbnailjpg': False,
        'write_all_thumbnails': False,
        'writecomments': False,
        'getcomments': False,
        'writethumbnail': False,
        'force_generic_extractor': False,  # 不強制使用通用提取器
        'extractor_args': {'youtube:player_client': ['android'], 'youtube:player_skip': ['webpage', 'configs', 'js']},  # 使用 Android 客戶端
        'cookiesfrombrowser': None,  # 不使用瀏覽器 cookie
        'geo_bypass': True,  # 繞過地理限制
        'geo_bypass_country': 'US',  # 使用美國 IP
        'extractor_retries': 5,  # 增加提取器重試次數
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"開始下載影片: {url}")
            ydl.download([url])
            
        # 檢查下載的檔案
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"下載完成: {output_path}")
            return output_path
        else:
            raise Exception("下載的檔案不存在或大小為0")
            
    except Exception as e:
        logger.error(f"下載失敗: {str(e)}")
        # 清理可能的不完整檔案
        if os.path.exists(output_path):
            os.remove(output_path)
        raise Exception(f"下載失敗: {str(e)}")