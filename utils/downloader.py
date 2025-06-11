import os
import time
from utils.logger import logger
import re
import yt_dlp
import hashlib
import urllib.parse

def is_valid_url(url: str) -> bool:
    """檢查 URL 是否為有效的網址"""
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def generate_safe_filename(url: str) -> str:
    """
    從 URL 生成安全且具語意的檔案名稱。
    
    檔名格式：<有意義的部分>_<URL MD5 雜湊值>
    - 若無有意義部分，僅回傳雜湊值
    - 自動避開純數字路徑，避免命名無意義
    - 清除非法檔名字符，確保跨平台安全
    """
    # 生成雜湊值（保底）
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    
    # 解析 URL，取路徑最後一段當作可能的語意部分
    parsed_url = urllib.parse.urlparse(url)
    path_parts = [p for p in parsed_url.path.strip('/').split('/') if p]

    meaningful_part = ""
    for part in reversed(path_parts):
        if not part.isdigit() and re.search(r'[a-zA-Z]', part):  # 避免純數字或無語意字串
            meaningful_part = part
            break

    if meaningful_part:
        # 清理掉奇怪字元，只留字母、數字、_ 和 -
        safe_part = re.sub(r'[^\w\-]', '_', meaningful_part)
        return f"{safe_part}_{url_hash}"
    else:
        return url_hash


def download_video(url: str, output_dir: str, resolution: str = "480p", max_retries: int = 3) -> str:
    """
    使用 yt-dlp 下載影片，支援各種網站
    
    參數:
        url: 影片完整 URL
        output_dir: 輸出目錄
        resolution: 影片解析度，預設為 "480p"
        max_retries: 最大重試次數
    
    返回:
        下載的影片路徑
    """
    # 驗證 URL 格式
    if not is_valid_url(url):
        raise ValueError(f"無效的 URL: {url}")
    
    # 生成安全的檔案名稱
    safe_filename = generate_safe_filename(url)
    
    # 建立輸出目錄
    try:
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"沒有輸出目錄的寫入權限: {output_dir}")
    except Exception as e:
        logger.error(f"建立或檢查輸出目錄時出錯: {str(e)}")
        raise
    
    # 建立輸出檔案路徑
    output_path = os.path.join(output_dir, f"{safe_filename}.mp4")
    
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
        'format': f'bestvideo[height<={resolution[:-1]}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={resolution[:-1]}]+bestaudio/best[height<={resolution[:-1]}]/best',  # 更靈活的格式選擇
        'outtmpl': output_path,
        'quiet': False,
        'no_warnings': True,
        'retries': max_retries,
        'noplaylist': True,  # 禁用播放清單下載
        'extract_flat': False,  # 不提取播放清單資訊
        'writeautomaticsub': False,  # 禁止下載自動生成的字幕
        'writesubtitles': True,     # 下載手動上傳的字幕
        'subtitlesformat': 'vtt',
        'subtitleslangs': ['all'],  # 排除 live chat
        'compat_opts': ['no-live-chat'],  # 禁用 live chat
        'ignoreerrors': True,  # 忽略字幕下載錯誤
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
        'geo_bypass': True,  # 繞過地理限制
        'geo_bypass_country': 'US',  # 使用美國 IP
        'extractor_retries': 5,  # 增加提取器重試次數
        'format_sort': ['res', 'ext:mp4:m4a', 'size', 'br', 'asr'],  # 添加格式排序選項
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
