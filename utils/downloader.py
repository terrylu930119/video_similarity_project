import os
import re
import hashlib
import yt_dlp
import urllib.parse
from utils.logger import logger
from typing import Dict

# =============== URL 檢查與命名工具 ===============
def is_valid_url(url: str) -> bool:
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def generate_safe_filename(url: str) -> str:
    url_hash: str = hashlib.md5(url.encode('utf-8')).hexdigest()
    parsed_url = urllib.parse.urlparse(url)
    path_parts: list[str] = [p for p in parsed_url.path.strip('/').split('/') if p]

    meaningful_part: str = ""
    for part in reversed(path_parts):
        if not part.isdigit() and re.search(r'[a-zA-Z]', part):
            meaningful_part = part
            break

    if meaningful_part:
        safe_part: str = re.sub(r'[^\w\-]', '_', meaningful_part)
        return f"{safe_part}_{url_hash}"
    else:
        return url_hash

# =============== 影片下載主函式 ===============
def download_video(url: str, output_dir: str, resolution: str = "480p", max_retries: int = 3) -> str:
    if not is_valid_url(url):
        raise ValueError(f"無效的 URL: {url}")

    safe_filename: str = generate_safe_filename(url)

    try:
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"沒有輸出目錄的寫入權限: {output_dir}")
    except Exception as e:
        logger.error(f"建立或檢查輸出目錄時出錯: {str(e)}")
        raise

    output_path: str = os.path.join(output_dir, f"{safe_filename}.mp4")
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info(f"影片已存在且大小正常 ({os.path.getsize(output_path)} bytes): {output_path}")
        return output_path
    elif os.path.exists(output_path):
        logger.warning(f"發現空檔案，將重新下載: {output_path}")
        os.remove(output_path)

    ydl_opts: Dict[str, object] = {
        'format': f'bestvideo[height<={resolution[:-1]}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={resolution[:-1]}]+bestaudio/best[height<={resolution[:-1]}]/best',
        'outtmpl': output_path,
        'quiet': False,
        'no_warnings': True,
        'retries': max_retries,
        'noplaylist': True,
        'extract_flat': False,
        'writeautomaticsub': False,
        'writesubtitles': True,
        'subtitlesformat': 'vtt',
        'subtitleslangs': ['all'],
        'compat_opts': ['no-live-chat'],
        'ignoreerrors': True,
        'skip_download': False,
        'writedescription': False,
        'writeinfojson': True,
        'writeannotations': False,
        'writethumbnailjpg': False,
        'write_all_thumbnails': False,
        'writecomments': False,
        'getcomments': False,
        'writethumbnail': False,
        'force_generic_extractor': False,
        'geo_bypass': True,
        'geo_bypass_country': 'US',
        'extractor_retries': 5,
        'format_sort': ['res', 'ext:mp4:m4a', 'size', 'br', 'asr'],
        'verbose': False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"開始下載影片: {url}")
            ydl.download([url])

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"下載完成: {output_path}")
            return output_path
        else:
            raise Exception("下載的檔案不存在或大小為0")

    except Exception as e:
        logger.error(f"下載失敗: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
