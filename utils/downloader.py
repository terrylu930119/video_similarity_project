import os
import re
import hashlib
import yt_dlp
import urllib.parse
import urllib.request
from pathlib import Path
from utils.logger import logger
from typing import Dict

_pann_weight_checked = False

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
    path_parts: list[str] = [
        p for p in parsed_url.path.strip('/').split('/') if p]

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


def download_video(url: str, output_dir: str, resolution: str = "720p", max_retries: int = 3) -> str:
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
        logger.info(
            f"影片已存在且大小正常 ({os.path.getsize(output_path)} bytes): {output_path}")
        return output_path
    elif os.path.exists(output_path):
        logger.warning(f"發現空檔案，將重新下載: {output_path}")
        os.remove(output_path)

    ydl_opts: Dict[str, object] = {
        'format': f'bestvideo[height<={resolution[:-1]}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={resolution[:-1]}]+bestaudio/best[height<={resolution[:-1]}]/best',
        'outtmpl': output_path,
        'cookiefile': 'www.youtube.com_cookies.txt',
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


def ensure_pann_weights(expected_size: int = 340_000_000) -> Path:
    """
    確保 PANN 權重檔案存在且完整，否則自動下載。
    回傳權重檔案的完整路徑。
    """
    global _pann_weight_checked
    checkpoint_path = Path.home() / 'panns_data' / 'Cnn14_mAP=0.431.pth'
    if checkpoint_path.exists() and checkpoint_path.stat().st_size > expected_size * 0.95:
        if not _pann_weight_checked:
            logger.info(f"PANN 權重已存在且完整: {checkpoint_path}")
            _pann_weight_checked = True
        return checkpoint_path
    # 若不存在或損壞則下載
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'
    logger.info(f"下載 PANN 權重到 {checkpoint_path} ...")
    urllib.request.urlretrieve(url, checkpoint_path)
    logger.info("PANN 權重下載完成。")
    return checkpoint_path
