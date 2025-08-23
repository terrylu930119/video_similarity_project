# utils/downloader.py
import os
import re
import hashlib
import yt_dlp
import urllib.parse
import urllib.request
from pathlib import Path
from utils.logger import emit
from utils.logger import logger
from typing import Dict, Optional

# =============== 全域變數 ===============
_pann_weight_checked = False


# =============== URL 檢查與命名工具 ===============
def is_valid_url(url: str) -> bool:
    """
    驗證 URL 是否有效

    Args:
        url: 要驗證的 URL 字串

    Returns:
        bool: URL 是否有效
    """
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def _extract_meaningful_filename_part(path_parts: list[str]) -> Optional[str]:
    """
    從路徑部分中提取有意義的檔案名稱部分

    Args:
        path_parts: URL 路徑分割後的列表

    Returns:
        Optional[str]: 有意義的檔案名稱部分，如果沒有則回傳 None
    """
    for part in reversed(path_parts):
        if not part.isdigit() and re.search(r'[a-zA-Z]', part):
            return re.sub(r'[^\w\-]', '_', part)
    return None


def generate_safe_filename(url: str) -> str:
    """
    根據 URL 產生安全的檔案名稱

    Args:
        url: 影片 URL

    Returns:
        str: 安全的檔案名稱
    """
    url_hash: str = hashlib.md5(url.encode('utf-8')).hexdigest()
    parsed_url = urllib.parse.urlparse(url)
    path_parts: list[str] = [
        p for p in parsed_url.path.strip('/').split('/') if p]

    meaningful_part: Optional[str] = _extract_meaningful_filename_part(path_parts)

    if meaningful_part:
        return f"{meaningful_part}_{url_hash}"
    else:
        return url_hash


# =============== 目錄管理工具 ===============
def _ensure_output_directory(output_dir: str) -> None:
    """
    確保輸出目錄存在且可寫入

    Args:
        output_dir: 輸出目錄路徑

    Raises:
        PermissionError: 沒有寫入權限時拋出
        Exception: 其他錯誤時拋出
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"沒有輸出目錄的寫入權限: {output_dir}")
    except Exception as e:
        logger.error(f"建立或檢查輸出目錄時出錯: {str(e)}")
        raise


def _check_existing_file(output_path: str) -> bool:
    """
    檢查檔案是否已存在且有效

    Args:
        output_path: 檔案路徑

    Returns:
        bool: 檔案是否已存在且有效
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info(f"影片已存在且大小正常 ({os.path.getsize(output_path)} bytes): {output_path}")
        return True
    elif os.path.exists(output_path):
        logger.warning(f"發現空檔案，將重新下載: {output_path}")
        os.remove(output_path)
    return False


# =============== yt-dlp 設定 ===============
def _create_format_string(resolution: str) -> str:
    """
    建立格式選擇字串

    Args:
        resolution: 解析度設定

    Returns:
        str: 格式選擇字串
    """
    height_limit = resolution[:-1]  # 移除 'p' 後綴
    return f'bestvideo[height<={height_limit}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={height_limit}]+bestaudio/best[height<={height_limit}]/best'


def _create_basic_options(output_path: str, max_retries: int) -> Dict[str, object]:
    """
    建立基本下載選項

    Args:
        output_path: 輸出檔案路徑
        max_retries: 最大重試次數

    Returns:
        Dict[str, object]: 基本選項字典
    """
    return {
        'outtmpl': output_path,
        'cookiefile': 'www.youtube.com_cookies.txt',
        'quiet': False,
        'no_warnings': True,
        'retries': max_retries,
        'noplaylist': True,
        'extract_flat': False,
        'ignoreerrors': True,
        'skip_download': False,
        'verbose': False,
    }


def _create_subtitle_options() -> Dict[str, object]:
    """
    建立字幕相關選項

    Returns:
        Dict[str, object]: 字幕選項字典
    """
    return {
        'writeautomaticsub': False,
        'writesubtitles': True,
        'subtitlesformat': 'vtt',
        'subtitleslangs': ['all'],
        'automaticsubtitlesformat': 'vtt',
        'automaticsubtitleslangs': ['all'],
    }


def _create_metadata_options() -> Dict[str, object]:
    """
    建立中繼資料相關選項

    Returns:
        Dict[str, object]: 中繼資料選項字典
    """
    return {
        'writedescription': False,
        'writeinfojson': True,
        'writeannotations': False,
        'writethumbnailjpg': False,
        'write_all_thumbnails': False,
        'writecomments': False,
        'getcomments': False,
        'writethumbnail': False,
    }


def _create_advanced_options() -> Dict[str, object]:
    """
    建立進階下載選項

    Returns:
        Dict[str, object]: 進階選項字典
    """
    return {
        'compat_opts': ['no-live-chat'],
        'force_generic_extractor': False,
        'geo_bypass': True,
        'geo_bypass_country': 'US',
        'extractor_retries': 5,
        'format_sort': ['res', 'ext:mp4:m4a', 'size', 'br', 'asr'],
        # 添加 PO Token 支援以解決字幕下載問題
        'extractor_args': {
            'youtube': {
                'po_token': 'web.subs',  # 使用 web.subs PO Token
            }
        },
    }


def _create_ydl_options(output_path: str, resolution: str, max_retries: int) -> Dict[str, object]:
    """
    建立 yt-dlp 下載選項

    Args:
        output_path: 輸出檔案路徑
        resolution: 解析度設定
        max_retries: 最大重試次數

    Returns:
        Dict[str, object]: yt-dlp 選項字典
    """
    options = {}

    # 合併所有選項
    options.update(_create_basic_options(output_path, max_retries))
    options.update(_create_subtitle_options())
    options.update(_create_metadata_options())
    options.update(_create_advanced_options())

    # 添加格式選擇
    options['format'] = _create_format_string(resolution)

    return options


# =============== 影片下載主函式 ===============
def _perform_download(url: str, output_path: str, ydl_opts: Dict[str, object]) -> None:
    """
    執行實際的下載操作

    Args:
        url: 影片 URL
        output_path: 輸出檔案路徑
        ydl_opts: yt-dlp 選項

    Raises:
        Exception: 下載失敗時拋出
    """
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        logger.info(f"開始下載影片: {url}")
        ydl.download([url])


def _verify_download(output_path: str) -> None:
    """
    驗證下載結果

    Args:
        output_path: 輸出檔案路徑

    Raises:
        Exception: 檔案不存在或大小為 0 時拋出
    """
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise Exception("下載的檔案不存在或大小為0")

    logger.info(f"下載完成: {output_path}")


def _setup_download_environment(url: str, output_dir: str) -> tuple[str, str]:
    """
    設定下載環境

    Args:
        url: 影片 URL
        output_dir: 輸出目錄

    Returns:
        tuple[str, str]: (安全檔案名稱, 輸出檔案路徑)
    """
    safe_filename = generate_safe_filename(url)
    output_path = os.path.join(output_dir, f"{safe_filename}.mp4")

    _ensure_output_directory(output_dir)

    return safe_filename, output_path


def _execute_download_process(url: str, output_path: str, resolution: str, max_retries: int) -> None:
    """
    執行下載流程

    Args:
        url: 影片 URL
        output_path: 輸出檔案路徑
        resolution: 解析度設定
        max_retries: 最大重試次數

    Raises:
        Exception: 下載失敗時拋出
    """
    ydl_opts = _create_ydl_options(output_path, resolution, max_retries)
    _perform_download(url, output_path, ydl_opts)
    _verify_download(output_path)


def _handle_download_error(output_path: str, error: Exception) -> None:
    """
    處理下載錯誤，清理失敗的檔案

    Args:
        output_path: 輸出檔案路徑
        error: 發生的錯誤
    """
    logger.error(f"下載失敗: {str(error)}")
    if os.path.exists(output_path):
        os.remove(output_path)


def download_video(url: str, output_dir: str, resolution: str = "720p", max_retries: int = 3, task_id: str = None) -> str:
    """
    下載影片到指定目錄

    Args:
        url: 影片 URL
        output_dir: 輸出目錄
        resolution: 影片解析度 (預設: "720p")
        max_retries: 最大重試次數 (預設: 3)
        task_id: 任務 ID，用於發送進度訊息

    Returns:
        str: 下載完成的檔案路徑

    Raises:
        ValueError: URL 無效時拋出
        PermissionError: 沒有寫入權限時拋出
        Exception: 下載失敗時拋出
    """
    if not is_valid_url(url):
        raise ValueError(f"無效的 URL: {url}")

    safe_filename, output_path = _setup_download_environment(url, output_dir)

    if _check_existing_file(output_path):
        # 如果檔案已存在，發送相應的訊息
        if task_id:
            emit("progress", task_id=task_id, phase="download", percent=10, msg="影片已存在（本地檔案）")
        return output_path

    # 如果檔案不存在，發送開始下載的訊息
    if task_id:
        emit("progress", task_id=task_id, phase="download", percent=1, msg="開始下載影片")

    try:
        _execute_download_process(url, output_path, resolution, max_retries)
        return output_path
    except Exception as e:
        _handle_download_error(output_path, e)
        raise


# =============== PANN 權重管理 ===============
def ensure_pann_weights(expected_size: int = 340_000_000) -> Path:
    """
    確保 PANN 權重檔案存在且完整，否則自動下載。

    Args:
        expected_size: 預期的檔案大小 (預設: 340MB)

    Returns:
        Path: 權重檔案的完整路徑
    """
    global _pann_weight_checked

    # 檢查權重檔案路徑
    checkpoint_path = Path.home() / 'panns_data' / 'Cnn14_mAP=0.431.pth'

    # 如果檔案存在且大小正常，直接回傳
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
