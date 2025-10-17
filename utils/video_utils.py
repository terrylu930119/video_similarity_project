# utils/video_utils.py
"""
檔案用途：影片處理與幀提取工具

此模組提供影片處理相關功能，包括：
- 影片檔案驗證與基本資訊擷取
- 幀提取（支援 GPU 加速）
- 關鍵幀擷取與分析
- 幀預處理與尺寸調整

主要功能：
- check_video_file: 影片檔案有效性檢查
- extract_frames: 從影片中提取幀（支援快取）
- extract_keyframes: 關鍵幀擷取
- get_video_info: 影片基本資訊擷取
- 各種幀處理工具函式
"""

import os
import cv2
import torch
import subprocess
import numpy as np
from utils.logger import logger
from typing import List, Tuple, Union

# ======================== 基礎工具：影片檔案檢查 ========================


def check_video_file(video_path: str) -> bool:
    """檢查影片檔案是否存在且可以打開。

    功能：
        - 驗證檔案是否存在
        - 嘗試使用 OpenCV 開啟影片檔案
        - 檢查檔案是否可正常讀取

    Args:
        video_path (str): 影片檔案路徑

    Returns:
        bool: 影片檔案是否有效且可讀取

    Note:
        - 會記錄錯誤訊息到日誌
        - 檢查完成後會釋放影片資源
    """
    if not os.path.exists(video_path):
        logger.error(f"影片檔案不存在: {video_path}")
        return False

    cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"無法打開影片檔案: {video_path}")
        return False

    cap.release()
    return True


# ======================== 幀儲存與處理 ========================
def save_frame(frame_data: Tuple[np.ndarray, str]) -> str:
    """
    保存單個幀

    Args:
        frame_data: (幀數據, 保存路徑) 的元組

    Returns:
        str: 成功保存的幀路徑，失敗則返回空字符串
    """
    frame, frame_path = frame_data
    try:
        success: bool = cv2.imwrite(frame_path, frame)
        return frame_path if success else ""
    except Exception as e:
        logger.error(f"保存幀時出錯 {frame_path}: {str(e)}")
        return ""


# ======================== 幀提取（含快取與 GPU 支援） ========================
def _setup_frame_extraction_environment(video_path: str, output_dir: str) -> Tuple[str, str, str]:
    """
    設定幀提取環境

    Args:
        video_path: 影片路徑
        output_dir: 輸出目錄

    Returns:
        Tuple[str, str, str]: (影片 ID, 幀目錄, 輸出模式)
    """
    video_basename: str = os.path.basename(video_path)
    video_id: str = os.path.splitext(video_basename)[0]
    frames_dir: str = os.path.join(output_dir, video_id)
    os.makedirs(frames_dir, exist_ok=True)

    output_pattern: str = os.path.join(frames_dir, f'{video_id}_frame_%04d.jpg')
    return video_id, frames_dir, output_pattern


def _check_existing_frames(frames_dir: str, video_id: str) -> List[str]:
    """
    檢查現有的幀檔案

    Args:
        frames_dir: 幀目錄
        video_id: 影片 ID

    Returns:
        List[str]: 現有幀檔案路徑列表
    """
    existing_frames: List[str] = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.startswith(f'{video_id}_frame_') and f.endswith('.jpg')
    ])

    if existing_frames:
        valid_frames: List[str] = [
            f for f in existing_frames if os.path.exists(f) and os.path.getsize(f) > 0]
        if valid_frames:
            logger.info(f"使用現有的 {len(valid_frames)} 個幀文件")
            return valid_frames
        else:
            logger.warning("現有的幀文件無效，將重新提取")

    return []


def _build_ffmpeg_command(video_path: str, time_interval: float) -> List[str]:
    """
    建立 FFmpeg 命令

    Args:
        video_path: 影片路徑
        time_interval: 時間間隔

    Returns:
        List[str]: FFmpeg 命令參數列表
    """
    return [
        'ffmpeg', '-i', video_path,
        '-vf', f'fps=1/{time_interval}',
        '-frame_pts', '1',
        '-qscale:v', '2',
        '-y'
    ]


def _try_gpu_acceleration(cmd: List[str], output_pattern: str) -> bool:
    """
    嘗試使用 GPU 加速

    Args:
        cmd: 基礎 FFmpeg 命令
        output_pattern: 輸出檔案模式

    Returns:
        bool: GPU 加速是否成功
    """
    if not torch.cuda.is_available():
        return False

    try:
        gpu_cmd: List[str] = cmd.copy()
        gpu_cmd.insert(1, '-hwaccel')
        gpu_cmd.insert(2, 'cuda')
        gpu_cmd.append(output_pattern)

        logger.info(f"嘗試使用 GPU 加速提取幀，命令: {' '.join(gpu_cmd)}")
        subprocess.run(gpu_cmd, check=True, capture_output=True, text=True)
        logger.info("成功使用 GPU 加速提取幀")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"GPU 加速失敗，切換到 CPU 模式: {e.stderr}")
        return False


def _extract_frames_with_ffmpeg(cmd: List[str], output_pattern: str) -> None:
    """
    使用 FFmpeg 提取幀

    Args:
        cmd: FFmpeg 命令
        output_pattern: 輸出檔案模式
    """
    cmd.append(output_pattern)
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _collect_extracted_frames(frames_dir: str, video_id: str) -> List[str]:
    """
    收集提取的幀檔案

    Args:
        frames_dir: 幀目錄
        video_id: 影片 ID

    Returns:
        List[str]: 幀檔案路徑列表
    """
    frames: List[str] = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.startswith(f'{video_id}_frame_') and f.endswith('.jpg')
    ])

    if not frames:
        logger.error("未生成任何幀文件")
        return []

    logger.info(f"成功提取 {len(frames)} 個幀")
    return frames


def extract_frames(video_path: str, output_dir: str, time_interval: float = 1.0) -> List[str]:
    """從影片中提取幀，支援快取與 GPU 加速。

    功能：
        - 檢查影片檔案有效性
        - 檢查現有幀檔案（快取機制）
        - 使用 FFmpeg 提取幀（支援 GPU 加速）
        - 收集並回傳提取的幀檔案路徑

    Args:
        video_path (str): 影片檔案路徑
        output_dir (str): 輸出目錄路徑
        time_interval (float, optional): 幀提取時間間隔（秒）。預設為 1.0。

    Returns:
        List[str]: 提取的幀檔案路徑列表

    Raises:
        subprocess.CalledProcessError: 當 FFmpeg 執行失敗時
        Exception: 當其他處理錯誤發生時

    Note:
        - 支援 GPU 加速（如果可用）
        - 具有快取機制，避免重複提取
        - 會記錄處理過程到日誌
    """
    try:
        # 檢查影片檔案
        if not check_video_file(video_path):
            return []

        # 設定環境
        video_id, frames_dir, output_pattern = _setup_frame_extraction_environment(video_path, output_dir)

        # 檢查現有幀
        existing_frames = _check_existing_frames(frames_dir, video_id)
        if existing_frames:
            return existing_frames

        # 建立 FFmpeg 命令
        cmd = _build_ffmpeg_command(video_path, time_interval)

        # 嘗試 GPU 加速，失敗則使用 CPU
        if not _try_gpu_acceleration(cmd, output_pattern):
            _extract_frames_with_ffmpeg(cmd, output_pattern)

        # 收集結果
        return _collect_extracted_frames(frames_dir, video_id)

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg 執行失敗: {str(e)}")
        if e.stderr:
            logger.error(f"FFmpeg 錯誤輸出: {e.stderr}")
        return []
    except Exception as e:
        logger.error(f"提取幀時出錯: {str(e)}")
        return []


# ======================== 影片基本資訊擷取 ========================
def get_video_info(video_path: str) -> Tuple[int, int, int, float]:
    """
    獲取影片基本資訊

    Args:
        video_path: 影片路徑

    Returns:
        Tuple[int, int, int, float]: (總幀數, 寬度, 高度, FPS)
    """
    if not check_video_file(video_path):
        return 0, 0, 0, 0.0

    cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
    try:
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps: float = cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"影片資訊: 總幀數={total_frames}, 寬度={width}, 高度={height}, FPS={fps}")
        return total_frames, width, height, fps
    except Exception as e:
        logger.error(f"取得影片資訊時發生錯誤: {str(e)}")
        return 0, 0, 0, 0.0
    finally:
        cap.release()


# ======================== 幀尺寸調整 ========================
def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    調整幀尺寸

    Args:
        frame: 輸入幀
        target_size: 目標尺寸 (寬度, 高度)

    Returns:
        np.ndarray: 調整後的幀
    """
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


# ======================== 幀預處理（灰階 + 模糊） ========================
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    預處理幀：轉換為灰階並應用高斯模糊

    Args:
        frame: 輸入幀

    Returns:
        np.ndarray: 預處理後的幀
    """
    gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred: np.ndarray = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


# ======================== 關鍵幀擷取邏輯 ========================
def _is_keyframe(processed_frame: np.ndarray, prev_frame: np.ndarray, threshold: float) -> bool:
    """
    判斷是否為關鍵幀

    Args:
        processed_frame: 當前處理後的幀
        prev_frame: 前一幀
        threshold: 差異閾值

    Returns:
        bool: 是否為關鍵幀
    """
    if prev_frame is None:
        return True

    diff: np.ndarray = cv2.absdiff(processed_frame, prev_frame)
    mean_diff: float = np.mean(diff)
    return mean_diff > threshold


def _save_keyframe(frame: np.ndarray, output_dir: str, frame_id: int) -> str:
    """
    保存關鍵幀

    Args:
        frame: 幀數據
        output_dir: 輸出目錄
        frame_id: 幀 ID

    Returns:
        str: 保存的檔案路徑
    """
    frame_path: str = os.path.join(output_dir, f"keyframe_{frame_id:04d}.jpg")
    cv2.imwrite(frame_path, frame)
    return frame_path


def extract_keyframes(video_path: str, output_dir: str, threshold: float = 30.0) -> List[str]:
    """從影片中提取關鍵幀，基於幀間差異分析。

    功能：
        - 逐幀讀取影片
        - 對每幀進行預處理（灰階轉換、高斯模糊）
        - 計算與前一幀的差異
        - 根據閾值判斷是否為關鍵幀
        - 保存關鍵幀到指定目錄

    Args:
        video_path (str): 影片檔案路徑
        output_dir (str): 輸出目錄路徑
        threshold (float, optional): 關鍵幀差異閾值。預設為 30.0。

    Returns:
        List[str]: 關鍵幀檔案路徑列表

    Note:
        - 使用灰階轉換與高斯模糊進行預處理
        - 基於幀間平均差異判斷關鍵幀
        - 會記錄提取的關鍵幀數量
    """
    if not check_video_file(video_path):
        return []

    os.makedirs(output_dir, exist_ok=True)
    cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
    keyframe_paths: List[str] = []
    prev_frame: Union[np.ndarray, None] = None
    frame_id: int = 0

    try:
        while True:
            ret: bool
            frame: np.ndarray
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame: np.ndarray = preprocess_frame(frame)

            if _is_keyframe(processed_frame, prev_frame, threshold):
                frame_path = _save_keyframe(frame, output_dir, frame_id)
                keyframe_paths.append(frame_path)
                frame_id += 1

            prev_frame = processed_frame
    except Exception as e:
        logger.error(f"提取關鍵幀時發生錯誤: {str(e)}")
    finally:
        cap.release()

    logger.info(f"從影片中提取了 {len(keyframe_paths)} 個關鍵幀")
    return keyframe_paths
