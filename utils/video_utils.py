import os
import cv2
import torch
import subprocess
import numpy as np
from utils.logger import logger
from typing import List, Tuple, Union

# =============== 基礎工具：影片檔案檢查 ===============


def check_video_file(video_path: str) -> bool:
    """檢查影片檔案是否存在且可以打開"""
    if not os.path.exists(video_path):
        logger.error(f"影片檔案不存在: {video_path}")
        return False

    cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"無法打開影片檔案: {video_path}")
        return False

    cap.release()
    return True

# =============== 幀儲存與處理 ===============


def save_frame(frame_data: Tuple[np.ndarray, str]) -> str:
    """
    保存單個幀

    參數:
        frame_data: (幀數據, 保存路徑) 的元組

    返回:
        成功保存的幀路徑，失敗則返回空字符串
    """
    frame, frame_path = frame_data
    try:
        success: bool = cv2.imwrite(frame_path, frame)
        return frame_path if success else ""
    except Exception as e:
        logger.error(f"保存幀時出錯 {frame_path}: {str(e)}")
        return ""

# =============== 幀提取（含快取與 GPU 支援） ===============


def extract_frames(video_path: str, output_dir: str, time_interval: float = 1.0) -> List[str]:
    try:
        if not check_video_file(video_path):
            return []

        video_basename: str = os.path.basename(video_path)
        video_id: str = os.path.splitext(video_basename)[0]
        frames_dir: str = os.path.join(output_dir, video_id)
        os.makedirs(frames_dir, exist_ok=True)

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

        output_pattern: str = os.path.join(frames_dir, f'{video_id}_frame_%04d.jpg')
        cmd: List[str] = [
            'ffmpeg', '-i', video_path,
            '-vf', f'fps=1/{time_interval}',
            '-frame_pts', '1',
            '-qscale:v', '2',
            '-y'
        ]

        if torch.cuda.is_available():
            try:
                gpu_cmd: List[str] = cmd.copy()
                gpu_cmd.insert(1, '-hwaccel')
                gpu_cmd.insert(2, 'cuda')
                gpu_cmd.append(output_pattern)

                logger.info(f"嘗試使用 GPU 加速提取幀，命令: {' '.join(gpu_cmd)}")
                subprocess.run(gpu_cmd, check=True, capture_output=True, text=True)
                logger.info("成功使用 GPU 加速提取幀")
            except subprocess.CalledProcessError as e:
                logger.warning(f"GPU 加速失敗，切換到 CPU 模式: {e.stderr}")
                cmd.append(output_pattern)
                subprocess.run(cmd, check=True, capture_output=True, text=True)
        else:
            cmd.append(output_pattern)
            subprocess.run(cmd, check=True, capture_output=True, text=True)

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

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg 執行失敗: {str(e)}")
        if e.stderr:
            logger.error(f"FFmpeg 錯誤輸出: {e.stderr}")
        return []
    except Exception as e:
        logger.error(f"提取幀時出錯: {str(e)}")
        return []

# =============== 影片基本資訊擷取 ===============


def get_video_info(video_path: str) -> Tuple[int, int, int, float]:
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

# =============== 幀尺寸調整 ===============


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

# =============== 幀預處理（灰階 + 模糊） ===============


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred: np.ndarray = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

# =============== 關鍵幀擷取邏輯 ===============


def extract_keyframes(video_path: str, output_dir: str, threshold: float = 30.0) -> List[str]:
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

            if prev_frame is not None:
                diff: np.ndarray = cv2.absdiff(processed_frame, prev_frame)
                mean_diff: float = np.mean(diff)

                if mean_diff > threshold:
                    frame_path: str = os.path.join(output_dir, f"keyframe_{frame_id:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    keyframe_paths.append(frame_path)
                    frame_id += 1

            prev_frame = processed_frame
    except Exception as e:
        logger.error(f"提取關鍵幀時發生錯誤: {str(e)}")
    finally:
        cap.release()

    logger.info(f"從影片中提取了 {len(keyframe_paths)} 個關鍵幀")
    return keyframe_paths
