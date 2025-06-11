import cv2
import os
import subprocess
import numpy as np
from utils.logger import logger
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import torch
def check_video_file(video_path: str) -> bool:
    """檢查影片檔案是否存在且可以打開"""
    if not os.path.exists(video_path):
        logger.error(f"影片檔案不存在: {video_path}")
        return False
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"無法打開影片檔案: {video_path}")
        return False
        
    cap.release()
    return True

def save_frame(frame_data: tuple) -> str:
    """
    保存單個幀
    
    參數:
        frame_data: (幀數據, 保存路徑) 的元組
    
    返回:
        成功保存的幀路徑，失敗則返回空字符串
    """
    frame, frame_path = frame_data
    try:
        success = cv2.imwrite(frame_path, frame)
        return frame_path if success else ""
    except Exception as e:
        logger.error(f"保存幀時出錯 {frame_path}: {str(e)}")
        return ""

def extract_frames(video_path: str, output_dir: str, time_interval: float = 1.0) -> List[str]:
    """
    使用 FFmpeg 提取視頻幀
    
    參數:
        video_path: 視頻路徑
        output_dir: 輸出目錄
        time_interval: 提取間隔（秒）
    
    返回:
        提取的幀文件路徑列表
    """
    try:
        if not check_video_file(video_path):
            return []
            
        # 從視頻文件名中提取 ID（包含播放清單索引）
        video_basename = os.path.basename(video_path)
        video_id = os.path.splitext(video_basename)[0]  # 去除副檔名，保留 videoId_index
        
        # 創建幀輸出目錄
        frames_dir = os.path.join(output_dir, video_id)
        os.makedirs(frames_dir, exist_ok=True)
        
        # 檢查是否已經存在幀文件
        existing_frames = sorted([
            os.path.join(frames_dir, f) 
            for f in os.listdir(frames_dir) 
            if f.startswith(f'{video_id}_frame_') and f.endswith('.jpg')
        ])
        
        # 如果存在幀文件，檢查它們是否有效
        if existing_frames:
            # 檢查所有幀文件是否都存在且大小不為0
            valid_frames = []
            for frame in existing_frames:
                if os.path.exists(frame) and os.path.getsize(frame) > 0:
                    valid_frames.append(frame)
                else:
                    logger.warning(f"發現無效的幀文件: {frame}")
            
            if valid_frames:
                logger.info(f"使用現有的 {len(valid_frames)} 個幀文件")
                return valid_frames
            else:
                logger.warning("現有的幀文件無效，將重新提取")
        
        # 如果沒有有效的現有幀文件，則進行提取
        output_pattern = os.path.join(frames_dir, f'{video_id}_frame_%04d.jpg')
        
        # 構建基本 FFmpeg 命令
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps=1/{time_interval}',  # 設置提取幀率
            '-frame_pts', '1',                # 添加時間戳
            '-qscale:v', '2',                # 設置輸出質量（2是很好的質量，範圍1-31）
            '-y'                             # 覆蓋已存在的文件
        ]
        
        # 嘗試使用 GPU 加速，如果失敗則回退到 CPU
        if torch.cuda.is_available():
            try:
                gpu_cmd = cmd.copy()
                gpu_cmd.insert(1, '-hwaccel')
                gpu_cmd.insert(2, 'cuda')
                gpu_cmd.append(output_pattern)
                
                logger.info(f"嘗試使用 GPU 加速提取幀，命令: {' '.join(gpu_cmd)}")
                result = subprocess.run(
                    gpu_cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info("成功使用 GPU 加速提取幀")
                
            except subprocess.CalledProcessError as e:
                logger.warning(f"GPU 加速失敗，切換到 CPU 模式: {e.stderr}")
                cmd.append(output_pattern)
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
        else:
            cmd.append(output_pattern)
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
        # 獲取生成的文件列表
        frames = sorted([
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

def get_video_info(video_path: str) -> Tuple[int, int, int, float]:
    """
    取得影片資訊
    
    參數:
        video_path: 影片檔案路徑
    
    返回:
        (總幀數, 寬度, 高度, fps)
    """
    if not check_video_file(video_path):
        return 0, 0, 0, 0
    
    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"影片資訊: 總幀數={total_frames}, 寬度={width}, 高度={height}, FPS={fps}")
        return total_frames, width, height, fps
    except Exception as e:
        logger.error(f"取得影片資訊時發生錯誤: {str(e)}")
        return 0, 0, 0, 0
    finally:
        cap.release()

def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    調整幀大小
    
    參數:
        frame: 輸入幀
        target_size: 目標大小 (width, height)
    
    返回:
        調整大小後的幀
    """
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    預處理幀
    
    參數:
        frame: 輸入幀
    
    返回:
        預處理後的幀
    """
    # 轉換為灰階圖
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 套用高斯模糊減少雜訊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return blurred

def extract_keyframes(video_path: str, output_dir: str, threshold: float = 30.0) -> List[str]:
    """
    提取關鍵幀
    
    參數:
        video_path: 影片檔案路徑
        output_dir: 輸出目錄
        threshold: 差異閾值
    
    返回:
        關鍵幀檔案路徑列表
    """
    if not check_video_file(video_path):
        return []
        
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    keyframe_paths = []
    prev_frame = None
    frame_id = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 預處理當前幀
            processed_frame = preprocess_frame(frame)
            
            if prev_frame is not None:
                # 計算與前一幀的差異
                diff = cv2.absdiff(processed_frame, prev_frame)
                mean_diff = np.mean(diff)
                
                # 如果差異大於閾值，保存為關鍵幀
                if mean_diff > threshold:
                    frame_path = os.path.join(output_dir, f"keyframe_{frame_id:04d}.jpg")
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