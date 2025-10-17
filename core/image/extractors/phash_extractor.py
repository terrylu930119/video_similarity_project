"""
感知哈希特徵提取器

此模組提供感知哈希（pHash）特徵提取功能，包括：
- GPU 和 CPU 兩種計算方式
- 多種特徵的哈希編碼（灰度、邊緣、顏色）
- 特徵快取機制
"""

import cv2
import torch
import numpy as np
from functools import lru_cache
from typing import Optional, Tuple
from utils.logger import logger
from utils.gpu_utils import gpu_manager
from ..config import PHASH_CONFIG


def _compute_gpu_phash(gray: np.ndarray, edges: np.ndarray,
                       hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用 GPU 計算感知哈希。

    功能：
        - 在 GPU 上計算三種特徵的感知哈希
        - 使用 FFT 進行頻域分析
        - 提取低頻特徵進行哈希編碼
        - 支援 GPU 加速計算

    Args:
        gray (np.ndarray): 灰度圖像
        edges (np.ndarray): 邊緣圖像
        hsv (np.ndarray): HSV 圖像

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 三種特徵的哈希
    """
    device = gpu_manager.get_device()

    gray_tensor = torch.from_numpy(gray).float().to(device)
    edge_tensor = torch.from_numpy(edges).float().to(device)
    hsv_tensor = torch.from_numpy(hsv[:, :, :2]).float().to(device)

    # 計算 DCT 並提取低頻部分
    gray_dct = torch.fft.rfft2(gray_tensor)
    gray_dct_low = torch.abs(gray_dct[:PHASH_CONFIG['dct_low_freq_size'], :PHASH_CONFIG['dct_low_freq_size']])
    gray_threshold = torch.mean(gray_dct_low.cpu()) + PHASH_CONFIG['gray_threshold_factor'] * torch.std(gray_dct_low.cpu())
    gray_hash = (gray_dct_low > gray_threshold.to(device)).cpu().numpy()

    edge_dct = torch.fft.rfft2(edge_tensor)
    edge_dct_low = torch.abs(edge_dct[:PHASH_CONFIG['dct_low_freq_size'], :PHASH_CONFIG['dct_low_freq_size']])
    edge_mean = torch.mean(edge_dct_low.cpu())
    edge_hash = (edge_dct_low > edge_mean.to(device)).cpu().numpy()

    hsv_mean = torch.mean(hsv_tensor, dim=(0, 1))
    hsv_hash = (hsv_tensor > hsv_mean.reshape(1, 1, -1)).cpu().numpy()

    return gray_hash, edge_hash, hsv_hash


def _compute_cpu_phash(gray: np.ndarray, edges: np.ndarray,
                       hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用 CPU 計算感知哈希。

    功能：
        - 在 CPU 上計算三種特徵的感知哈希
        - 使用 OpenCV 的 DCT 變換
        - 提取低頻特徵進行哈希編碼
        - 作為 GPU 計算的備援方案

    Args:
        gray (np.ndarray): 灰度圖像
        edges (np.ndarray): 邊緣圖像
        hsv (np.ndarray): HSV 圖像

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 三種特徵的哈希
    """
    # 計算 DCT 並提取低頻部分
    gray_dct = cv2.dct(np.float32(gray))[:PHASH_CONFIG['dct_low_freq_size'], :PHASH_CONFIG['dct_low_freq_size']]
    gray_hash = gray_dct > np.mean(gray_dct) + PHASH_CONFIG['gray_threshold_factor'] * np.std(gray_dct)

    edge_dct = cv2.dct(np.float32(edges))[:PHASH_CONFIG['dct_low_freq_size'], :PHASH_CONFIG['dct_low_freq_size']]
    edge_hash = edge_dct > np.mean(edge_dct)

    hsv_mean = np.mean(hsv[:, :, :2], axis=(0, 1))
    hsv_hash = hsv[:, :, :2] > hsv_mean.reshape(1, 1, -1)

    return gray_hash, edge_hash, hsv_hash


@lru_cache(maxsize=1024)
def compute_phash(image_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """計算感知哈希特徵。

    功能：
        - 計算灰度、邊緣與顏色三種特徵的感知哈希
        - 支援 GPU 和 CPU 兩種計算方式
        - 使用 LRU 快取提升效能
        - 預處理圖像以提升特徵品質

    Args:
        image_path (str): 圖像檔案路徑

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]: 三種特徵的哈希，失敗時回傳 None

    Note:
        - 會對圖像進行高斯模糊和縮放預處理
        - 根據可用裝置自動選擇計算方式
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        # 預處理圖像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, PHASH_CONFIG['gaussian_blur_kernel'], PHASH_CONFIG['gaussian_blur_sigma'])
        gray = cv2.resize(gray, (PHASH_CONFIG['resize_size'], PHASH_CONFIG['resize_size']))

        edges = cv2.Canny(gray, PHASH_CONFIG['canny_low_threshold'], PHASH_CONFIG['canny_high_threshold'])
        edges = cv2.resize(edges, (PHASH_CONFIG['resize_size'], PHASH_CONFIG['resize_size']))

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = cv2.resize(hsv, (PHASH_CONFIG['resize_size'], PHASH_CONFIG['resize_size']))

        # 根據設備選擇計算方式
        if gpu_manager.get_device().type == "cuda":
            return _compute_gpu_phash(gray, edges, hsv)
        else:
            return _compute_cpu_phash(gray, edges, hsv)

    except Exception as e:
        logger.error(f"計算多重特徵哈希時出錯 {image_path}: {str(e)}")
        return None
