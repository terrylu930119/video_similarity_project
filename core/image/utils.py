"""
影像處理工具函式

此模組提供影像處理相關的工具函式，包括：
- 圖像預處理（裁切、正規化）
- 邊緣檢測與字幕帶移除
- 矩陣操作工具
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    """正規化矩陣的每一行。

    功能：
        - 對矩陣的每一行進行 L2 正規化
        - 避免除零錯誤
        - 確保特徵向量的單位長度

    Args:
        x (np.ndarray): 輸入矩陣

    Returns:
        np.ndarray: 正規化後的矩陣
    """
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _scan_vertical_edges(gray: np.ndarray, max_border_frac: float, black_thr: int) -> Tuple[int, int]:
    """掃描垂直方向的邊緣。

    功能：
        - 從上下兩端掃描黑色邊框
        - 計算每行的平均亮度
        - 找到非黑色的內容區域邊界

    Args:
        gray (np.ndarray): 灰度圖像
        max_border_frac (float): 最大邊框比例
        black_thr (int): 黑色閾值

    Returns:
        Tuple[int, int]: (上邊界, 下邊界)
    """
    rows_mean = gray.mean(axis=1)
    h = gray.shape[0]

    top = 0
    while top < int(h * max_border_frac) and rows_mean[top] <= black_thr:
        top += 1

    bottom = h - 1
    while bottom > h - int(h * max_border_frac) - 1 and rows_mean[bottom] <= black_thr:
        bottom -= 1

    return top, bottom


def _scan_horizontal_edges(gray: np.ndarray, max_border_frac: float, black_thr: int) -> Tuple[int, int]:
    """掃描水平方向的邊緣。

    功能：
        - 從左右兩端掃描黑色邊框
        - 計算每列的平均亮度
        - 找到非黑色的內容區域邊界

    Args:
        gray (np.ndarray): 灰度圖像
        max_border_frac (float): 最大邊框比例
        black_thr (int): 黑色閾值

    Returns:
        Tuple[int, int]: (左邊界, 右邊界)
    """
    cols_mean = gray.mean(axis=0)
    w = gray.shape[1]

    left = 0
    while left < int(w * max_border_frac) and cols_mean[left] <= black_thr:
        left += 1

    right = w - 1
    while right > w - int(w * max_border_frac) - 1 and cols_mean[right] <= black_thr:
        right -= 1

    return left, right


def _scan_edge_axis(gray: np.ndarray, axis: int, max_border_frac: float, black_thr: int) -> Tuple[int, int]:
    """掃描指定軸向的邊緣。

    功能：
        - 根據軸向選擇掃描方向
        - 找到非黑色區域的邊界
        - 統一的邊緣掃描介面

    Args:
        gray (np.ndarray): 灰度圖像
        axis (int): 掃描軸向 (0: 垂直, 1: 水平)
        max_border_frac (float): 最大邊框比例
        black_thr (int): 黑色閾值

    Returns:
        Tuple[int, int]: (起始位置, 結束位置)
    """
    if axis == 0:  # top/bottom
        return _scan_vertical_edges(gray, max_border_frac, black_thr)
    else:  # left/right
        return _scan_horizontal_edges(gray, max_border_frac, black_thr)


def _detect_subtitle_band(cropped: np.ndarray, subtitle_frac: float) -> np.ndarray:
    """偵測並移除底部字幕帶。

    功能：
        - 檢測圖像底部是否為字幕帶
        - 根據亮度和方差判斷字幕存在
        - 移除字幕帶以提升特徵品質

    Args:
        cropped (np.ndarray): 裁切後的圖像
        subtitle_frac (float): 字幕帶比例

    Returns:
        np.ndarray: 移除字幕帶後的圖像
    """
    ch, cw = cropped.shape[:2]
    band_h = max(4, int(ch * 0.10))
    band = cv2.cvtColor(cropped[ch - band_h: ch, :, :], cv2.COLOR_RGB2GRAY)

    if band.mean() >= 140 and band.std() >= 25:
        cut = max(1, int(ch * subtitle_frac))
        cropped = cropped[:ch - cut, :, :]

    return cropped


def _autocrop_letterbox(pil_img: Image.Image,
                        black_thr: int = 12,
                        max_border_frac: float = 0.25,
                        subtitle_frac: float = 0.10) -> Image.Image:
    """自動裁切 letterbox 邊框和字幕帶。

    功能：
        - 自動檢測並移除黑色邊框
        - 偵測並移除底部字幕帶
        - 提升特徵提取的準確性
        - 減少無關內容的干擾

    Args:
        pil_img (Image.Image): 輸入的 PIL 圖像
        black_thr (int, optional): 黑色閾值。預設為 12。
        max_border_frac (float, optional): 最大邊框比例。預設為 0.25。
        subtitle_frac (float, optional): 字幕帶比例。預設為 0.10。

    Returns:
        Image.Image: 裁切後的圖像

    Note:
        - 會先轉換為灰度圖像進行分析
        - 確保邊界有效且不超出圖像範圍
    """
    img = np.array(pil_img)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 掃描邊緣
    top, bottom = _scan_edge_axis(gray, 0, max_border_frac, black_thr)
    left, right = _scan_edge_axis(gray, 1, max_border_frac, black_thr)

    # 確保邊界有效
    top = max(0, min(top, h - 2))
    bottom = max(top + 1, min(bottom, h - 1))
    left = max(0, min(left, w - 2))
    right = max(left + 1, min(right, w - 1))

    # 裁切圖像
    cropped = img[top:bottom + 1, left:right + 1, :]

    # 移除字幕帶
    cropped = _detect_subtitle_band(cropped, subtitle_frac)

    return Image.fromarray(cropped)
