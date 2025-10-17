"""
時間序列平滑（Temporal Smoothing）

檔案用途：提供針對逐幀向量序列的輕量平滑方法，以提升異源片源條件下的穩定性，
例如重編碼帶來的逐幀抖動、亮度小擾動等。此模組僅做「穩定化」，不做降維或聚段。

提供方法：
- 指數移動平均（EMA）：連續值平滑，對突發雜訊具抗性，保序且計算量低。
- 多數表決（Majority）：針對 0/1 bit 表示做小窗投票，提升二值穩定度。
"""

from typing import List
import numpy as np


def ema_smooth_sequence(vectors: List[np.ndarray], alpha: float = 0.7) -> List[np.ndarray]:
    """對向量序列進行 EMA 平滑。

    功能：
        - 對每一維度做因果的指數移動平均，輸出長度與輸入一致。
        - 適用於連續值或 0/1 浮點化後的向量表示。

    Args:
        vectors (List[np.ndarray]): 長度為 T 的向量序列，每個 shape=(D,)。
        alpha (float): 平滑係數，越接近 1 越依賴過去（建議 0.6-0.8）。

    Returns:
        List[np.ndarray]: EMA 後的向量序列（同長度、同維度）。
    """
    if not vectors:
        return vectors
    out: List[np.ndarray] = []
    prev = vectors[0].astype(np.float32)
    out.append(prev.copy())
    for t in range(1, len(vectors)):
        x = vectors[t].astype(np.float32)
        prev = alpha * prev + (1.0 - alpha) * x
        out.append(prev.copy())
    return out


def majority_smooth_sequence(vectors: List[np.ndarray], window: int = 3) -> List[np.ndarray]:
    """對 0/1 向量序列做多數表決平滑（滑動視窗）。

    功能：
        - 以因果視窗對每個時間點取過去 `window` 幀的平均後二值化（>=0.5 → 1）。
        - 適用於 bit/binary 表示；若為連續值會先視為 [0,1] 後門檻化。

    Args:
        vectors (List[np.ndarray]): 長度為 T 的向量序列，每個 shape=(D,)。
        window (int): 視窗大小（建議 3-5）。

    Returns:
        List[np.ndarray]: 多數表決後的向量序列（同長度、同維度，元素為 {0,1}）。
    """
    if not vectors:
        return vectors
    T = len(vectors)
    D = vectors[0].shape[0]
    M = np.stack([v.astype(np.float32) for v in vectors], axis=0)  # [T, D]
    out = []
    for t in range(T):
        s = max(0, t - window + 1)
        votes = M[s:t + 1].mean(axis=0)
        out.append((votes >= 0.5).astype(np.float32))
    return out


def smooth_sequence(vectors: List[np.ndarray], method: str = "ema",
                    alpha: float = 0.7, window: int = 3) -> List[np.ndarray]:
    """通用平滑入口。

    Args:
        vectors (List[np.ndarray]): 長度為 T 的向量序列，每個 shape=(D,)。
        method (str): "ema" 或 "majority"。
        alpha (float): EMA 平滑係數。
        window (int): 多數表決視窗大小。

    Returns:
        List[np.ndarray]: 平滑後的向量序列。
    """
    if method == "ema":
        return ema_smooth_sequence(vectors, alpha=alpha)
    if method == "majority":
        return majority_smooth_sequence(vectors, window=window)
    return vectors


