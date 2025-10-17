"""
對齊模組（alignment）

檔案用途：封裝影片時間對齊相關邏輯（DTW 與其輔助函式）

設計重點：
- 提供穩定、可重用的時間對齊介面（`compute_dtw_alignment` / `execute_dtw_stage`）。
- 僅進行「時間對齊」與其必要的向量前處理；不直接產生相似度分數。
- 與特徵提取（`compute_phash`）與比對融合解耦，便於替換或擴充。
"""

import numpy as np
from fastdtw import fastdtw
from typing import List, Tuple

from utils.logger import logger
from .extractors.phash_extractor import compute_phash
from .utils import _normalize_rows
from .temporal_smoothing import smooth_sequence
from .config import PHASH_CONFIG


def safe_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """安全的餘弦距離計算。

    功能：
        - 計算兩個向量之間的餘弦距離（1 - cosine_similarity）。
        - 對零向量做保護，避免除零造成的數值錯誤。

    Args:
        a (np.ndarray): 第一個向量，shape=(D,)。
        b (np.ndarray): 第二個向量，shape=(D,)。

    Returns:
        float: 餘弦距離（範圍約 0-2）。
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    cosine_sim = float(np.dot(a, b)) / float(norm_a * norm_b)
    return 1.0 - cosine_sim


def pairs_from_dtw_path(frames1: List[str], frames2: List[str],
                         path: List[Tuple[int, int]],
                         max_pairs: int = 400) -> List[Tuple[str, str]]:
    """從 DTW 路徑抽取對齊配對（稀疏抽樣）。

    功能：
        - 沿著 DTW 對齊路徑擷取對應的幀檔名配對，並以固定步長做稀疏化。
        - 避免重複索引，控制配對數量以便後續批次特徵計算。

    Args:
        frames1 (List[str]): 影片 1 的採樣幀檔案路徑列表。
        frames2 (List[str]): 影片 2 的採樣幀檔案路徑列表。
        path (List[Tuple[int, int]]): DTW 對齊路徑（索引配對序列）。
        max_pairs (int): 最多抽取的配對數量。

    Returns:
        List[Tuple[str, str]]: 對齊的幀檔名配對列表（已稀疏化）。
    """
    if not path:
        return []
    step = max(1, len(path) // max_pairs)
    picked: List[Tuple[str, str]] = []
    last_i, last_j = -1, -1
    for k in range(0, len(path), step):
        i, j = path[k]
        if i == last_i and j == last_j:
            continue
        if 0 <= i < len(frames1) and 0 <= j < len(frames2):
            picked.append((frames1[i], frames2[j]))
            last_i, last_j = i, j
    return picked


def _compute_dtw_alignment(sampled_frames1: List[str], sampled_frames2: List[str],
                           batch_size: int,
                           enable_temporal_smoothing: bool = False,
                           smoothing_method: str = "ema",
                           ema_alpha: float = 0.7,
                           majority_window: int = 3) -> Tuple[float, List[Tuple[int, int]], float, int]:
    """計算 DTW 對齊。

    功能：
        - 基於 pHash 特徵向量進行動態時間規整（DTW）以對齊兩段影片的時間軸。
        - 僅回傳對齊品質、路徑、覆蓋率與延遲資訊，不產生相似度分數。

    設計理由：
        - 對齊使用與比對相同的 pHash 模態，有助於降低模態落差造成的漂移。
        - 先行正規化各幀向量，確保距離度量的一致性與數值穩定性。

    Args:
        sampled_frames1 (List[str]): 影片 1 的採樣幀檔案路徑列表。
        sampled_frames2 (List[str]): 影片 2 的採樣幀檔案路徑列表。
        batch_size (int): 批次大小（保留參數以便未來在此函式內做批次化計算）。

    Returns:
        Tuple[float, List[Tuple[int, int]], float, int]:
            - alignment_quality (float): 對齊品質分數（由 DTW 距離轉換，僅供權重調整）。
            - path (List[Tuple[int, int]]): DTW 對齊路徑（幀索引配對序列）。
            - coverage (float): 覆蓋率（兩側被對齊到的索引比例的最小值）。
            - lag_frames (int): 估計的延遲（j-i 的中位數，以幀為單位）。
    """
    phash_features1: List[np.ndarray] = []
    phash_features2: List[np.ndarray] = []

    for frame in sampled_frames1:
        phash_feat = compute_phash(frame)
        if phash_feat is not None and len(phash_feat) == 3:
            gray_vec = phash_feat[0].flatten()   # 32*32 = 1024
            edge_vec = phash_feat[1].flatten()   # 32*32 = 1024
            hsv_vec = phash_feat[2].flatten()    # 64*64*2 = 8192
            phash_features1.append(np.concatenate([gray_vec, edge_vec, hsv_vec]))
        else:
            phash_features1.append(np.zeros(PHASH_CONFIG['phash_feature_dim']))

    for frame in sampled_frames2:
        phash_feat = compute_phash(frame)
        if phash_feat is not None and len(phash_feat) == 3:
            gray_vec = phash_feat[0].flatten()
            edge_vec = phash_feat[1].flatten()
            hsv_vec = phash_feat[2].flatten()
            phash_features2.append(np.concatenate([gray_vec, edge_vec, hsv_vec]))
        else:
            phash_features2.append(np.zeros(PHASH_CONFIG['phash_feature_dim']))

    if not phash_features1 or not phash_features2:
        logger.warning("pHash 特徵提取失敗，DTW 對齊略過")
        return 0.0, [], 0.0, 0

    # 可選：時間平滑（僅穩定化，不降維）
    if enable_temporal_smoothing:
        phash_features1 = smooth_sequence(phash_features1, method=smoothing_method,
                                          alpha=ema_alpha, window=majority_window)
        phash_features2 = smooth_sequence(phash_features2, method=smoothing_method,
                                          alpha=ema_alpha, window=majority_window)

    seq_emb1 = _normalize_rows(np.asarray(phash_features1))
    seq_emb2 = _normalize_rows(np.asarray(phash_features2))

    dtw_dist, path = fastdtw(seq_emb1, seq_emb2, dist=safe_cosine_distance)

    alignment_quality = 1.0 / (1.0 + dtw_dist / max(len(seq_emb1), len(seq_emb2)))

    lags = [j - i for i, j in path]
    lag_frames = int(np.median(lags)) if lags else 0

    covered_i = len({i for i, _ in path})
    covered_j = len({j for _, j in path})
    coverage = min(covered_i / max(1, len(seq_emb1)), covered_j / max(1, len(seq_emb2)))

    logger.info(f"DTW 對齊完成: 品質={alignment_quality:.3f}, 覆蓋率={coverage:.3f}, 延遲={lag_frames}幀")

    return alignment_quality, path, coverage, lag_frames


def execute_dtw_stage(sampled_frames1: List[str], sampled_frames2: List[str],
                       batch_size: int,
                       enable_temporal_smoothing: bool = False,
                       smoothing_method: str = "ema",
                       ema_alpha: float = 0.7,
                       majority_window: int = 3) -> Tuple[float, List[Tuple[int, int]], float, int]:
    """執行第二階段：DTW 對齊（包裝呼叫）。

    功能：
        - 呼叫 `compute_dtw_alignment` 完成對齊；若無路徑則回傳空結果。

    Args:
        sampled_frames1 (List[str]): 影片 1 的採樣幀檔案路徑列表。
        sampled_frames2 (List[str]): 影片 2 的採樣幀檔案路徑列表。
        batch_size (int): 批次大小。

    Returns:
        Tuple[float, List[Tuple[int, int]], float, int]: 同 `compute_dtw_alignment`。

    Raises:
        無。
    """
    alignment_quality, path, coverage, lag_frames = _compute_dtw_alignment(
        sampled_frames1, sampled_frames2, batch_size,
        enable_temporal_smoothing=enable_temporal_smoothing,
        smoothing_method=smoothing_method,
        ema_alpha=ema_alpha,
        majority_window=majority_window,
    )
    if not path:
        return 0.0, [], 0.0, 0
    return alignment_quality, path, coverage, lag_frames
