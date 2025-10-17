"""
影像相似度計算

此模組提供影像相似度計算功能，包括：
- 感知哈希相似度計算
- DTW 時間序列對齊
- 影片相似度計算主流程
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from utils.logger import logger
from .extractors.phash_extractor import compute_phash
from .extractors.deep_extractor import compute_batch_embeddings
from .utils import _normalize_rows
from .alignment import (
    execute_dtw_stage,
    pairs_from_dtw_path,
)
from .config import (
    SAMPLING_CONFIG,
    DTW_CONFIG,
    SIMILARITY_WEIGHTS,
    BATCH_CONFIG,
    TEMPORAL_SMOOTHING_CONFIG
)


def fast_similarity(feat1: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    feat2: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    """快速比對 pHash 特徵的綜合相似度。

    功能：
        - 計算三種特徵的相似度分數
        - 使用加權平均融合結果
        - 處理無效特徵的情況

    Args:
        feat1 (Tuple[np.ndarray, np.ndarray, np.ndarray]): 第一個特徵組
        feat2 (Tuple[np.ndarray, np.ndarray, np.ndarray]): 第二個特徵組

    Returns:
        float: 綜合相似度分數（0-1）

    Note:
        - 權重：灰度 0.5，邊緣 0.3，顏色 0.2
        - 無效特徵時回傳 0.0
    """
    if all(f1 is not None and f2 is not None for f1, f2 in zip(feat1, feat2)):
        gray_sim = 1 - np.count_nonzero(feat1[0] != feat2[0]) / feat1[0].size
        edge_sim = 1 - np.count_nonzero(feat1[1] != feat2[1]) / feat1[1].size
        hsv_sim = 1 - np.count_nonzero(feat1[2] != feat2[2]) / feat1[2].size
        return (gray_sim * SIMILARITY_WEIGHTS['gray_weight'] + 
                edge_sim * SIMILARITY_WEIGHTS['edge_weight'] + 
                hsv_sim * SIMILARITY_WEIGHTS['hsv_weight'])
    return 0.0


def _calculate_sampling_parameters(video_duration: float) -> Tuple[int, float]:
    """根據影片長度計算採樣參數。

    功能：
        - 根據影片長度動態調整採樣策略
        - 短影片使用密集採樣
        - 長影片使用稀疏採樣
        - 設定相應的 pHash 閾值

    Args:
        video_duration (float): 影片長度（秒）

    Returns:
        Tuple[int, float]: (採樣間隔, pHash 閾值)
    """
    if video_duration <= 60:
        return SAMPLING_CONFIG['short_video_interval'], SAMPLING_CONFIG['short_video_threshold']
    elif video_duration <= 300:
        return SAMPLING_CONFIG['medium_video_interval'], SAMPLING_CONFIG['medium_video_threshold']
    else:
        return SAMPLING_CONFIG['long_video_interval'], SAMPLING_CONFIG['long_video_threshold']


def _find_max_similarity(phash1: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         valid2: List[Tuple[int, str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]) -> float:
    """找到與指定 pHash 的最大相似度。

    功能：
        - 計算與所有有效 pHash 的相似度
        - 找出最高的相似度分數
        - 用於粗估階段的快速比對

    Args:
        phash1 (Tuple[np.ndarray, np.ndarray, np.ndarray]): 第一個 pHash 特徵
        valid2 (List[Tuple[int, str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]): 第二個影片的有效 pHash 列表

    Returns:
        float: 最大相似度分數
    """
    max_sim = 0.0
    for _, _, p2 in valid2:
        s = fast_similarity(phash1, p2)
        if s > max_sim:
            max_sim = s
    return max_sim


def _extract_valid_phash_data(sampled_frames: List[str], phash_results: List[Optional[Tuple[np.ndarray,
                              np.ndarray, np.ndarray]]]) -> List[Tuple[int, str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """提取有效的 pHash 資料。

    功能：
        - 過濾掉計算失敗的 pHash 結果
        - 保留有效的特徵資料
        - 建立索引對應關係

    Args:
        sampled_frames (List[str]): 採樣幀列表
        phash_results (List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]): pHash 計算結果列表

    Returns:
        List[Tuple[int, str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]: 有效的 pHash 資料
    """
    return [(i, f, p) for i, (f, p) in enumerate(zip(sampled_frames, phash_results)) if p is not None]


def _compute_phash_similarities(sampled_frames1: List[str], sampled_frames2: List[str]) -> Tuple[List[float], float]:
    """計算 pHash 相似度。

    功能：
        - 並行計算所有幀的 pHash 特徵
        - 找出每個幀的最佳匹配
        - 計算平均相似度分數

    Args:
        sampled_frames1 (List[str]): 採樣幀列表 1
        sampled_frames2 (List[str]): 採樣幀列表 2

    Returns:
        Tuple[List[float], float]: (相似度列表, 平均相似度)
    """
    with ThreadPoolExecutor() as executor:
        phash1 = list(executor.map(compute_phash, sampled_frames1))
        phash2 = list(executor.map(compute_phash, sampled_frames2))

    valid1 = _extract_valid_phash_data(sampled_frames1, phash1)
    valid2 = _extract_valid_phash_data(sampled_frames2, phash2)

    if not valid1 or not valid2:
        return [], 0.0

    phash_similarities = []
    for _, _, p1 in valid1:
        max_sim = _find_max_similarity(p1, valid2)
        phash_similarities.append(max_sim)

    avg_phash = float(np.mean(phash_similarities)) if phash_similarities else 0.0
    return phash_similarities, avg_phash


def _calculate_truncated_mean(values: np.ndarray, min_length: int = None, ratio: float = None) -> float:
    """計算截尾平均。

    功能：
        - 計算數值陣列的截尾平均
        - 移除極值以提升穩定性
        - 根據長度和比例動態調整

    Args:
        values (np.ndarray): 數值陣列
        min_length (int, optional): 最小長度閾值。預設使用配置值。
        ratio (float, optional): 截尾比例。預設使用配置值。

    Returns:
        float: 截尾平均
    """
    if min_length is None:
        min_length = SAMPLING_CONFIG['truncated_mean_min_length']
    if ratio is None:
        ratio = SAMPLING_CONFIG['truncated_mean_ratio']
        
    if len(values) >= min_length:
        k = max(SAMPLING_CONFIG['truncated_mean_min_k'], int(ratio * len(values)))
        return float(np.mean(np.partition(values, -k)[-k:]))
    else:
        return float(np.mean(values))


def _compute_deep_similarity(aligned_pairs: List[Tuple[str, str]], batch_size: int) -> float:
    """計算深度相似度。

    功能：
        - 對對齊的幀對計算深度特徵
        - 計算逐幀的餘弦相似度
        - 使用截尾平均提升穩定性

    Args:
        aligned_pairs (List[Tuple[str, str]]): 對齊的幀對
        batch_size (int): 批次大小

    Returns:
        float: 深度相似度分數
    """
    e1 = compute_batch_embeddings([a for a, _ in aligned_pairs], batch_size)
    e2 = compute_batch_embeddings([b for _, b in aligned_pairs], batch_size)

    if e1 is None or e2 is None:
        return 0.0

    e1 = _normalize_rows(e1)
    e2 = _normalize_rows(e2)

    per_frame_cos = np.sum(e1 * e2, axis=1)
    per_frame_cos = np.clip(per_frame_cos, -1.0, 1.0)

    return _calculate_truncated_mean(per_frame_cos)


def _compute_phash_aligned_similarity(aligned_pairs: List[Tuple[str, str]], avg_phash: float) -> float:
    """計算對齊後的 pHash 相似度。

    功能：
        - 對對齊的幀對計算 pHash 相似度
        - 使用截尾平均提升穩定性
        - 回退到平均相似度

    Args:
        aligned_pairs (List[Tuple[str, str]]): 對齊的幀對
        avg_phash (float): 平均 pHash 相似度

    Returns:
        float: 對齊後的 pHash 相似度
    """
    ph_sims = []
    for a, b in aligned_pairs:
        pa = compute_phash(a)
        pb = compute_phash(b)
        if pa is not None and pb is not None:
            ph_sims.append(fast_similarity(pa, pb))

    if ph_sims:
        ph_sims = np.asarray(ph_sims, dtype=float)
        return _calculate_truncated_mean(ph_sims)
    else:
        return avg_phash


def _compute_aligned_similarities(aligned_pairs: List[Tuple[str, str]],
                                  batch_size: int, avg_phash: float) -> Tuple[float, float]:
    """計算對齊後的相似度。

    功能：
        - 計算對齊幀對的深度相似度
        - 計算對齊幀對的 pHash 相似度
        - 處理無配對的情況

    Args:
        aligned_pairs (List[Tuple[str, str]]): 對齊的幀對
        batch_size (int): 批次大小
        avg_phash (float): 平均 pHash 相似度

    Returns:
        Tuple[float, float]: (深度相似度, pHash 相似度)
    """
    if not aligned_pairs:
        logger.warning("DTW 對齊後無配對，回退到 avg_phash")
        return 0.0, avg_phash

    # 計算深度相似度
    deep_aligned = _compute_deep_similarity(aligned_pairs, batch_size)

    # 計算對齊後的 pHash 相似度
    phash_aligned = _compute_phash_aligned_similarity(aligned_pairs, avg_phash)

    return deep_aligned, phash_aligned


def _calculate_final_similarity(deep_aligned: float, phash_aligned: float, alignment_quality: float,
                                avg_phash: float, coverage: float) -> float:
    """計算最終相似度分數。

    功能：
        - 基於對齊後的畫面相似度計算最終分數
        - 高覆蓋率時使用對齊後分數
        - 低覆蓋率時使用平均分數
        - 應用覆蓋率權重

    Args:
        deep_aligned (float): 對齊後的深度相似度
        phash_aligned (float): 對齊後的 pHash 相似度
        alignment_quality (float): DTW 對齊品質（用於權重調整，不直接參與相似度計算）
        avg_phash (float): 平均 pHash 相似度
        coverage (float): 覆蓋率

    Returns:
        float: 最終相似度分數
    """
    # 高品質對齊：使用對齊後的分數
    if coverage >= DTW_CONFIG['high_quality_coverage_threshold'] and alignment_quality >= DTW_CONFIG['high_quality_alignment_threshold']:
        # 對齊品質高時，主要使用對齊後的分數
        base_sim = (SIMILARITY_WEIGHTS['high_quality_primary_weight'] * max(deep_aligned, phash_aligned) + 
                   SIMILARITY_WEIGHTS['high_quality_secondary_weight'] * min(deep_aligned, phash_aligned))
    else:
        # 對齊品質低時，使用深度特徵和平均 pHash
        base_sim = (SIMILARITY_WEIGHTS['low_quality_deep_weight'] * deep_aligned + 
                   SIMILARITY_WEIGHTS['low_quality_phash_weight'] * avg_phash)

    # 權重調整：對齊品質影響最終權重（不直接參與相似度計算）
    # 對齊品質高 → 更信任對齊後的分數；對齊品質低 → 降低對齊後分數的權重
    alignment_weight = (DTW_CONFIG['alignment_weight_base'] + 
                       DTW_CONFIG['alignment_weight_scale'] * alignment_quality)
    
    # 覆蓋率高 → 對齊更完整；覆蓋率低 → 對齊不完整，降低權重
    coverage_weight = (DTW_CONFIG['coverage_weight_base'] + 
                      DTW_CONFIG['coverage_weight_scale'] * 
                      min(1.0, coverage / DTW_CONFIG['coverage_threshold']))
    
    # 綜合權重：對齊品質 × 覆蓋率
    final_weight = alignment_weight * coverage_weight
    
    # 應用權重到基礎相似度
    final_sim = base_sim * final_weight

    return final_sim


def _log_similarity_results(lag_frames: int, estimated_lag_seconds: float, coverage: float,
                            alignment_quality: float, deep_aligned: float, avg_phash: float,
                            phash_aligned: float, final_sim: float, aligned_pairs: List[Tuple[str, str]]) -> None:
    """記錄相似度計算結果。

    功能：
        - 記錄對齊資訊和延遲統計
        - 記錄各階段相似度分數
        - 記錄最終結果和配對數量

    Args:
        lag_frames (int): 延遲幀數
        estimated_lag_seconds (float): 估計延遲秒數
        coverage (float): 覆蓋率
        alignment_quality (float): DTW 對齊品質
        deep_aligned (float): 對齊後的深度相似度
        avg_phash (float): 平均 pHash 相似度
        phash_aligned (float): 對齊後的 pHash 相似度
        final_sim (float): 最終相似度
        aligned_pairs (List[Tuple[str, str]]): 對齊的幀對
    """
    logger.info(
        f"[ALIGN] lag={lag_frames} ({estimated_lag_seconds:.2f}s) "
        f"coverage={coverage:.3f} alignment_quality={alignment_quality:.3f}"
    )
    logger.info(
        f"[SCORES] deep_aligned={deep_aligned:.3f} pHash_avg={avg_phash:.3f} "
        f"pHash_aligned={phash_aligned:.3f} base={final_sim:.3f}"
    )
    logger.info(f"[FINAL] final={final_sim:.3f} pairs={len(aligned_pairs)}")


def _execute_phash_stage(sampled_frames1: List[str], sampled_frames2: List[str]) -> Tuple[List[float], float]:
    """執行第一階段：pHash 粗估。

    功能：
        - 使用 pHash 進行快速相似度估算
        - 計算所有幀的相似度分數
        - 提供粗估結果用於後續處理

    Args:
        sampled_frames1 (List[str]): 採樣幀列表 1
        sampled_frames2 (List[str]): 採樣幀列表 2

    Returns:
        Tuple[List[float], float]: (相似度列表, 平均相似度)
    """
    phash_similarities, avg_phash = _compute_phash_similarities(sampled_frames1, sampled_frames2)
    if not phash_similarities:
        return [], 0.0
    return phash_similarities, avg_phash

def _execute_alignment_stage(sampled_frames1: List[str], sampled_frames2: List[str],
                             path: List[Tuple[int, int]], batch_size: int,
                             avg_phash: float) -> Tuple[float, float, List[Tuple[str, str]]]:
    """執行對齊階段：計算對齊後的相似度。

    功能：
        - 從 DTW 路徑中抽取對齊配對
        - 計算對齊後的深度相似度
        - 計算對齊後的 pHash 相似度

    Args:
        sampled_frames1 (List[str]): 採樣幀列表 1
        sampled_frames2 (List[str]): 採樣幀列表 2
        path (List[Tuple[int, int]]): DTW 路徑
        batch_size (int): 批次大小
        avg_phash (float): 平均 pHash 相似度

    Returns:
        Tuple[float, float, List[Tuple[str, str]]]: (深度相似度, pHash 相似度, 對齊幀對)
    """
    aligned_pairs = pairs_from_dtw_path(sampled_frames1, sampled_frames2, path, max_pairs=DTW_CONFIG['max_pairs'])
    deep_aligned, phash_aligned = _compute_aligned_similarities(aligned_pairs, batch_size, avg_phash)
    return deep_aligned, phash_aligned, aligned_pairs


def _estimate_lag_time(lag_frames: int, video_duration: float, frames1: List[str], sample_interval: int) -> float:
    """估計延遲時間。

    功能：
        - 根據延遲幀數估計實際延遲時間
        - 考慮採樣間隔和影片長度
        - 提供時間維度的對齊資訊

    Args:
        lag_frames (int): 延遲幀數
        video_duration (float): 影片長度
        frames1 (List[str]): 第一影片幀列表
        sample_interval (int): 採樣間隔

    Returns:
        float: 估計的延遲秒數
    """
    seconds_per_sample = (video_duration / max(1, len(frames1))) * sample_interval
    return float(lag_frames) * seconds_per_sample


def _create_result_dict(final_sim: float, deep_aligned: float, phash_aligned: float, avg_phash: float,
                        alignment_quality: float, coverage: float, aligned_pairs: List[Tuple[str, str]],
                        sampled_frames1: List[str], phash_threshold: float, lag_frames: int,
                        estimated_lag_seconds: float) -> Dict[str, float]:
    """建立結果字典。

    功能：
        - 將所有計算結果組織成字典格式
        - 包含各階段相似度分數
        - 包含對齊和延遲資訊
        - 便於後續處理和分析

    Args:
        final_sim (float): 最終相似度
        deep_aligned (float): 對齊後的深度相似度
        phash_aligned (float): 對齊後的 pHash 相似度
        avg_phash (float): 平均 pHash 相似度
        alignment_quality (float): DTW 對齊品質
        coverage (float): 覆蓋率
        aligned_pairs (List[Tuple[str, str]]): 對齊的幀對
        sampled_frames1 (List[str]): 採樣幀列表
        phash_threshold (float): pHash 閾值
        lag_frames (int): 延遲幀數
        estimated_lag_seconds (float): 估計延遲秒數

    Returns:
        Dict[str, float]: 結果字典
    """
    return {
        "similarity": final_sim,
        "deep_similarity": deep_aligned,        # 對齊後逐幀深度分數（截尾平均）
        "phash_similarity": phash_aligned,      # 沿 path 的 pHash（截尾平均）
        "phash_avg_global": avg_phash,          # 方便觀察
        "alignment_quality": alignment_quality,
        "matched_ratio": coverage,              # 用 coverage 直觀反映重疊
        "similar_pairs": len(aligned_pairs),
        "total_pairs": len(sampled_frames1),
        "phash_threshold": phash_threshold,
        "aligned_coverage": float(coverage),
        "estimated_lag_frames": int(lag_frames),
        "estimated_lag_seconds": float(estimated_lag_seconds),
    }


def video_similarity(frames1: List[str], frames2: List[str],
                     video_duration: float, batch_size: int = None) -> Dict[str, float]:
    """計算兩段影片的相似度。

    功能：
        - 使用兩階段流程計算影片相似度
        - 第一階段：pHash 快速粗估
        - 第二階段：深度特徵 DTW 對齊
        - 第三階段：對齊後相似度計算
        - 提供詳細的相似度分析結果

    Args:
        frames1 (List[str]): 第一段影片的幀檔案列表
        frames2 (List[str]): 第二段影片的幀檔案列表
        video_duration (float): 影片長度（秒）
        batch_size (int, optional): 批次大小。預設為 64。

    Returns:
        Dict[str, float]: 相似度分析結果字典

    Note:
        - 會根據影片長度動態調整採樣參數
        - 提供多種相似度指標和對齊資訊
        - 支援 GPU 加速計算
    """
    try:
        # 使用配置的批次大小
        if batch_size is None:
            batch_size = BATCH_CONFIG['default_batch_size']
            
        # 動態調整採樣與門檻
        sample_interval, phash_threshold = _calculate_sampling_parameters(video_duration)
        sampled_frames1 = frames1[::sample_interval]
        sampled_frames2 = frames2[::sample_interval]

        logger.info(f"採樣幀數: 視頻1={len(sampled_frames1)}，視頻2={len(sampled_frames2)}")

        # 第一階段：pHash 粗估
        phash_similarities, avg_phash = _execute_phash_stage(sampled_frames1, sampled_frames2)
        if not phash_similarities:
            return {"similarity": 0.0}

        # 第二階段：DTW 對齊（可選的時間平滑：針對異源片源抖動做輕量穩定化）
        alignment_quality, path, coverage, lag_frames = execute_dtw_stage(
            sampled_frames1, sampled_frames2, batch_size,
            enable_temporal_smoothing=TEMPORAL_SMOOTHING_CONFIG['enabled'],
            smoothing_method=TEMPORAL_SMOOTHING_CONFIG['method'],
            ema_alpha=TEMPORAL_SMOOTHING_CONFIG['ema_alpha'],
            majority_window=TEMPORAL_SMOOTHING_CONFIG['majority_window']
        )
        if not path:
            return {"similarity": avg_phash}

        # 第三階段：對齊後相似度計算
        deep_aligned, phash_aligned, aligned_pairs = _execute_alignment_stage(
            sampled_frames1, sampled_frames2, path, batch_size, avg_phash)

        # 估計延遲時間
        estimated_lag_seconds = _estimate_lag_time(lag_frames, video_duration, frames1, sample_interval)

        # 計算最終相似度
        final_sim = _calculate_final_similarity(deep_aligned, phash_aligned, alignment_quality, avg_phash, coverage)

        # 記錄日誌
        _log_similarity_results(lag_frames, estimated_lag_seconds, coverage, alignment_quality,
                                deep_aligned, avg_phash, phash_aligned, final_sim, aligned_pairs)

        # 建立結果字典
        return _create_result_dict(final_sim, deep_aligned, phash_aligned, avg_phash, alignment_quality,
                                   coverage, aligned_pairs, sampled_frames1, phash_threshold,
                                   lag_frames, estimated_lag_seconds)

    except Exception as e:
        logger.error(f"計算視頻相似度時出錯: {e}")
        return {"similarity": 0.0}
