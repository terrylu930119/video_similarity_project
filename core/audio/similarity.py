"""
音訊相似度計算

此模組提供音訊相似度計算功能，包括：
- 多模態特徵相似度計算
- 加權融合策略
- 感知校正
- 主流程整合
"""

import traceback
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from utils.logger import logger
from .extractors.pann_extractor import extract_pann_features
from .extractors.openl3_extractor import extract_openl3_features
from .extractors.dl_extractor import extract_dl_features
from .extractors.statistical_extractor import extract_statistical_features
from .utils import (
    load_audio_features_from_cache,
    save_audio_features_to_cache,
    perceptual_score,
    cos_sim,
    dtw_sim,
    chamfer_sim,
    _as_float,
    fit_pca_if_needed,
    apply_pca
)

# =============== 配置參數 ===============
from .config import SIMILARITY_WEIGHTS

THREAD_CONFIG = {'max_workers': 6}


def chunkwise_dtw_sim(chunk1: np.ndarray, chunk2: np.ndarray, n_components: int = 32) -> float:
    """計算 OpenL3 chunkwise 特徵的 DTW 相似度。

    功能：
        - 對 OpenL3 chunkwise 特徵進行 PCA 降維
        - 使用 DTW 比較序列相似度
        - 處理維度不足的情況
        - 處理空集合

    Args:
        chunk1 (np.ndarray): 第一個 chunkwise 特徵
        chunk2 (np.ndarray): 第二個 chunkwise 特徵
        n_components (int, optional): PCA 主成分數。預設為 32。

    Returns:
        float: 相似度分數（0-1）

    Note:
        - 降維後若為 1D 會改用餘弦相似度
        - 空集合時回傳 0.0
    """
    # ──────────────── 第1階段：輸入整形與邊界 ────────────────
    if not isinstance(chunk1, np.ndarray):
        chunk1 = np.asarray(chunk1)
    if not isinstance(chunk2, np.ndarray):
        chunk2 = np.asarray(chunk2)
    if chunk1.ndim == 1:
        chunk1 = chunk1.reshape(-1, 1)
    if chunk2.ndim == 1:
        chunk2 = chunk2.reshape(-1, 1)
    if chunk1.shape[0] < 2 or chunk2.shape[0] < 2:
        return 0.0
    # ──────────────── 第2階段：擬合/取得 PCA 並降維 ────────────────
    combined = np.vstack([chunk1, chunk2])
    fit_pca_if_needed('openl3_chunkwise', combined, n_components=n_components)
    r1 = apply_pca('openl3_chunkwise', chunk1, n_components=n_components)
    r2 = apply_pca('openl3_chunkwise', chunk2, n_components=n_components)
    # ──────────────── 第3階段：退化條件與 DTW 比較 ────────────────
    if r1.ndim == 1 or r2.ndim == 1:
        return cos_sim(r1, r2)
    import librosa
    cost = librosa.sequence.dtw(X=r1.T, Y=r2.T, metric='euclidean')[0]
    dtw_dist = cost[-1, -1]
    return float(1.0 / (1.0 + dtw_dist / len(r1)))


def score_onset(f1: Dict[str, Any], f2: Dict[str, Any]) -> Optional[Tuple[str, float, str]]:
    """計算 onset 強度包絡的相似度。

    功能：
        - 使用 DTW 比較兩個 onset 強度包絡
        - 計算時間序列的相似度
        - 處理缺失欄位的情況

    Args:
        f1 (Dict[str, Any]): 第一個特徵字典
        f2 (Dict[str, Any]): 第二個特徵字典

    Returns:
        Optional[Tuple[str, float, str]]: (特徵名, 相似度, 權重鍵)，缺失時回傳 None
    """
    if 'onset_env' not in f1 or 'onset_env' not in f2:
        return None
    return ('onset_env', _as_float(dtw_sim(f1['onset_env'], f2['onset_env'])), 'onset_env')


def score_stats_block(name: str, f1: Dict[str, Any], f2: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """比較統計特徵區塊的相似度。

    功能：
        - 比較指定名稱的統計特徵區塊
        - 計算 mean 和 std 的相似度
        - 處理缺失區塊和型別不符的情況

    Args:
        name (str): 特徵區塊名稱
        f1 (Dict[str, Any]): 第一個特徵字典
        f2 (Dict[str, Any]): 第二個特徵字典

    Returns:
        List[Tuple[str, float, str]]: 相似度結果列表

    Note:
        - 會計算 mean 和 std 的相似度
        - 缺失區塊時回傳空列表
    """
    res: List[Tuple[str, float, str]] = []
    b1, b2 = f1.get(name), f2.get(name)
    if not isinstance(b1, dict) or not isinstance(b2, dict):
        return res
    for stat in ('mean', 'std'):
        if stat in b1 and stat in b2:
            sim = _as_float(cos_sim(b1[stat], b2[stat]) ** 2)
            res.append((f"{name}_{stat}", sim, f"{name}_{stat}"))
    return res


def score_tempo(f1: Dict[str, Any], f2: Dict[str, Any]) -> Optional[Tuple[str, float, str]]:
    """計算節奏相似度。

    功能：
        - 使用 mean、std、range 綜合衡量節奏相似度
        - 計算各指標的相似度分數
        - 加權平均得到最終分數

    Args:
        f1 (Dict[str, Any]): 第一個特徵字典
        f2 (Dict[str, Any]): 第二個特徵字典

    Returns:
        Optional[Tuple[str, float, str]]: (特徵名, 相似度, 權重鍵)，缺失時回傳 None

    Note:
        - 權重：mean 0.5, std 0.25, range 0.25
    """
    if 'tempo' not in f1 or 'tempo' not in f2:
        return None
    t1, t2 = f1['tempo'], f2['tempo']
    s1 = 1.0 / (1.0 + abs(float(t1['mean']) - float(t2['mean'])) / 30.0)
    s2 = 1.0 / (1.0 + abs(float(t1['std']) - float(t2['std'])) / 15.0)
    s3 = 1.0 / (1.0 + abs(float(t1['range']) - float(t2['range'])) / 30.0)
    return ('tempo', _as_float(0.5 * s1 + 0.25 * s2 + 0.25 * s3), 'tempo')


def score_openl3(f1: Dict[str, Any], f2: Dict[str, Any]) -> Optional[Tuple[str, float, str]]:
    """計算 OpenL3 特徵的相似度。

    功能：
        - 比較 OpenL3 merged 和 chunkwise 特徵
        - 計算各子特徵的相似度
        - 取平均得到最終分數

    Args:
        f1 (Dict[str, Any]): 第一個特徵字典
        f2 (Dict[str, Any]): 第二個特徵字典

    Returns:
        Optional[Tuple[str, float, str]]: (特徵名, 相似度, 權重鍵)，無有效子項時回傳 None

    Note:
        - 會比較 merged 和 chunkwise 特徵
        - 若無有效子項會回傳 None
    """
    if 'openl3_features' not in f1 or 'openl3_features' not in f2:
        return None
    o1, o2 = f1['openl3_features'], f2['openl3_features']
    sims: List[float] = []
    if isinstance(o1, dict) and isinstance(o2, dict):
        if 'merged' in o1 and 'merged' in o2:
            sims.append(_as_float(cos_sim(o1['merged'], o2['merged'])))
        if 'chunkwise' in o1 and 'chunkwise' in o2:
            sims.append(_as_float(chunkwise_dtw_sim(o1['chunkwise'], o2['chunkwise'])))
    else:
        sims.append(_as_float(cos_sim(np.asarray(o1), np.asarray(o2))))
    if not sims:
        return None
    return ('openl3_features', float(np.mean(sims)), 'openl3_features')


def score_deep(name: str, v1: Any, v2: Any) -> Optional[Tuple[str, float, str]]:
    """計算深度特徵的相似度。

    功能：
        - 通用的深度特徵相似度計算
        - 2D 特徵使用 Chamfer 距離
        - 1D 特徵使用餘弦相似度
        - 處理 None 值

    Args:
        name (str): 特徵名稱
        v1 (Any): 第一個特徵值
        v2 (Any): 第二個特徵值

    Returns:
        Optional[Tuple[str, float, str]]: (特徵名, 相似度, 權重鍵)，任一為 None 時回傳 None
    """
    if v1 is None or v2 is None:
        return None
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    if v1.ndim == 2 and v2.ndim == 2:
        sim = _as_float(chamfer_sim(v1, v2, top_k=3))
    else:
        sim = _as_float(cos_sim(v1, v2))
    return (name, sim, name)


def score_pann(f1: Dict[str, Any], f2: Dict[str, Any]) -> Optional[Tuple[str, float, str]]:
    """計算 PANN 特徵的相似度。

    功能：
        - 計算 PANN 特徵的相似度
        - 可拆解為 embedding 和 tag 的加權融合
        - 處理形狀不符的情況

    Args:
        f1 (Dict[str, Any]): 第一個特徵字典
        f2 (Dict[str, Any]): 第二個特徵字典

    Returns:
        Optional[Tuple[str, float, str]]: (特徵名, 相似度, 權重鍵)

    Note:
        - 權重：embedding 0.6, tag 0.4
        - 形狀不符時會退化為通用深度相似度
    """
    if 'pann_features' not in f1 or 'pann_features' not in f2:
        return None
    v1, v2 = np.asarray(f1['pann_features']), np.asarray(f2['pann_features'])
    if v1.ndim == 1 and v1.shape[0] >= 2575 and v2.ndim == 1 and v2.shape[0] >= 2575:
        split = 2048
        emb1, tag1 = v1[:split], v1[split:]
        emb2, tag2 = v2[:split], v2[split:]
        sim = _as_float(0.6 * cos_sim(emb1, emb2) + 0.4 * cos_sim(tag1, tag2))
        return ('pann_features', sim, 'pann_features')
    return score_deep('pann_features', v1, v2)


class ScoreAccumulator:
    """分數累積器。

    功能：
        - 統一累積各模組的分數與權重
        - 計算加權平均分數
        - 記錄詳細的分數資訊

    Attributes:
        detailed (Dict[str, Tuple[float, float]]): 詳細分數記錄
        scores (List[float]): 分數列表
        weights (List[float]): 權重列表
    """

    def __init__(self) -> None:
        self.detailed: Dict[str, Tuple[float, float]] = {}
        self.scores: List[float] = []
        self.weights: List[float] = []

    def push(self, item: Optional[Tuple[str, float, str]]) -> None:
        """寫入一筆分數。

        Args:
            item (Optional[Tuple[str, float, str]]): (特徵名, 分數, 權重鍵)

        Note:
            - None 會被直接忽略
            - 權重優先順序：weight_key > name > 1.0
        """
        if item is None:
            return
        name, score, weight_key = item
        w = float(SIMILARITY_WEIGHTS.get(weight_key, SIMILARITY_WEIGHTS.get(name, 1.0)))
        self.scores.append(score)
        self.weights.append(w)
        self.detailed[name] = (score, w)

    def weighted_average(self) -> float:
        """計算加權平均分數。

        Returns:
            float: 加權平均分數，無分數時回傳 0
        """
        if not self.scores:
            return 0.0
        return float(np.average(np.asarray(self.scores, dtype=np.float64),
                                weights=np.asarray(self.weights, dtype=np.float64)))

    def log(self) -> None:
        """輸出詳細分項資訊。

        Note:
            - 會記錄特徵名稱、分數和權重
        """
        for name, (s, w) in self.detailed.items():
            logger.info(f"  {name:20s} | 相似度: {s:.4f} | 權重: {w}")


def compute_similarity(f1: Dict[str, Any], f2: Dict[str, Any]) -> float:
    """計算音訊特徵的綜合相似度。

    功能：
        - 彙整各計分器的分數
        - 根據權重計算加權平均
        - 記錄詳細的分數資訊
        - 處理無有效分數的情況

    Args:
        f1 (Dict[str, Any]): 第一個特徵字典
        f2 (Dict[str, Any]): 第二個特徵字典

    Returns:
        float: 綜合相似度分數

    Note:
        - 會計算多種特徵的相似度
        - 無有效分數時會回傳 0
    """
    # ──────────────── 第1階段：初始化累加器 ────────────────
    acc = ScoreAccumulator()
    # ──────────────── 第2階段：累積各模組分數 ────────────────
    acc.push(score_onset(f1, f2))
    for k in ('mfcc', 'mfcc_delta', 'chroma'):
        for t in score_stats_block(k, f1, f2):
            acc.push(t)
    acc.push(score_tempo(f1, f2))
    acc.push(score_openl3(f1, f2))
    acc.push(score_pann(f1, f2))
    acc.push(score_deep('dl_features', f1.get('dl_features'), f2.get('dl_features')))
    # ──────────────── 第3階段：加權融合與日誌 ────────────────
    final = acc.weighted_average()
    if final == 0.0:
        logger.error("所有相似度評估皆失敗，無法進行加權")
    acc.log()
    return final


def _gather_deep_features(audio_path: str, use_openl3: bool) -> Dict[str, Any]:
    """收集深度學習特徵。

    功能：
        - 收集各種深度學習特徵
        - 包含 DL、PANN、OpenL3 特徵
        - 可選擇是否使用 OpenL3
        - 處理特徵提取失敗

    Args:
        audio_path (str): 音訊檔案路徑
        use_openl3 (bool): 是否使用 OpenL3 特徵

    Returns:
        Dict[str, Any]: 深度特徵字典

    Note:
        - 失敗的特徵會回傳 None 值
    """
    return {
        'dl_features': extract_dl_features(audio_path),
        'pann_features': extract_pann_features(audio_path),
        'openl3_features': extract_openl3_features(audio_path) if use_openl3 else None
    }


def compute_audio_features(audio_path: str, use_openl3: bool = True) -> Optional[Dict[str, Any]]:
    """計算音訊特徵。

    功能：
        - 整合統計特徵和深度特徵的提取
        - 進行型態校正和快取管理
        - 使用並行處理提升效能
        - 處理特徵提取失敗

    Args:
        audio_path (str): 音訊檔案路徑
        use_openl3 (bool, optional): 是否使用 OpenL3 特徵。預設為 True。

    Returns:
        Optional[Dict[str, Any]]: 特徵字典，失敗時回傳 None

    Note:
        - 會先檢查快取，再並行提取特徵
        - 統計特徵為 None 時會終止處理
        - 部分深度特徵可以缺失
    """
    # ──────────────── 第1階段：檢查快取 ────────────────
    cached = load_audio_features_from_cache(audio_path)
    if cached is not None:
        return cached
    # ──────────────── 第2階段：並行提取統計與深度特徵 ────────────────
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_stat = pool.submit(extract_statistical_features, audio_path)
        f_deep = pool.submit(_gather_deep_features, audio_path, use_openl3)
        stat_feat = f_stat.result()
        deep_feat = f_deep.result()
    # ──────────────── 第3階段：錯誤與缺失處理 ────────────────
    if stat_feat is None:
        logger.warning(f"Statistical 特徵提取失敗: {audio_path}")
        return None
    for k in ('dl_features', 'pann_features', 'openl3_features'):
        if deep_feat.get(k) is None:
            logger.warning(f"Deep 特徵缺失：{k} 在 {audio_path}")
    # ──────────────── 第4階段：合併、校正與快取 ────────────────
    features: Dict[str, Any] = {**stat_feat, **{k: v for k, v in deep_feat.items() if v is not None}}
    from .utils import _ensure_feature_shapes
    features = _ensure_feature_shapes(features)
    save_audio_features_to_cache(audio_path, features)
    return features


def audio_similarity(path1: str, path2: str) -> float:
    """計算兩段音訊的相似度。

    功能：
        - 音訊相似度計算的主流程
        - 提取特徵 → 加權計算 → 感知校正
        - 記錄記憶體使用情況
        - 處理特徵無效的情況

    Args:
        path1 (str): 第一段音訊檔案路徑
        path2 (str): 第二段音訊檔案路徑

    Returns:
        float: 音訊相似度分數（0-1）

    Note:
        - 會記錄開始和完成時的記憶體使用量
        - 任一側特徵無效時會回傳 0
    """
    try:
        # ──────────────── 第1階段：提取特徵並記錄內存 ────────────────
        from .utils import log_memory
        log_memory("開始")
        f1 = compute_audio_features(path1)
        f2 = compute_audio_features(path2)
        log_memory("完成")
        # ──────────────── 第2階段：型別檢查 ────────────────
        if not isinstance(f1, dict) or not isinstance(f2, dict):
            logger.error(f"音頻特徵型別錯誤: f1={type(f1)}, f2={type(f2)}")
            return 0.0
        # ──────────────── 第3階段：融合並感知校正 ────────────────
        raw = compute_similarity(f1, f2)
        adj = perceptual_score(raw)
        logger.info(f"音頻原始相似度: {raw:.4f} -> 感知相似度: {adj:.4f}")
        return adj
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"計算音頻相似度時出錯: {e}\n{tb}")
        return 0.0
