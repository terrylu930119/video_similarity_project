"""
統計特徵提取器

此模組提供統計型音訊特徵提取功能，包括：
- MFCC、Chroma、Mel 等統計特徵
- 節奏和 Onset 特徵
- 特徵組合與標準化
"""

import numpy as np
import librosa
from typing import List, Dict, Optional, Any
from utils.logger import logger
from librosa.feature.rhythm import tempo

# =============== 配置參數 ===============
from ..config import AUDIO_CONFIG, FEATURE_CONFIG


def _stats_matrix(x: np.ndarray) -> Dict[str, np.ndarray]:
    """計算矩陣在軸 1 的基本統計量。

    功能：
        - 計算矩陣在軸 1 方向的統計量
        - 包含均值、標準差、最大值、最小值、中位數
        - 轉換為 float32 類型

    Args:
        x (np.ndarray): 輸入矩陣

    Returns:
        Dict[str, np.ndarray]: 統計量字典
    """
    return {
        'mean': np.mean(x, axis=1).astype(np.float32),
        'std': np.std(x, axis=1).astype(np.float32),
        'max': np.max(x, axis=1).astype(np.float32),
        'min': np.min(x, axis=1).astype(np.float32),
        'median': np.median(x, axis=1).astype(np.float32),
    }


def _extract_segment_stats(seg: np.ndarray, sr: int) -> Optional[Dict[str, Any]]:
    """提取單一片段的統計特徵。

    功能：
        - 計算音訊片段的各種統計特徵
        - 包含 MFCC、Delta、Chroma、Mel 等
        - 計算 Onset 強度和節奏資訊
        - 處理過短片段

    Args:
        seg (np.ndarray): 音訊片段
        sr (int): 取樣率

    Returns:
        Optional[Dict[str, Any]]: 特徵字典，片段過短時回傳 None

    Note:
        - 片段小於 1 秒時會回傳 None
        - 會計算多種音訊特徵
    """
    if len(seg) < sr:
        return None
    mel = librosa.feature.melspectrogram(y=seg, sr=sr, **FEATURE_CONFIG['mel'])
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, **FEATURE_CONFIG['mfcc'])
    delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=seg, sr=sr, **FEATURE_CONFIG['chroma'])
    onset = librosa.onset.onset_strength(y=seg, sr=sr)
    tempos = tempo(onset_envelope=onset, sr=sr, aggregate=None)
    return {
        'mfcc': _stats_matrix(mfcc),
        'mfcc_delta': _stats_matrix(delta),
        'chroma': _stats_matrix(chroma),
        'mel': _stats_matrix(mel),
        'onset_env': onset.astype(np.float32),
        'tempo': {
            'mean': float(np.mean(tempos)) if len(tempos) else 0.0,
            'std': float(np.std(tempos)) if len(tempos) else 0.0,
            'range': float(np.max(tempos) - np.min(tempos)) if len(tempos) else 0.0
        }
    }


def _merge_onset(out: Dict[str, Any], feats: List[Dict[str, Any]]) -> None:
    """合併所有片段的 onset 強度包絡。

    功能：
        - 將所有片段的 onset_env 合併為單一序列
        - 保持時間順序
        - 處理沒有 onset_env 的情況

    Args:
        out (Dict[str, Any]): 輸出字典
        feats (List[Dict[str, Any]]): 片段特徵列表

    Note:
        - 若沒有 onset_env 則不會產生該欄位
    """
    arrs = [f['onset_env'] for f in feats if 'onset_env' in f]
    if arrs:
        out['onset_env'] = np.concatenate(arrs).astype(np.float32)


def _merge_tempo(out: Dict[str, Any], feats: List[Dict[str, Any]]) -> None:
    """合併所有片段的節奏資訊。

    功能：
        - 聚合所有片段的 tempo 統計量
        - 計算 mean、std、range 的平均值
        - 處理沒有 tempo 的情況

    Args:
        out (Dict[str, Any]): 輸出字典
        feats (List[Dict[str, Any]]): 片段特徵列表

    Note:
        - 若片段沒有 tempo 則不會產生該欄位
    """
    if 'tempo' in feats[0] and isinstance(feats[0]['tempo'], dict):
        keys = feats[0]['tempo'].keys()
        out['tempo'] = {k: float(np.mean([f['tempo'][k] for f in feats])) for k in keys}


def _merge_stats_blocks(out: Dict[str, Any], feats: List[Dict[str, Any]], names: tuple) -> None:
    """合併指定名稱的統計特徵區塊。

    功能：
        - 合併指定名稱的統計特徵區塊
        - 對每個統計量進行逐段平均
        - 確保特徵格式的一致性

    Args:
        out (Dict[str, Any]): 輸出字典
        feats (List[Dict[str, Any]]): 片段特徵列表
        names (tuple): 要合併的特徵名稱
    """
    for name in names:
        if name in feats[0]:
            out[name] = {stat: np.mean([f[name][stat] for f in feats], axis=0).astype(np.float32)
                         for stat in feats[0][name]}


def combine_features(features: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """將逐段特徵彙整為整體描述。

    功能：
        - 將多個片段的特徵彙整為單一樣本
        - 分別處理 onset、tempo 和統計特徵
        - 降低處理複雜度
        - 處理空輸入

    Args:
        features (List[Dict[str, Any]]): 片段特徵列表

    Returns:
        Optional[Dict[str, Any]]: 彙整後的特徵字典，空輸入時回傳 None

    Note:
        - 會分別處理 onset、tempo 和統計特徵
        - 降低處理複雜度
    """
    # ──────────────── 第1階段：空集合處理 ────────────────
    if not features:
        return None
    out: Dict[str, Any] = {}
    # ──────────────── 第2階段：合併 onset 與 tempo ────────────────
    _merge_onset(out, features)
    _merge_tempo(out, features)
    # ──────────────── 第3階段：合併統計特徵 ────────────────
    _merge_stats_blocks(out, features, ('mfcc', 'mfcc_delta', 'chroma', 'mel'))
    return out


def extract_statistical_features(audio_path: str, chunk_sec: float = 10.0) -> Optional[Dict[str, Any]]:
    """提取統計型特徵。

    功能：
        - 使用 librosa 計算統計型特徵
        - 將音訊分段處理
        - 彙整所有片段的特徵
        - 處理載入和計算錯誤

    Args:
        audio_path (str): 音訊檔案路徑
        chunk_sec (float, optional): 片段長度（秒）。預設為 10.0。

    Returns:
        Optional[Dict[str, Any]]: 統計特徵字典，失敗時回傳 None

    Note:
        - 會逐段計算特徵然後彙整
        - 包含 MFCC、Chroma、Onset 等特徵
    """
    try:
        # ──────────────── 第1階段：載入音訊 ────────────────
        y, sr = librosa.load(audio_path, sr=AUDIO_CONFIG['sample_rate'])
        chunk_size = int(chunk_sec * sr)
        features: List[Dict[str, Any]] = []
        # ──────────────── 第2階段：逐段特徵計算 ────────────────
        for i in range(0, len(y), chunk_size):
            seg = y[i:i + chunk_size]
            item = _extract_segment_stats(seg, sr)
            if item is not None:
                features.append(item)
        # ──────────────── 第3階段：彙整輸出 ────────────────
        return combine_features(features)
    except Exception as e:
        logger.error(f"Stat feature error: {e}")
        return None
