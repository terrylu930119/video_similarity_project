"""
音訊處理模組

此模組提供音訊處理相關功能，包括：
- 音訊特徵提取（PANN、OpenL3、MFCC 等）
- 多模態音訊相似度計算
- 特徵快取與記憶體管理
- 音訊預處理與標準化

主要功能：
- audio_similarity: 音訊相似度計算主函式
- 深度學習特徵提取（PANN、OpenL3）
- 傳統統計特徵提取（MFCC、色度圖等）
- 多種相似度計算方法
"""

from .similarity import audio_similarity
from .extractors.pann_extractor import extract_pann_features, get_pann_model
from .extractors.openl3_extractor import extract_openl3_features, get_openl3_model
from .extractors.statistical_extractor import extract_statistical_features
from .extractors.dl_extractor import extract_dl_features, get_mel_transform
from .utils import (
    normalize_waveform,
    _to_path_hash,
    get_cache_path,
    save_audio_features_to_cache,
    load_audio_features_from_cache,
    load_audio,
    perceptual_score,
    cos_sim,
    dtw_sim,
    chamfer_sim,
    _as_float,
    _ensure_feature_shapes,
    extract_audio
)
from .config import AUDIO_CONFIG, FEATURE_CONFIG, SIMILARITY_WEIGHTS

__all__ = [
    'audio_similarity',
    'extract_pann_features',
    'get_pann_model',
    'extract_openl3_features',
    'get_openl3_model',
    'extract_statistical_features',
    'extract_dl_features',
    'get_mel_transform',
    'normalize_waveform',
    '_to_path_hash',
    'get_cache_path',
    'save_audio_features_to_cache',
    'load_audio_features_from_cache',
    'load_audio',
    'perceptual_score',
    'cos_sim',
    'dtw_sim',
    'chamfer_sim',
    '_as_float',
    '_ensure_feature_shapes',
    'extract_audio',
    'AUDIO_CONFIG',
    'FEATURE_CONFIG',
    'SIMILARITY_WEIGHTS'
]
