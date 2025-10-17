"""
影像處理模組

此模組提供影像處理相關功能，包括：
- 影像特徵提取（感知哈希、深度特徵）
- 影片幀相似度計算
- 動態時間規整（DTW）對齊
- 多模態特徵融合

主要功能：
- video_similarity: 影片相似度計算主函式
- 感知哈希（pHash）特徵提取
- MobileNetV3 深度特徵提取
- DTW 時間序列對齊
"""

from .similarity import video_similarity
from .extractors.phash_extractor import compute_phash
from .extractors.deep_extractor import compute_batch_embeddings, get_image_model
from .utils import (
    _normalize_rows,
    _autocrop_letterbox,
    _scan_vertical_edges,
    _scan_horizontal_edges,
    _scan_edge_axis,
    _detect_subtitle_band
)

__all__ = [
    'video_similarity',
    'compute_phash',
    'compute_batch_embeddings',
    'get_image_model',
    '_normalize_rows',
    '_autocrop_letterbox',
    '_scan_vertical_edges',
    '_scan_horizontal_edges',
    '_scan_edge_axis',
    '_detect_subtitle_band'
]
