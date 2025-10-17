"""
文本處理模組

此模組提供文本處理相關功能，包括：
- 音訊轉錄（使用 Whisper 模型）
- 文本語義相似度計算（使用 Sentence Transformers）
- 文本品質評估與有效性判斷
- 多語言支援與模型管理

主要功能：
- text_similarity: 文本相似度計算主函式
- 音訊轉錄與文本清理
- 語義嵌入提取與相似度計算
- 模型載入與記憶體管理
"""

from .similarity import text_similarity
from .extractors.whisper_extractor import (
    transcribe_audio,
    get_whisper_model,
    extract_subtitles,
    get_subtitle_language,
    get_preferred_subtitle,
    extract_video_id_from_url
)
from .extractors.embedding_extractor import (
    compute_text_embedding,
    get_sentence_transformer
)
from .utils import (
    normalize_text_for_embedding,
    is_meaningful_text,
    is_excessive_repetition,
    is_hallucination_phrase,
    warn_if_language_abnormal,
    format_segment_transcripts
)

__all__ = [
    'text_similarity',
    'transcribe_audio',
    'get_whisper_model',
    'extract_subtitles',
    'get_subtitle_language',
    'get_preferred_subtitle',
    'extract_video_id_from_url',
    'compute_text_embedding',
    'get_sentence_transformer',
    'normalize_text_for_embedding',
    'is_meaningful_text',
    'is_excessive_repetition',
    'is_hallucination_phrase',
    'warn_if_language_abnormal',
    'format_segment_transcripts'
]
