"""
文本特徵提取器模組

此模組提供文本特徵提取相關功能，包括：
- Whisper 音訊轉錄
- 文本語義嵌入提取
- 多語言文本處理
- 文本品質評估
"""

from .whisper_extractor import (
    transcribe_audio,
    get_whisper_model,
    extract_subtitles,
    get_subtitle_language,
    get_preferred_subtitle,
    extract_video_id_from_url,
    split_audio_for_transcription
)

from .embedding_extractor import (
    compute_text_embedding,
    compute_text_embeddings_batch,
    compute_text_similarities_batch,
    get_sentence_transformer
)

__all__ = [
    'transcribe_audio',
    'get_whisper_model',
    'extract_subtitles',
    'get_subtitle_language',
    'get_preferred_subtitle',
    'extract_video_id_from_url',
    'split_audio_for_transcription',
    'compute_text_embedding',
    'compute_text_embeddings_batch',
    'compute_text_similarities_batch',
    'get_sentence_transformer'
]
