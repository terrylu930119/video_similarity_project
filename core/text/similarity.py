"""
文本相似度計算模組

此模組提供文本相似度計算相關功能，包括：
- 文本相似度計算主函式
- 多語言文本處理
- 文本品質評估
- 相似度結果整合
"""

import math
import torch
from typing import Optional, Tuple
from utils.logger import logger
from sentence_transformers import SentenceTransformer, util
from .extractors.embedding_extractor import compute_text_embedding
from .utils import (
    normalize_text_for_embedding,
    is_meaningful_text
)

def _calculate_length_penalty(text1: str, text2: str) -> float:
    """計算長度懲罰係數。

    功能：
        - 根據兩個文本的長度差異計算懲罰係數
        - 使用詞數而非字元數避免誤導
        - 採用指數遞減方式，長度差異越大懲罰越重

    Args:
        text1 (str): 第一個文本
        text2 (str): 第二個文本

    Returns:
        float: 懲罰係數（0-1之間）
    """
    # 使用詞數（避免字元數誤導）計算長度比例
    len1 = len(text1.split())
    len2 = len(text2.split())
    len_ratio = min(len1, len2) / max(len1, len2)

    # 改良版懲罰係數（指數遞減方式，越長差越懲罰）
    penalty = 1.0 - 0.3 * math.exp(-len_ratio * 5)
    return penalty


def _validate_single_text(text: str, text_name: str) -> Tuple[bool, str]:
    """驗證單個文本的有效性。

    功能：
        - 檢查文本是否具有意義
        - 提供詳細的驗證結果

    Args:
        text (str): 要驗證的文本
        text_name (str): 文本名稱（用於錯誤訊息）

    Returns:
        Tuple[bool, str]: (是否有效, 驗證結果)
    """
    valid, msg = is_meaningful_text(text)
    if not valid:
        return False, f"{text_name}: {msg}"
    return True, ""


def _validate_texts(text1: str, text2: str) -> Tuple[bool, str]:
    """驗證兩個文本的有效性。

    功能：
        - 逐一驗證兩個文本的有效性
        - 提供詳細的驗證結果

    Args:
        text1 (str): 第一個文本
        text2 (str): 第二個文本

    Returns:
        Tuple[bool, str]: (是否都有效, 驗證結果)
    """
    # 驗證第一個文本
    valid1, msg1 = _validate_single_text(text1, "文本1")
    if not valid1:
        return False, msg1

    # 驗證第二個文本
    valid2, msg2 = _validate_single_text(text2, "文本2")
    if not valid2:
        return False, msg2

    return True, ""


def _compute_embeddings(text1: str, text2: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """計算兩個文本的嵌入向量。

    功能：
        - 計算兩個文本的語義嵌入向量
        - 處理嵌入計算失敗的情況

    Args:
        text1 (str): 第一個文本
        text2 (str): 第二個文本

    Returns:
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: (嵌入向量1, 嵌入向量2)
    """
    emb1 = compute_text_embedding(text1)
    emb2 = compute_text_embedding(text2)
    if emb1 is None or emb2 is None:
        logger.error("嵌入計算失敗")
        return None, None
    return emb1, emb2


def _calculate_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """計算餘弦相似度。

    功能：
        - 計算兩個嵌入向量的餘弦相似度
        - 使用 PyTorch 的優化實現

    Args:
        emb1 (torch.Tensor): 第一個嵌入向量
        emb2 (torch.Tensor): 第二個嵌入向量

    Returns:
        float: 餘弦相似度分數

    Raises:
        Exception: 當相似度計算失敗時
    """
    with torch.no_grad():
        try:
            sim = util.pytorch_cos_sim(emb1, emb2)[0][0].item()
            return sim
        except Exception as e:
            logger.error(f"相似度計算錯誤: {e}")
            raise


def text_similarity(text1: str, text2: str, model: Optional[SentenceTransformer] = None) -> Tuple[float, bool, str]:
    """計算兩個文本的語義相似度。

    功能：
        - 計算兩個文本的語義相似度
        - 支援多語言文本處理
        - 包含文本品質驗證和長度懲罰

    Args:
        text1 (str): 第一個文本
        text2 (str): 第二個文本
        model (Optional[SentenceTransformer]): 嵌入模型，預設使用快取模型

    Returns:
        Tuple[float, bool, str]: (相似度分數, 是否有效, 描述資訊)

    Note:
        - 會自動載入模型（如果未提供）
        - 會驗證文本品質
        - 會應用長度懲罰機制
    """
    if not text1 or not text2:
        logger.warning("任一文本為空")
        return 0.0, False, "任一文本為空"

    # 對文本做標準化
    text1 = normalize_text_for_embedding(text1)
    text2 = normalize_text_for_embedding(text2)

    # 驗證文本有效性
    is_valid, error_msg = _validate_texts(text1, text2)
    if not is_valid:
        return 0.0, False, error_msg

    # 計算嵌入向量
    emb1, emb2 = _compute_embeddings(text1, text2)
    if emb1 is None or emb2 is None:
        return 0.0, False, "嵌入計算失敗"

    # 計算相似度
    try:
        sim = _calculate_cosine_similarity(emb1, emb2)
    except Exception:
        return 0.0, False, "相似度計算錯誤"

    # 計算長度懲罰
    penalty = _calculate_length_penalty(text1, text2)
    adj_sim = sim * penalty

    return adj_sim, True, f"文本相似度: 原始={sim:.4f}, 懲罰={penalty:.4f}, 調整後={adj_sim:.4f}"
