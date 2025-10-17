"""
文本處理工具函式

此模組提供文本處理相關的工具函式，包括：
- 文本標準化與清理
- 文本品質評估
- 語言檢測與警告
- 文本格式化
"""

import re
from typing import List, Tuple, Set
from utils.logger import logger

# ======================== 全域變數與常數 ========================


def normalize_text_for_embedding(text: str) -> str:
    """
    將輸入文本標準化為語意更穩定的形式，統一比對基準。
    - 移除字幕標記 (WEBVTT / Kind / Language)
    - 移除多餘空行
    - 合併句子（避免格式差導致語意偏移）
    """
    lines = text.splitlines()
    clean_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 跳過完全沒意義的標頭與時間軸
        if line.startswith("WEBVTT") or re.match(r"^\d{2}:\d{2}:\d{2}", line) or line.startswith("來源："):
            continue

        # 若含有 Kind: / Language:，只去掉這些 prefix 的部分
        if "Kind:" in line or "Language:" in line:
            # 嘗試移除 Kind 與 Language 資訊
            line = re.sub(r"Kind:[^\s]+", "", line)
            line = re.sub(r"Language:[^\s]+", "", line)
            line = line.strip()
            if not line:
                continue  # 被清到空也跳過

        clean_lines.append(line)

    return " ".join(clean_lines)


def _check_text_length(text: str, min_length: int) -> Tuple[bool, str]:
    """檢查文本長度。

    功能：
        - 檢查文本是否為空
        - 檢查文本長度是否足夠
        - 檢查文本是否只包含空白字符

    Args:
        text (str): 要檢查的文本
        min_length (int): 最小長度要求

    Returns:
        Tuple[bool, str]: (是否通過檢查, 檢查結果)
    """
    if not text.strip():
        return False, "文本為空"
    if len(text) < min_length:
        return False, f"文本過短 ({len(text)})"
    if not re.sub(r"\s+", "", text):  # 全是空白、換行等
        return False, "文本為空"
    return True, ""


def _calculate_dynamic_thresholds(text_length: int) -> Tuple[int, int]:
    """計算動態重複判定閾值。

    功能：
        - 根據文本長度動態調整重複檢測的閾值
        - 短文本使用較小的閾值，長文本使用較大的閾值

    Args:
        text_length (int): 文本長度

    Returns:
        Tuple[int, int]: (模式長度, 重複次數)
    """
    n = 3 if text_length < 100 else 5 if text_length < 500 else 10
    r = 3 if text_length < 100 else 4 if text_length < 500 else 5
    return n, r


def _check_repetition_ratio(matches: List, text_length: int) -> Tuple[bool, str]:
    """檢查重複比例。

    功能：
        - 計算重複內容佔總文本的比例
        - 如果重複內容超過70%，判定為無意義

    Args:
        matches (List): 重複模式匹配結果
        text_length (int): 文本總長度

    Returns:
        Tuple[bool, str]: (是否通過檢查, 檢查結果)
    """
    if not matches:
        logger.debug("沒有找到重複模式")
        return True, ""

    # 計算重複內容佔總文本的比例
    repeat_ratio = sum(len(m.group(0)) for m in matches) / text_length
    logger.debug(f"重複比例檢查: 重複內容長度={sum(len(m.group(0)) for m in matches)}, 總長度={text_length}, 比例={repeat_ratio:.3f}")

    # 如果重複內容超過文本的 70%，判定為無意義
    if repeat_ratio > 0.7:
        repeated_examples = [m.group(1) for m in matches[:3]]  # 取前三個重複示例
        logger.debug(f"重複內容過多: {repeated_examples}")
        return False, f"存在大量重複內容（佔比 {repeat_ratio:.1%}），例如：{', '.join(repeated_examples)}"

    logger.debug("重複比例檢查通過")
    return True, ""


def _check_repetition_patterns(text: str) -> Tuple[bool, str]:
    """檢查重複模式。

    功能：
        - 使用正則表達式找出重複模式
        - 動態調整重複檢測的閾值

    Args:
        text (str): 要檢查的文本

    Returns:
        Tuple[bool, str]: (是否通過檢查, 檢查結果)
    """
    # 動態重複判定
    n, r = _calculate_dynamic_thresholds(len(text))
    pattern = rf"(.{{{n},}}?)\1{{{r-1},}}"
    matches = list(re.finditer(pattern, text))
    
    logger.debug(f"重複模式檢查: 文本長度={len(text)}, 閾值n={n}, r={r}, 匹配數={len(matches)}")
    if matches:
        logger.debug(f"找到重複模式: {[m.group(1) for m in matches[:3]]}")

    return _check_repetition_ratio(matches, len(text))


def is_meaningful_text(text: str, min_length: int = 10) -> Tuple[bool, str]:
    """判斷文本是否具有意義。

    功能：
        - 檢查文本長度是否足夠
        - 檢測是否為重複內容
        - 評估文本的整體品質
        - 提供詳細的評估結果

    Args:
        text (str): 要評估的文本
        min_length (int, optional): 最小長度要求。預設為 10。

    Returns:
        Tuple[bool, str]: (是否有意義, 評估結果)

    Note:
        - 會檢查長度、重複性等多個指標
        - 提供詳細的評估資訊
    """
    if not text or not text.strip():
        return False, "文本為空"
    
    # 檢查長度
    length_ok, length_info = _check_text_length(text, min_length)
    if not length_ok:
        return False, length_info
    
    # 檢查重複模式
    repetition_ok, repetition_info = _check_repetition_patterns(text)
    if not repetition_ok:
        return False, repetition_info
    
    return True, f"文本品質良好：{length_info}，{repetition_info}"


def is_excessive_repetition(text: str, phrase_threshold: int = 15, length_threshold: float = 0.8) -> bool:
    """檢查轉錄文本是否存在過度重複的三字詞片段。

    功能：
        - 檢測文本中的重複三字詞片段
        - 根據重複次數和佔比判斷是否過度重複
        - 用於過濾低品質的轉錄結果

    Args:
        text (str): 要檢查的文本
        phrase_threshold (int, optional): 重複次數閾值。預設為 15。
        length_threshold (float, optional): 重複佔比閾值。預設為 0.8。

    Returns:
        bool: 是否存在過度重複，True 表示存在過度重複
    """
    words = text.split()
    total_len = len(words)
    if total_len < 6:
        return False  # 太短不判定

    phrase_counts = {}
    for i in range(total_len - 2):
        phrase = " ".join(words[i:i + 3])
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

    max_phrase = max(phrase_counts, key=phrase_counts.get, default=None)
    if max_phrase:
        count = phrase_counts[max_phrase]
        ratio = (count * 3) / total_len  # 該短語所佔比例
        # Debug 輸出
        logger.debug(f"Most repeated phrase: '{max_phrase}' appears {count} times, ratio: {ratio:.2f}")
        if count >= phrase_threshold or ratio >= length_threshold:
            return True
    return False


def is_hallucination_phrase(text: str, phrases: List[str] = None, threshold: int = 5) -> bool:
    """檢查文本是否包含大量幻覺語句。

    功能：
        - 檢測文本中是否包含常見的幻覺語句
        - 用於過濾轉錄模型產生的無意義內容
        - 支援自定義幻覺語句列表

    Args:
        text (str): 要檢查的文本
        phrases (List[str], optional): 幻覺語句列表。預設為常見的結尾語句。
        threshold (int, optional): 出現次數閾值。預設為 5。

    Returns:
        bool: 是否包含大量幻覺語句，True 表示包含
    """
    if phrases is None:
        phrases = [
            "Thank you for watching",
            "Thank for watching",
        ]
    for phrase in phrases:
        if text.count(phrase) >= threshold:
            return True
    return False


def warn_if_language_abnormal(lang: str, allowed: Set[str] = None) -> None:
    """檢查語言是否異常並記錄警告。

    功能：
        - 檢查檢測到的語言是否在允許的語言列表中
        - 若不在允許列表中則記錄警告日誌
        - 用於監控轉錄品質

    Args:
        lang (str): 檢測到的語言代碼
        allowed (Set[str], optional): 允許的語言代碼集合。預設為 {"zh", "en", "ja"}。
    """
    if allowed is None:
        allowed = {"zh", "en", "ja"}
    
    if lang not in allowed:
        logger.warning(f"語言偵測異常（{lang}），請檢查內容合理性")


def format_segment_transcripts(transcripts: List[str], langs: List[str]) -> str:
    """格式化片段轉錄結果。

    功能：
        - 將多個片段的轉錄結果合併
        - 添加語言標記
        - 格式化輸出文本

    Args:
        transcripts (List[str]): 轉錄結果列表
        langs (List[str]): 語言列表

    Returns:
        str: 格式化後的文本

    Note:
        - 會為每個片段添加語言標記
        - 會合併所有轉錄結果
    """
    if not transcripts:
        return ""
    
    formatted = []
    for i, (transcript, lang) in enumerate(zip(transcripts, langs)):
        if transcript.strip():
            formatted.append(f"[{lang}] {transcript.strip()}")
    
    return "\n".join(formatted)
