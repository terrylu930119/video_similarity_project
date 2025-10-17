"""
Whisper 轉錄提取器

此模組提供 Whisper 音訊轉錄功能，包括：
- Whisper 模型載入與管理
- 音訊轉錄與文本清理
- 字幕提取與語言檢測
- 轉錄結果快取機制
"""

import os
import re
import json
import librosa
import threading
import torchaudio
import numpy as np
from math import ceil
from tqdm import tqdm
import soundfile as sf
from collections import Counter
from utils.logger import logger, emit
from utils.gpu_utils import gpu_manager
from faster_whisper import WhisperModel
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from utils.audio_cleaner import load_and_clean_audio
from utils.downloader import generate_safe_filename
from ..utils import (
    is_excessive_repetition,
    is_hallucination_phrase,
    warn_if_language_abnormal,
    format_segment_transcripts
)

# ======================== 全域變數與常數 ========================
_whisper_model: Optional[WhisperModel] = None
_whisper_lock = threading.Lock()


def get_whisper_model() -> WhisperModel:
    """載入並快取 Whisper 模型。

    功能：
        - 載入 Whisper 模型並設定為評估模式
        - 使用 GPU 加速（如果可用）
        - 設定適當的計算精度
        - 提供模型快取機制

    Returns:
        WhisperModel: Whisper 模型實例

    Note:
        - 使用多執行緒鎖確保模型只載入一次
        - 會自動檢測並使用 GPU 加速
        - 使用 float16 精度以節省記憶體
    """
    global _whisper_model
    if _whisper_model is None:
        device = "cuda" if gpu_manager.get_device().type == "cuda" else "cpu"
        compute_type = "int8_float16" if device == "cuda" else "int8"
        logger.info("載入 Faster-Whisper medium 權重…")
        _whisper_model = WhisperModel(
            "medium", device=device, compute_type=compute_type)
        logger.info("Whisper 模型載入完成！")
    return _whisper_model


def get_subtitle_language(filename: str) -> str:
    """從字幕檔檔名中擷取語言代碼。

    檔名格式: <safe_filename>.<lang>.vtt
    """
    try:
        parts = filename.split(".")
        return parts[-2] if len(parts) >= 3 else ""
    except Exception as e:
        logger.error(f"擷取語言代碼失敗: {e}")
        return ""


def get_preferred_subtitle(subtitle_files: list[str], safe_filename: str) -> str:
    """依語言優先序 (繁 → 英) 選出最佳字幕檔名。"""
    language_priority = [
        "zh-Hant", "zh-HK", "zh-TW", "zh", "en", "en-US", "en-GB"
    ]

    # 嘗試先比對完全相符的 safe_filename
    exact_matches = [f for f in subtitle_files if f.startswith(safe_filename)]
    subtitle_files = exact_matches or subtitle_files

    # 依優先序找語言代碼
    for lang in language_priority:
        for f in subtitle_files:
            if lang in f:
                return f

    return subtitle_files[0] if subtitle_files else ""


def extract_video_id_from_url(url: str) -> str:
    """從 URL 提取影片 ID。

    功能：
        - 支援多種影片平台的 URL 格式
        - 提取唯一的影片識別碼
        - 處理各種 URL 參數

    Args:
        url (str): 影片 URL

    Returns:
        str: 影片 ID

    Note:
        - 支援 YouTube、Bilibili 等平台
        - 會處理各種 URL 參數
    """
    # YouTube
    youtube_pattern = r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(youtube_pattern, url)
    if match:
        return f"youtube_{match.group(1)}"
    
    # Bilibili
    bilibili_pattern = r'bilibili\.com/video/([a-zA-Z0-9]+)'
    match = re.search(bilibili_pattern, url)
    if match:
        return f"bilibili_{match.group(1)}"
    
    # 其他平台
    return f"unknown_{hash(url) % 1000000}"


def _find_best_subtitle_file(subtitle_files: List[str], preferred_language: Optional[str] = None) -> Tuple[str, str]:
    """找到最佳的字幕檔案"""
    # 建立語言對應表
    lang_map = {f: get_subtitle_language(f) for f in subtitle_files}
    logger.info(f"可用的字幕語言: {', '.join(lang_map.values())}")

    # 嘗試找到與目標語言一致的字幕
    if preferred_language:
        for file, lang in lang_map.items():
            if (lang.lower() == preferred_language.lower() or
                    lang.lower().startswith(preferred_language.lower() + "-")):
                logger.info(f"優先使用 {lang} 字幕")
                return file, lang
        else:
            logger.info(f"未找到符合語言 {preferred_language} 的字幕，改用預設策略")

    # 預設策略：優先英文，否則第一個
    best_file = next(
        (f for f in lang_map if lang_map[f].lower().startswith("en")), list(
            lang_map.keys())[0])
    return best_file, lang_map[best_file]


def _read_subtitle_content(subtitle_path: str) -> Tuple[str, str]:
    """讀取字幕內容"""
    with open(subtitle_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    if not content:
        logger.warning("字幕內容為空")
        return "", ""

    lang_code = get_subtitle_language(os.path.basename(subtitle_path))
    logger.info(
        f"成功讀取字幕（{lang_code}），長度: {len(content)} 字符")

    return content, lang_code


def extract_subtitles(video_url: str, output_dir: str,
                      preferred_language: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """提取影片字幕。

    功能：
        - 從已下載的字幕文件中讀取字幕
        - 支援多種字幕格式
        - 處理語言偏好選擇

    Args:
        video_url (str): 影片 URL
        output_dir (str): 輸出目錄
        preferred_language (Optional[str]): 偏好語言

    Returns:
        Tuple[str, Optional[str]]: (字幕內容, 語言代碼)

    Note:
        - 會從已下載的字幕文件中讀取
        - 會根據語言偏好選擇最佳字幕
    """
    try:
        # 生成安全檔名
        safe_filename = generate_safe_filename(video_url)
        
        # 尋找字幕檔案
        subtitle_files = [f for f in os.listdir(output_dir)
                          if f.startswith(safe_filename) and f.endswith('.vtt')]
        
        if not subtitle_files:
            logger.info("未找到字幕文件")
            return "", None

        # 找到最佳字幕檔案
        best_file, lang_code = _find_best_subtitle_file(subtitle_files, preferred_language)
        subtitle_path = os.path.join(output_dir, best_file)

        # 讀取字幕內容
        content, _ = _read_subtitle_content(subtitle_path)
        return content, lang_code

    except Exception as e:
        logger.error(f"提取字幕時出錯: {str(e)}")
        return "", None


def _create_temp_directory(audio_path: str) -> str:
    """建立暫存目錄"""
    temp_dir = os.path.join(os.path.dirname(
        audio_path), f"temp_segments_{os.path.splitext(os.path.basename(audio_path))[0]}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def _process_silence_segments(y: np.ndarray, sr: int, temp_dir: str, min_segment_duration: int) -> List[str]:
    """處理靜音分段。

    功能：
        - 檢測音訊中的靜音段落
        - 將音訊分段處理
        - 保存分段檔案

    Args:
        y (np.ndarray): 音訊資料
        sr (int): 取樣率
        temp_dir (str): 臨時目錄
        min_segment_duration (int): 最小分段長度

    Returns:
        List[str]: 分段檔案路徑列表
    """
    intervals = librosa.effects.split(y, top_db=30)  # top_db 可調整
    segment_paths = []
    seg_idx = 0

    for start, end in intervals:
        segment = y[start:end]
        duration = (end - start) / sr
        if duration < min_segment_duration:
            continue
        seg_path = os.path.join(temp_dir, f"segment_{seg_idx:03d}.wav")
        sf.write(seg_path, segment, sr)
        logger.debug(f"[靜音分段] 已儲存片段 {seg_idx+1}，長度: {duration:.2f} 秒")
        segment_paths.append(seg_path)
        seg_idx += 1

    return segment_paths


def _save_segment(i: int, data: np.ndarray, sr: int, samples_per_seg: int,
                  overlap_samples: int, total_samples: int, temp_dir: str, num_segments: int) -> Optional[str]:
    """保存音訊分段。

    功能：
        - 保存單個音訊分段
        - 處理重疊區域
        - 確保分段品質

    Args:
        i (int): 分段索引
        data (np.ndarray): 音訊資料
        sr (int): 取樣率
        samples_per_seg (int): 每分段樣本數
        overlap_samples (int): 重疊樣本數
        total_samples (int): 總樣本數
        temp_dir (str): 臨時目錄
        num_segments (int): 總分段數

    Returns:
        Optional[str]: 分段檔案路徑，失敗時回傳 None
    """
    try:
        start = i * (samples_per_seg - overlap_samples)
        end = min(start + samples_per_seg, total_samples)
        segment = data[start:end]
        logger.debug(f"segment_{i:03d}: max={np.abs(segment).max():.4f} len={len(segment)}")
        seg_path = os.path.join(temp_dir, f"segment_{i:03d}.wav")
        sf.write(seg_path, segment, sr)
        logger.debug(f"已儲存片段 {i+1}/{num_segments}")
        return seg_path
    except Exception as e:
        logger.error(f"片段 {i} 儲存失敗: {e}")
        return None


def _process_fixed_length_segments(y: np.ndarray, sr: int, segment_duration: int,
                                   overlap: int, temp_dir: str) -> List[str]:
    """處理固定長度分段。

    功能：
        - 將音訊分段為固定長度
        - 處理重疊區域
        - 保存分段檔案

    Args:
        y (np.ndarray): 音訊資料
        sr (int): 取樣率
        segment_duration (int): 分段長度（秒）
        overlap (int): 重疊長度（秒）
        temp_dir (str): 臨時目錄

    Returns:
        List[str]: 分段檔案路徑列表
    """
    total_samples = len(y)
    samples_per_seg = int(segment_duration * sr)
    overlap_samples = int(overlap * sr)
    num_segments = ceil(
        (total_samples - overlap_samples) / (samples_per_seg - overlap_samples))

    with ThreadPoolExecutor(max_workers=max(1, min(os.cpu_count() - 2, 4))) as pool:
        segs = list(pool.map(
            lambda i: _save_segment(i, y, sr, samples_per_seg, overlap_samples, total_samples, temp_dir, num_segments),
            range(num_segments)
        ))

    return [p for p in segs if p]


def _process_torchaudio_fallback(audio_path: str, segment_duration: int,
                                 overlap: int, temp_dir: str) -> List[str]:
    """使用 torchaudio 作為備援方案。

    功能：
        - 當 librosa 載入失敗時使用 torchaudio
        - 提供音訊載入的備援方案
        - 確保音訊處理的穩定性

    Args:
        audio_path (str): 音訊檔案路徑
        segment_duration (int): 分段長度（秒）
        overlap (int): 重疊長度（秒）
        temp_dir (str): 臨時目錄

    Returns:
        List[str]: 分段檔案路徑列表
    """
    waveform, sr = torchaudio.load(audio_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze().numpy()

    total_samples = waveform.shape[0]
    samples_per_seg = segment_duration * sr
    overlap_samples = overlap * sr
    num_segments = ceil(
        (total_samples - overlap_samples) / (samples_per_seg - overlap_samples))

    with ThreadPoolExecutor(max_workers=4) as pool:
        segs = list(pool.map(
            lambda i: _save_segment(i, waveform, sr, samples_per_seg, overlap_samples, total_samples, temp_dir, num_segments),
            range(num_segments)
        ))

    return [p for p in segs if p]


def _try_load_audio_with_librosa(audio_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """嘗試使用 librosa 載入音訊。

    功能：
        - 使用 librosa 載入音訊檔案
        - 處理載入錯誤
        - 提供載入結果

    Args:
        audio_path (str): 音訊檔案路徑

    Returns:
        Tuple[Optional[np.ndarray], Optional[int]]: (音訊資料, 取樣率)
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        return y, sr
    except Exception:
        return None, None


def _handle_audio_loading(audio_path: str, use_silence_detection: bool,
                          min_segment_duration: int, temp_dir: str) -> List[str]:
    """處理音訊載入。

    功能：
        - 嘗試多種方法載入音訊
        - 處理載入失敗的情況
        - 提供分段結果

    Args:
        audio_path (str): 音訊檔案路徑
        use_silence_detection (bool): 是否使用靜音檢測
        min_segment_duration (int): 最小分段長度
        temp_dir (str): 臨時目錄

    Returns:
        List[str]: 分段檔案路徑列表
    """
    # 嘗試使用 librosa 載入
    y, sr = _try_load_audio_with_librosa(audio_path)
    segment_paths = []

    if y is not None and use_silence_detection:
        # 靜音分段
        segment_paths = _process_silence_segments(y, sr, temp_dir, min_segment_duration)
        if not segment_paths:
            logger.warning("靜音分段未產生任何片段，回退固定長度切割")
            use_silence_detection = False

    if y is not None and not use_silence_detection:
        # 固定長度分段
        segment_paths = _process_fixed_length_segments(y, sr, 30, 2, temp_dir)
    elif y is None:
        # torchaudio 備援
        segment_paths = _process_torchaudio_fallback(audio_path, 30, 2, temp_dir)

    return segment_paths


def split_audio_for_transcription(audio_path: str, segment_duration: int = 30, overlap: int = 2,
                                  use_silence_detection: bool = True,
                                  merge_gap_threshold: int = 1000,
                                  min_segment_duration: int = 3) -> List[str]:
    """將音訊分段以進行轉錄。

    功能：
        - 將長音訊分段處理
        - 支援靜音檢測和固定長度分段
        - 處理重疊區域
        - 提供分段檔案列表

    Args:
        audio_path (str): 音訊檔案路徑
        segment_duration (int, optional): 分段長度（秒）。預設為 30。
        overlap (int, optional): 重疊長度（秒）。預設為 2。
        use_silence_detection (bool, optional): 是否使用靜音檢測。預設為 True。
        merge_gap_threshold (int, optional): 合併間隙閾值。預設為 1000。
        min_segment_duration (int, optional): 最小分段長度（秒）。預設為 3。

    Returns:
        List[str]: 分段檔案路徑列表

    Note:
        - 會創建臨時目錄存放分段檔案
        - 支援多種分段策略
    """
    if not os.path.exists(audio_path):
        logger.error(f"音訊不存在: {audio_path}")
        return []

    temp_dir = _create_temp_directory(audio_path)

    try:
        # 處理音訊載入和分段
        segment_paths = _handle_audio_loading(audio_path, use_silence_detection, min_segment_duration, temp_dir)

        if not segment_paths:
            logger.error("無有效音訊片段！")
        else:
            logger.info(f"音訊切割完成，共產生 {len(segment_paths)} 個片段")

        return segment_paths
    
    except Exception as e:
        logger.error(f"音訊分段失敗：{e}")
        return []


def _load_cached_transcript(transcript_path: str, task_id: Optional[str]) -> Optional[Tuple[str, Optional[str]]]:
    """載入快取的轉錄結果。

    功能：
        - 檢查是否存在快取的轉錄結果
        - 載入快取的轉錄內容
        - 驗證快取的有效性

    Args:
        transcript_path (str): 轉錄檔案路徑
        task_id (Optional[str]): 任務 ID

    Returns:
        Optional[Tuple[str, Optional[str]]]: (轉錄內容, 語言代碼)，無快取時回傳 None
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return None
    
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            cached = f.read().strip()

        lang = None
        meta_path = transcript_path.replace(".txt", ".json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as meta_file:
                lang_data = json.load(meta_file)
                lang = lang_data.get("lang", None)

        logger.info(f"已載入快取轉錄: {transcript_path}（語言: {lang})")

        # 判斷來源字樣，對前端發 45%「文本就緒（快取）」訊息
        if task_id:
            src = "字幕" if cached.startswith("來源：字幕文件") else "轉錄"
            emit("progress", task_id=task_id, phase="extract", percent=45, msg=f"文本就緒（快取，{src})")

        return cached, lang
    except Exception as e:
        logger.warning(f"讀取快取失敗: {e}")
        return None


def _save_subtitle_transcript(transcript_path: str, sub_txt: str, sub_lang: str) -> None:
    """保存字幕轉錄結果。

    功能：
        - 保存字幕轉錄結果到檔案
        - 包含語言資訊
        - 使用 JSON 格式儲存

    Args:
        transcript_path (str): 轉錄檔案路徑
        sub_txt (str): 字幕內容
        sub_lang (str): 語言代碼
    """
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write("來源：字幕文件\n")
        f.write(f"語言統計: {{'{sub_lang}': 1}}\n\n")
        f.write(sub_txt)

    meta_path = transcript_path.replace(".txt", ".json")
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump({"lang": sub_lang}, meta_file)


def _transcribe_single_segment(seg: str, idx: int) -> Optional[Tuple[str, str]]:
    """轉錄單個音訊分段。

    功能：
        - 使用 Whisper 模型轉錄單個音訊分段
        - 進行語言檢測和質量檢查
        - 過濾低質量轉錄結果

    Args:
        seg (str): 音訊分段檔案路徑
        idx (int): 分段索引

    Returns:
        Optional[Tuple[str, str]]: (轉錄文本, 語言代碼)，質量不佳時回傳 None
    """
    try:
        # 同一時間只允許一條執行緒呼叫 GPU 推論，防止 OOM
        with _whisper_lock:
            seg_result, info = get_whisper_model().transcribe(
                seg,
                beam_size=5,
                temperature=0.0,
                vad_filter=False,
                compression_ratio_threshold=5.0,
                log_prob_threshold=-2.5,
                no_speech_threshold=0.6,
            )

        seg_text = "".join([s.text for s in seg_result]).strip()
        lang = info.language or "unknown"

        # 語言異常警告
        warn_if_language_abnormal(lang)

        # 過短、重複、幻覺則跳過
        if len(seg_text) < 10 or is_excessive_repetition(
                seg_text) or is_hallucination_phrase(seg_text):
            logger.debug(f"片段 {idx} 質量不佳，略過…")
            return None

        return seg_text, lang
    except Exception as e:
        logger.error(f"片段 {idx} 轉錄錯誤: {e}")
        return None


def _save_whisper_transcript(transcript_path: str, transcripts: List[str], seg_langs: List[str],
                             most_common_lang: Optional[str], track_languages: bool) -> None:
    """保存 Whisper 轉錄結果。

    功能：
        - 保存 Whisper 轉錄結果到檔案
        - 包含語言統計資訊
        - 使用 JSON 格式儲存

    Args:
        transcript_path (str): 轉錄檔案路徑
        transcripts (List[str]): 轉錄結果列表
        seg_langs (List[str]): 語言列表
        most_common_lang (Optional[str]): 最常見語言
        track_languages (bool): 是否追蹤語言
    """
    try:
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write("來源：Faster‑Whisper\n")
            if track_languages:
                stats = {l: seg_langs.count(l) for l in set(seg_langs)}
                f.write(f"語言統計: {stats}\n")
                f.write(format_segment_transcripts(transcripts, seg_langs))
            else:
                f.write(" ".join([t.strip() for t in transcripts if len(t.strip()) >= 5]))

        # 儲存語言資訊到 json
        meta_path = transcript_path.replace(".txt", ".json")
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            json.dump({"lang": most_common_lang}, meta_file)

        logger.info(f"轉錄已儲存: {transcript_path}")
    except Exception as e:
        logger.warning(f"寫入轉錄檔失敗: {e}")


def _process_whisper_transcription(clean_vocal_path: str, use_silence_detection: bool,
                                   merge_gap_threshold: int, min_segment_duration: int) -> Tuple[List[str], List[str]]:
    """處理 Whisper 轉錄流程。

    功能：
        - 將音訊切割成多個分段
        - 使用 Whisper 模型轉錄每個分段
        - 收集轉錄結果和語言資訊
        - 清理臨時分段檔案

    Args:
        clean_vocal_path (str): 清理後的音訊檔案路徑
        use_silence_detection (bool): 是否使用靜音檢測分段
        merge_gap_threshold (int): 合併間隙閾值
        min_segment_duration (int): 最小分段長度（秒）

    Returns:
        Tuple[List[str], List[str]]: (轉錄結果列表, 語言列表)
    """
    # 切段 → 小檔
    segments = split_audio_for_transcription(
        clean_vocal_path,
        segment_duration=120,
        overlap=2,
        use_silence_detection=use_silence_detection,
        merge_gap_threshold=merge_gap_threshold,
        min_segment_duration=min_segment_duration,
    )
    if not segments:
        logger.error("音訊切割失敗，無法轉錄")
        return [], []

    transcripts: list[str] = []
    seg_langs: list[str] = []

    # 轉錄所有分段
    for idx, seg in enumerate(tqdm(segments, desc="轉錄進度"), 1):
        result = _transcribe_single_segment(seg, idx)
        if result:
            seg_text, lang = result
            transcripts.append(seg_text)
            seg_langs.append(lang)

        # 清理分段檔案
        try:
            os.remove(seg)
        except Exception:
            pass

    return transcripts, seg_langs


def _process_transcription_results(transcripts: List[str], seg_langs: List[str],
                                   transcript_path: str, track_languages: bool) -> Optional[str]:
    """處理轉錄結果。

    功能：
        - 合併轉錄結果
        - 計算語言統計
        - 保存轉錄結果

    Args:
        transcripts (List[str]): 轉錄結果列表
        seg_langs (List[str]): 語言列表
        transcript_path (str): 轉錄檔案路徑
        track_languages (bool): 是否追蹤語言

    Returns:
        Optional[str]: 合併後的轉錄文本
    """
    final_txt = " ".join([t.strip() for t in transcripts if len(t.strip()) >= 5])

    # 統計語言
    most_common_lang: Optional[str] = None
    lang_counter: Dict[str, int] = {}
    if seg_langs:
        lang_counter = Counter(seg_langs)
        most_common_lang = lang_counter.most_common(1)[0][0]

    # 儲存文字
    if transcript_path:
        _save_whisper_transcript(transcript_path, transcripts, seg_langs, most_common_lang, track_languages)

    return most_common_lang


def _try_subtitle_extraction(video_url: str, output_dir: str, preferred_lang: Optional[str],
                             transcript_path: str, task_id: Optional[str]) -> Optional[Tuple[str, Optional[str]]]:
    """嘗試提取字幕。

    功能：
        - 嘗試從影片 URL 提取字幕
        - 處理字幕提取結果
        - 提供字幕內容和語言

    Args:
        video_url (str): 影片 URL
        output_dir (str): 輸出目錄
        preferred_lang (Optional[str]): 偏好語言
        transcript_path (str): 轉錄檔案路徑
        task_id (Optional[str]): 任務 ID

    Returns:
        Optional[Tuple[str, Optional[str]]]: (字幕內容, 語言代碼)
    """
    if not video_url or not output_dir:
        return None

    sub_txt, sub_lang = extract_subtitles(video_url, output_dir, preferred_lang)
    if sub_txt:
        if transcript_path:
            _save_subtitle_transcript(transcript_path, sub_txt, sub_lang)

        # 告知前端：這次是「找到字幕」→ 文本完成
        if task_id:
            emit("progress", task_id=task_id, phase="extract", percent=45, msg="字幕就緒")

        return sub_txt, sub_lang
    return None


def _perform_whisper_transcription(audio_path: str, use_silence_detection: bool,
                                   merge_gap_threshold: int, min_segment_duration: int) -> Tuple[List[str], List[str]]:
    """執行 Whisper 轉錄。

    功能：
        - 執行完整的 Whisper 轉錄流程
        - 處理音訊分段和轉錄
        - 提供轉錄結果

    Args:
        audio_path (str): 音訊檔案路徑
        use_silence_detection (bool): 是否使用靜音檢測
        merge_gap_threshold (int): 合併間隙閾值
        min_segment_duration (int): 最小分段長度

    Returns:
        Tuple[List[str], List[str]]: (轉錄結果列表, 語言列表)
    """
    logger.info("開始 Whisper 轉錄…")
    if gpu_manager.get_device().type == "cuda":
        gpu_manager.clear_gpu_memory()

    get_whisper_model()

    # 前處理 (去雜訊 / 正規化…)
    clean_vocal_path = load_and_clean_audio(audio_path)

    # 處理 Whisper 轉錄
    transcripts, seg_langs = _process_whisper_transcription(
        clean_vocal_path, use_silence_detection, merge_gap_threshold, min_segment_duration)

    return transcripts, seg_langs


def _cleanup_temp_files(clean_vocal_path: str) -> None:
    """清理臨時檔案。

    功能：
        - 清理轉錄過程中的臨時檔案
        - 釋放磁碟空間
        - 保持環境整潔

    Args:
        clean_vocal_path (str): 清理後的音訊檔案路徑
    """
    # 移除暫存資料夾（若為空）
    try:
        os.rmdir(os.path.join(os.path.dirname(clean_vocal_path), "temp_segments"))
    except OSError:
        pass


def transcribe_audio(audio_path: str, video_url: Optional[str] = None, output_dir: Optional[str] = None,
                     preferred_lang: Optional[str] = None, use_silence_detection: bool = True,
                     merge_gap_threshold: int = 1000, min_segment_duration: int = 3, track_languages: bool = True,
                     task_id: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """轉錄音訊為文本。

    功能：
        - 使用 Whisper 模型轉錄音訊
        - 支援字幕提取作為備援方案
        - 提供多語言支援
        - 包含快取機制

    Args:
        audio_path (str): 音訊檔案路徑
        video_url (Optional[str]): 影片 URL（用於字幕提取）
        output_dir (Optional[str]): 輸出目錄
        preferred_lang (Optional[str]): 偏好語言
        use_silence_detection (bool, optional): 是否使用靜音檢測。預設為 True。
        merge_gap_threshold (int, optional): 合併間隙閾值。預設為 1000。
        min_segment_duration (int, optional): 最小分段長度（秒）。預設為 3。
        track_languages (bool, optional): 是否追蹤語言。預設為 True。
        task_id (Optional[str]): 任務 ID

    Returns:
        Tuple[str, Optional[str]]: (轉錄文本, 語言代碼)

    Note:
        - 會先嘗試提取字幕，失敗時使用 Whisper 轉錄
        - 支援多種分段策略
        - 包含完整的錯誤處理
    """
    try:
        # 設定輸出目錄
        if output_dir is None:
            output_dir = os.path.dirname(audio_path)
        
        # 生成轉錄檔案路徑
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
        
        # 檢查快取
        cached_result = _load_cached_transcript(transcript_path, task_id)
        if cached_result:
            return cached_result
        
        # 嘗試提取字幕
        if video_url:
            subtitle_result = _try_subtitle_extraction(video_url, output_dir, preferred_lang, transcript_path, task_id)
            if subtitle_result:
                return subtitle_result
        
        # 執行 Whisper 轉錄
        transcripts, seg_langs = _perform_whisper_transcription(
            audio_path, use_silence_detection, merge_gap_threshold, min_segment_duration
        )
        
        if not transcripts:
            return "", None
        
        # 處理轉錄結果
        most_common_lang = _process_transcription_results(transcripts, seg_langs, transcript_path, track_languages)
        
        # 合併轉錄結果
        combined_transcript = ' '.join(transcripts)
        
        return combined_transcript, most_common_lang
    
    except Exception as e:
        logger.error(f"音訊轉錄失敗：{e}")
        return "", None
    
    finally:
        # 清理臨時檔案
        try:
            _cleanup_temp_files(audio_path)
        except Exception as e:
            logger.warning(f"清理臨時檔案失敗：{e}")
