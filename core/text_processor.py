# core/text_processor.py
import gc
import os
import re
import math
import json
import torch
import librosa
import inspect
import threading
import torchaudio
import numpy as np
from math import ceil
from tqdm import tqdm
import soundfile as sf
from torch import Tensor
from collections import Counter
from utils.logger import logger, emit
from utils.gpu_utils import gpu_manager
from faster_whisper import WhisperModel
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from utils.downloader import generate_safe_filename
from utils.audio_cleaner import load_and_clean_audio
from sentence_transformers import SentenceTransformer, util

# ======================== 全域變數與常數 ========================
_whisper_model: Optional[WhisperModel] = None
_whisper_lock = threading.Lock()
_sentence_transformer: Optional[SentenceTransformer] = None
_debug_mode: bool = True
_st_lock = threading.Lock()

# ======================== 模型載入與管理 ========================


def _setup_model_device(model: torch.nn.Module, device: str, dtype: torch.dtype) -> torch.nn.Module:
    """設定模型的裝置和資料類型"""
    if hasattr(model, "to_empty"):
        model.to_empty(device=device, dtype=dtype)
    else:
        model.to(device=device, dtype=dtype)
    return model


def _load_and_process_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    """載入並處理檢查點檔案"""
    state = torch.load(ckpt_path, map_location="cpu")

    # 處理 state_dict 包裝
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    # 處理 DataParallel 的 'module.' 前綴
    if isinstance(state, dict) and state and all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}

    return state


def _load_state_dict_with_assign(model: torch.nn.Module, state: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """使用 assign=True 載入狀態字典（如果可用）"""
    sig = inspect.signature(model.load_state_dict)
    try:
        if "assign" in sig.parameters:
            missing, unexpected = model.load_state_dict(state, assign=True)
        else:
            missing, unexpected = model.load_state_dict(state)
    except TypeError:
        missing, unexpected = model.load_state_dict(state)

    return missing, unexpected


def safe_load_module(model: torch.nn.Module, ckpt_path: str,
                     device: str = None, dtype: torch.dtype = torch.float32) -> torch.nn.Module:
    """
    通用安全載入：
    1) 優先用 to_empty() 在正確 device/dtype 分配參數
    2) load_state_dict(assign=True)（PyTorch 2.4+）避免 meta no-op
    3) 兼容 DataParallel 的 'module.' 前綴
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 分配參數記憶體
    model = _setup_model_device(model, device, dtype)

    # 2) 讀取並處理檢查點
    state = _load_and_process_checkpoint(ckpt_path)

    # 3) 載入狀態字典
    missing, unexpected = _load_state_dict_with_assign(model, state)

    if missing or unexpected:
        print(f"[safe_load_module] missing={missing}, unexpected={unexpected}")

    model.eval()
    return model


def _create_cpu_model() -> SentenceTransformer:
    """在 CPU 上建立 SentenceTransformer 模型"""
    torch.set_grad_enabled(False)
    cpu_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="cpu")

    with torch.inference_mode():
        test_embed: Tensor = cpu_model.encode("warmup_materialize", convert_to_tensor=True)

    # 基本健康檢查
    if getattr(test_embed, "is_meta", False):
        raise RuntimeError("模型權重尚未實體化 (meta tensor)")
    if not torch.isfinite(test_embed).all():
        raise RuntimeError("模型初始化失敗：embedding 含有 NaN 或 Inf")

    return cpu_model


def _move_model_to_target_device(cpu_model: SentenceTransformer, target: str) -> SentenceTransformer:
    """將模型移動到目標裝置"""
    if target == "cpu":
        return cpu_model
    else:
        return cpu_model.to(target)


def _fallback_to_cpu() -> SentenceTransformer:
    """回退到 CPU 載入模型"""
    logger.error("載入 SentenceTransformer 失敗，回退 CPU")

    # 清顯存避免殘留
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # 最終保底：CPU 直載
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="cpu")
    model.eval()
    return model


def get_sentence_transformer() -> SentenceTransformer:
    """
    穩健載入：CPU 先實體化避免 meta → 視需要搬到 CUDA。
    失敗時清顯存並回退 CPU。
    """
    global _sentence_transformer
    if _sentence_transformer is not None:
        return _sentence_transformer

    with _st_lock:
        if _sentence_transformer is not None:
            return _sentence_transformer

        # 目標裝置字串化
        dev_obj = gpu_manager.get_device()
        target = "cuda" if getattr(dev_obj, "type", "cpu") == "cuda" and torch.cuda.is_available() else "cpu"
        logger.info(f"載入模型中，目標設備：{target}")

        try:
            # 1) 先在 CPU 完整實體化，徹底避開 meta tensor
            cpu_model = _create_cpu_model()

            # 2) 如需 CUDA，再搬裝置（此時已無 meta）
            model = _move_model_to_target_device(cpu_model, target)
            model.eval()

            _sentence_transformer = model
            logger.info(f"模型載入成功，裝置：{_sentence_transformer.device}")
            return _sentence_transformer

        except Exception as e:
            logger.error(f"載入 SentenceTransformer 失敗，回退 CPU：{e}")
            _sentence_transformer = _fallback_to_cpu()
            return _sentence_transformer


def get_whisper_model() -> WhisperModel:
    """
    全域只載入一次 WhisperModel。
    回傳前 **不加鎖**，鎖留給呼叫端保護 transcribe 時段即可。
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

# ======================== 字幕提取與處理 ========================


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
    """自 YouTube 連結擷取 <video_id> 或 <video_id>_<index> (若為播放清單)"""
    try:
        if "youtu.be/" in url:
            video_id = url.split("youtu.be/")[-1].split("?")[0]
        elif "youtube.com/watch" in url:
            from urllib.parse import urlparse, parse_qs
            video_id = parse_qs(urlparse(url).query)["v"][0]
        else:
            logger.error(f"不支援的 YouTube URL 格式: {url}")
            return ""

        video_id = video_id.split("&")[0]  # 去除多餘參數
        idx_match = re.search(r"index=(\d+)", url)
        return f"{video_id}_{idx_match.group(1)}" if idx_match else video_id

    except Exception as e:
        logger.error(f"擷取 Video ID 失敗: {e}")
        return ""


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
    """
    從已下載的字幕文件中讀取字幕，並優先選擇指定語言（如有）。
    """
    try:
        safe_filename = generate_safe_filename(video_url)

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

# ======================== 文本正規化與清理 ========================


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

# ======================== 音訊分段處理 ========================


def _create_temp_directory(audio_path: str) -> str:
    """建立暫存目錄"""
    temp_dir = os.path.join(os.path.dirname(
        audio_path), f"temp_segments_{os.path.splitext(os.path.basename(audio_path))[0]}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def _process_silence_segments(y: np.ndarray, sr: int, temp_dir: str, min_segment_duration: int) -> List[str]:
    """使用靜音檢測處理音訊分段"""
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
    """保存單個音訊分段"""
    try:
        start = i * (samples_per_seg - overlap_samples)
        end = min(start + samples_per_seg, total_samples)
        segment = data[start:end]
        if _debug_mode:
            print(f"[DEBUG] segment_{i:03d}: max={np.abs(segment).max():.4f} len={len(segment)}")
        seg_path = os.path.join(temp_dir, f"segment_{i:03d}.wav")
        sf.write(seg_path, segment, sr)
        logger.debug(f"已儲存片段 {i+1}/{num_segments}")
        return seg_path
    except Exception as e:
        logger.error(f"片段 {i} 儲存失敗: {e}")
        return None


def _process_fixed_length_segments(y: np.ndarray, sr: int, segment_duration: int,
                                   overlap: int, temp_dir: str) -> List[str]:
    """處理固定長度分段"""
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
    """使用 torchaudio 作為備援方案處理音訊"""
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
    """嘗試使用 librosa 載入音訊"""
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        return y, sr
    except Exception:
        return None, None


def _handle_audio_loading(audio_path: str, use_silence_detection: bool,
                          min_segment_duration: int, temp_dir: str) -> List[str]:
    """處理音訊載入和分段"""
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
                                  use_silence_detection: bool = True,  # 保留參數以便日後擴充
                                  merge_gap_threshold: int = 1000,
                                  min_segment_duration: int = 3,) -> List[str]:
    """將音訊切成重疊小段，避免 Whisper 處理過長。若 use_silence_detection=True，則以靜音點分段。"""
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
        logger.error(f"切割音訊失敗: {e}")
        return []
    finally:
        if gpu_manager.get_device().type == "cuda":
            gpu_manager.clear_gpu_memory()
        gc.collect()

# ======================== 音訊轉錄流程 ========================


def _load_cached_transcript(transcript_path: str, task_id: Optional[str]) -> Optional[Tuple[str, Optional[str]]]:
    """載入快取的轉錄結果"""
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
    """保存字幕轉錄結果"""
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write("來源：字幕文件\n")
        f.write(f"語言統計: {{'{sub_lang}': 1}}\n\n")
        f.write(sub_txt)

    meta_path = transcript_path.replace(".txt", ".json")
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump({"lang": sub_lang}, meta_file)


def _transcribe_single_segment(seg: str, idx: int) -> Optional[Tuple[str, str]]:
    """轉錄單個音訊分段"""
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
            if _debug_mode:
                print(f"片段 {idx} 質量不佳，略過…")
            return None

        return seg_text, lang
    except Exception as e:
        logger.error(f"片段 {idx} 轉錄錯誤: {e}")
        return None


def _save_whisper_transcript(transcript_path: str, transcripts: List[str], seg_langs: List[str],
                             most_common_lang: Optional[str], track_languages: bool) -> None:
    """保存 Whisper 轉錄結果"""
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
    """處理 Whisper 轉錄流程"""
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
    """處理轉錄結果"""
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
    """嘗試從字幕檔案提取文本"""
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
    """執行 Whisper 轉錄流程"""
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
    """清理暫存檔案"""
    # 移除暫存資料夾（若為空）
    try:
        os.rmdir(os.path.join(os.path.dirname(clean_vocal_path), "temp_segments"))
    except OSError:
        pass


def transcribe_audio(audio_path: str, video_url: Optional[str] = None, output_dir: Optional[str] = None,
                     preferred_lang: Optional[str] = None, use_silence_detection: bool = True,
                     merge_gap_threshold: int = 1000, min_segment_duration: int = 3, track_languages: bool = True,
                     task_id: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """核心：先嘗試使用字幕；若無字幕則切割 → Whisper 轉錄。"""

    # 若已存在轉錄結果直接回傳
    transcript_path = (
        os.path.join(output_dir, os.path.basename(
            audio_path).replace(".wav", "_transcript.txt"))
        if output_dir
        else None
    )

    cached_result = _load_cached_transcript(transcript_path, task_id)
    if cached_result:
        return cached_result

    # 嘗試使用字幕 (.vtt)
    subtitle_result = _try_subtitle_extraction(video_url, output_dir, preferred_lang, transcript_path, task_id)
    if subtitle_result:
        return subtitle_result

    # Whisper 轉錄
    transcripts, seg_langs = _perform_whisper_transcription(
        audio_path, use_silence_detection, merge_gap_threshold, min_segment_duration)

    if not transcripts:
        return "", None

    # 處理轉錄結果
    most_common_lang = _process_transcription_results(transcripts, seg_langs, transcript_path, track_languages)

    # 清理暫存檔案
    clean_vocal_path = load_and_clean_audio(audio_path)
    _cleanup_temp_files(clean_vocal_path)

    logger.info(
        f"轉錄完成，共成功 {len(transcripts)} 段，語言分布: {dict(Counter(seg_langs)) if seg_langs else {}}，最多語言: {most_common_lang}")

    if task_id:
        emit("progress", task_id=task_id, phase="extract", percent=45, msg="轉錄完成")

    final_txt = " ".join([t.strip() for t in transcripts if len(t.strip()) >= 5])
    return final_txt, most_common_lang

# ======================== 文本質量檢測與驗證 ========================


def _check_text_length(text: str, min_length: int) -> Tuple[bool, str]:
    """檢查文本長度"""
    if not text.strip():
        return False, "文本為空"
    if len(text) < min_length:
        return False, f"文本過短 ({len(text)})"
    if not re.sub(r"\s+", "", text):  # 全是空白、換行等
        return False, "文本為空"
    return True, ""


def _calculate_dynamic_thresholds(text_length: int) -> Tuple[int, int]:
    """計算動態重複判定閾值"""
    n = 3 if text_length < 100 else 5 if text_length < 500 else 10
    r = 3 if text_length < 100 else 4 if text_length < 500 else 5
    return n, r


def _check_repetition_ratio(matches: List, text_length: int) -> Tuple[bool, str]:
    """檢查重複比例"""
    if not matches:
        return True, ""

    # 計算重複內容佔總文本的比例
    repeat_ratio = sum(len(m.group(0)) for m in matches) / text_length

    # 如果重複內容超過文本的 70%，判定為無意義
    if repeat_ratio > 0.7:
        repeated_examples = [m.group(1) for m in matches[:3]]  # 取前三個重複示例
        return (False, f"存在大量重複內容（佔比 {repeat_ratio:.1%}），例如：{', '.join(repeated_examples)}")

    return True, ""


def _check_repetition_patterns(text: str) -> Tuple[bool, str]:
    """檢查重複模式"""
    # 動態重複判定
    n, r = _calculate_dynamic_thresholds(len(text))
    pattern = rf"(.{{{n},}}?)\1{{{r-1},}}"
    matches = list(re.finditer(pattern, text))

    return _check_repetition_ratio(matches, len(text))


def is_meaningful_text(text: str, min_length: int = 10) -> tuple[bool, str]:
    """粗略判斷文本是否『有意義』(非重複、長度足夠)。"""
    try:
        # 檢查文本長度
        length_check, length_msg = _check_text_length(text, min_length)
        if not length_check:
            return False, length_msg

        # 檢查重複模式
        repetition_check, repetition_msg = _check_repetition_patterns(text)
        if not repetition_check:
            return False, repetition_msg

        return (True, "文本有效")

    except Exception as e:
        logger.error(f"判斷文本意義時出錯: {str(e)}")
        return (False, f"錯誤: {str(e)}")


def is_excessive_repetition(text: str, phrase_threshold: int = 15, length_threshold: float = 0.8):
    """
    檢查轉錄文本是否存在過度重複的三字詞片段。
    如果某句重複出現次數 ≥ phrase_threshold 或佔比 ≥ length_threshold，
    返回 True，否則返回 False。
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
        if _debug_mode:
            print(f"[DEBUG] Most repeated phrase: '{max_phrase}' appears {count} times, ratio: {ratio:.2f}")
        if count >= phrase_threshold or ratio >= length_threshold:
            return True
    return False


def is_hallucination_phrase(text: str, phrases: list[str] = None, threshold: int = 5) -> bool:
    """
    檢查文本是否包含大量幻覺語句（如 Thank you for watching）。
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


def warn_if_language_abnormal(lang: str, allowed: set = {"zh", "en", "ja"}):
    """
    若語言異常則 log 警告。
    """
    if _debug_mode and (lang not in allowed):
        logger.warning(f"語言偵測異常（{lang}），請檢查內容合理性")


def format_segment_transcripts(transcripts: list[str], langs: list[str]) -> str:
    """
    將每段語言與內容格式化為可寫入檔案的字串。
    """
    lines = []
    for idx, (text, lang) in enumerate(zip(transcripts, langs)):
        lines.append(f"\n[{lang}] 段落 {idx+1}:\n{text.strip()}\n")
    return "".join(lines)

# ======================== 文本相似度計算 ========================


def _check_embedding_validity(embedding: torch.Tensor) -> bool:
    """檢查嵌入向量的有效性"""
    if embedding is None:
        return False
    if hasattr(embedding, 'is_meta') and embedding.is_meta:
        return False
    if not torch.isfinite(embedding).all():
        return False
    return True


def compute_text_embedding(text: str) -> Optional[torch.Tensor]:
    """計算文本向量，避免 meta tensor 與 NaN 問題，並可自動重載模型。"""
    global _sentence_transformer

    try:
        model = get_sentence_transformer()

        with torch.no_grad():
            embedding = model.encode(text, convert_to_tensor=True)

            # 檢查是否為 meta tensor
            if not _check_embedding_validity(embedding):
                logger.warning("檢測到 meta tensor，重新載入模型編碼")
                _sentence_transformer = None
                model = get_sentence_transformer()
                embedding = model.encode(text, convert_to_tensor=True)

            # 檢查 tensor 合法性
            if not _check_embedding_validity(embedding):
                logger.error("嵌入向量含 NaN 或 Inf，回傳 None")
                return None

            return embedding

    except Exception as e:
        logger.error(f"計算文本嵌入向量時出錯: {str(e)}")
        return None


def _calculate_length_penalty(text1: str, text2: str) -> float:
    """計算長度懲罰係數"""
    # 使用詞數（避免字元數誤導）計算長度比例
    len1 = len(text1.split())
    len2 = len(text2.split())
    len_ratio = min(len1, len2) / max(len1, len2)

    # 改良版懲罰係數（指數遞減方式，越長差越懲罰）
    penalty = 1.0 - 0.3 * math.exp(-len_ratio * 5)
    return penalty


def _validate_single_text(text: str, text_name: str) -> Tuple[bool, str]:
    """驗證單個文本的有效性"""
    valid, msg = is_meaningful_text(text)
    if not valid:
        return False, f"{text_name}: {msg}"
    return True, ""


def _validate_texts(text1: str, text2: str) -> Tuple[bool, str]:
    """驗證兩個文本的有效性"""
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
    """計算兩個文本的嵌入向量"""
    emb1 = compute_text_embedding(text1)
    emb2 = compute_text_embedding(text2)
    if emb1 is None or emb2 is None:
        logger.error("嵌入計算失敗")
        return None, None
    return emb1, emb2


def _calculate_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """計算餘弦相似度"""
    with torch.no_grad():
        try:
            sim = util.pytorch_cos_sim(emb1, emb2)[0][0].item()
            return sim
        except Exception as e:
            logger.error(f"相似度計算錯誤: {e}")
            raise


def text_similarity(text1: str, text2: str) -> Tuple[float, bool, str]:
    """回傳 (相似度, 是否有效, 描述)。"""
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
