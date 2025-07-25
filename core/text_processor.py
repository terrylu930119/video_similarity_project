import gc
import os
import re
import torch
import librosa
import threading 
import torchaudio
import numpy as np
from math import ceil
from tqdm import tqdm
import soundfile as sf
from utils.logger import logger
from utils.gpu_utils import gpu_manager
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor
from utils.downloader import generate_safe_filename
from utils.audio_cleaner import load_and_clean_audio
from sentence_transformers import SentenceTransformer, util

# ======================== 全域模型 ========================
_whisper_model: WhisperModel | None = None      
_whisper_lock  = threading.Lock()
_sentence_transformer: SentenceTransformer | None = None
_debug_mode: bool = False

# ======================== meta tensor 檢查 ========================
def check_tensor_status(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """檢查 tensor 狀態，避免 meta tensor 問題"""
    try:
        if hasattr(tensor, 'is_meta') and tensor.is_meta:
            logger.error(f"{name} 是 meta tensor，無法進行計算")
            return False
            
        if tensor.numel() == 0:
            logger.warning(f"{name} 是空 tensor")
            return False
            
        # 嘗試訪問數據
        _ = tensor.sum()
        return True
        
    except Exception as e:
        logger.error(f"檢查 {name} 時出錯: {e}")
        return False

def safe_tensor_operation(tensor1: torch.Tensor, tensor2: torch.Tensor, operation_name: str):
    """安全的 tensor 操作"""
    if not check_tensor_status(tensor1, "tensor1"):
        return None
    if not check_tensor_status(tensor2, "tensor2"):
        return None
        
    try:
        return util.pytorch_cos_sim(tensor1, tensor2)
    except Exception as e:
        logger.error(f"{operation_name} 操作失敗: {e}")
        return None
    
# ======================== 模型載入函數 ========================
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
        _whisper_model = WhisperModel("medium", device=device, compute_type=compute_type)
        logger.info("Whisper 模型載入完成！")
    return _whisper_model

def get_sentence_transformer() -> SentenceTransformer:
    global _sentence_transformer
    if _sentence_transformer is not None:
        return _sentence_transformer

    try:
        device = gpu_manager.get_device()
        
        # 先在 CPU 上載入模型
        model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device='cpu')
        
        # 確保模型完全初始化
        _ = model.encode("test initialization", convert_to_tensor=False)
        
        # 然後安全地移動到目標設備
        if device.type == "cuda":
            model = model.to(device)
            
        # 再次測試以確保權重已實體化
        test_embedding = model.encode("test", convert_to_tensor=True)
        if test_embedding.is_meta:
            raise RuntimeError("模型仍在 meta device 上")
            
        model.eval()
        _sentence_transformer = model
        return model
        
    except Exception as e:
        logger.error(f"載入 SentenceTransformer 失敗: {e}")
        # 回退到 CPU
        model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device='cpu')
        model.eval()
        _sentence_transformer = model
        return model

# ======================== URL 與字幕處理 ========================
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

def extract_subtitles(video_url: str, output_dir: str) -> str:
    """
    從已下載的字幕文件中讀取字幕
    
    參數:
        video_url: 影片 URL
        output_dir: 輸出目錄
    
    返回:
        字幕文本，如果沒有字幕則返回空字符串
    """
    try:
        # 使用與下載器相同的檔案命名規則
        safe_filename = generate_safe_filename(video_url)
            
        # 檢查字幕文件
        subtitle_files = [f for f in os.listdir(output_dir) if f.startswith(safe_filename) and f.endswith('.vtt')]
        
        if not subtitle_files:
            logger.info("未找到字幕文件")
            return ""
            
        # 列出所有可用的字幕語言
        available_languages = [get_subtitle_language(f) for f in subtitle_files]
        logger.info(f"可用的字幕語言: {', '.join(available_languages)}")
        
        # 選擇優先語言的字幕
        preferred_subtitle = get_preferred_subtitle(subtitle_files, safe_filename)
        if not preferred_subtitle:
            logger.info("未找到合適的字幕文件")
            return ""
            
        subtitle_path = os.path.join(output_dir, preferred_subtitle)
        subtitle_lang = get_subtitle_language(preferred_subtitle)
        logger.info(f"使用 {subtitle_lang} 字幕")
        
        try:
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if not content.strip():
                logger.warning("字幕內容為空")
                return ""
                
            # 處理 VTT 格式字幕，移除時間戳和其他格式信息
            lines = content.split('\n')
            cleaned_lines = []
            for line in lines:
                # 跳過 WebVTT 頭部信息
                if line.startswith('WEBVTT') or '-->' in line or line.strip().isdigit():
                    continue
                # 保留非空的文本行
                if line.strip():
                    cleaned_lines.append(line.strip())
            
            cleaned_content = ' '.join(cleaned_lines)
            logger.info(f"成功讀取字幕，長度: {len(cleaned_content)} 字符")
            return cleaned_content
            
        except Exception as e:
            logger.error(f"讀取字幕文件時出錯: {str(e)}")
            return ""
            
    except Exception as e:
        logger.error(f"提取字幕時出錯: {str(e)}")
        return ""

# ======================== 後處理與清理邏輯 ========================
def merge_transcripts(transcripts: list) -> str:
    """
    合併多個轉錄文本
    """
    return " ".join(filter(None, transcripts))

# ======================== 音訊分段處理 ========================
def split_audio_for_transcription(audio_path: str, segment_duration: int = 30, overlap: int = 2,
                                  use_silence_detection: bool = True,  # 保留參數以便日後擴充 
                                  merge_gap_threshold: int = 1000,
                                  min_segment_duration: int = 3,) -> list[str]:
    """將音訊切成重疊小段，避免 Whisper 處理過長。若 use_silence_detection=True，則以靜音點分段。"""
    if not os.path.exists(audio_path):
        logger.error(f"音訊不存在: {audio_path}")
        return []

    temp_dir = os.path.join(os.path.dirname(audio_path), f"temp_segments_{os.path.splitext(os.path.basename(audio_path))[0]}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception:
        y, sr = None, None

    segment_paths = []

    try:
        if y is not None and use_silence_detection:
            # ────────────── 靜音分段 ────────────────
            intervals = librosa.effects.split(y, top_db=30)  # top_db 可調整
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
            if not segment_paths:
                logger.warning("靜音分段未產生任何片段，回退固定長度切割")
                use_silence_detection = False  # fallback

        if y is not None and not use_silence_detection:
            #  ──────────────  固定長度分段 ────────────────
            total_samples = len(y)
            samples_per_seg = int(segment_duration * sr)
            overlap_samples = int(overlap * sr)
            num_segments = ceil((total_samples - overlap_samples) / (samples_per_seg - overlap_samples))

            def _save_segment_seg(i: int, data, sr_) -> str | None:
                try:
                    start = i * (samples_per_seg - overlap_samples)
                    end = min(start + samples_per_seg, total_samples)
                    segment = data[start:end]
                    if _debug_mode:
                        print(f"[DEBUG] segment_{i:03d}: max={np.abs(segment).max():.4f} len={len(segment)}")
                    seg_path = os.path.join(temp_dir, f"segment_{i:03d}.wav")
                    sf.write(seg_path, segment, sr_)
                    logger.debug(f"已儲存片段 {i+1}/{num_segments}")
                    return seg_path
                except Exception as e:
                    logger.error(f"片段 {i} 儲存失敗: {e}")
                    return None

            with ThreadPoolExecutor(max_workers=max(1, min(os.cpu_count() - 2, 4))) as pool:
                segs = list(pool.map(lambda i: _save_segment_seg(i, y, sr), range(num_segments)))
            segment_paths.extend([p for p in segs if p])

        elif y is None:
            # ── torchaudio 備援 ────────────────────────────────
            waveform, sr = torchaudio.load(audio_path)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze().numpy()
            total_samples = waveform.shape[0]
            samples_per_seg = segment_duration * sr
            overlap_samples = overlap * sr
            num_segments = ceil((total_samples - overlap_samples) / (samples_per_seg - overlap_samples))

            def _save_segment_seg(i: int, data, sr_) -> str | None:
                try:
                    start = i * (samples_per_seg - overlap_samples)
                    end = min(start + samples_per_seg, total_samples)
                    segment = data[start:end]
                    seg_path = os.path.join(temp_dir, f"segment_{i:03d}.wav")
                    sf.write(seg_path, segment, sr_)
                    logger.debug(f"已儲存片段 {i+1}/{num_segments}")
                    return seg_path
                except Exception as e:
                    logger.error(f"片段 {i} 儲存失敗: {e}")
                    return None

            with ThreadPoolExecutor(max_workers=4) as pool:
                segs = list(pool.map(lambda i: _save_segment_seg(i, waveform, sr), range(num_segments)))
            segment_paths.extend([p for p in segs if p])

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
def transcribe_audio(audio_path: str, video_url: str | None = None, output_dir: str | None = None,
                     use_silence_detection: bool = True, merge_gap_threshold: int = 1000, min_segment_duration: int = 3,
                     track_languages: bool = True, ) -> str:
    """核心：先嘗試使用字幕；若無字幕則切割 → Whisper 轉錄。"""

    # 若已存在轉錄結果直接回傳
    transcript_path = (
        os.path.join(output_dir, os.path.basename(audio_path).replace(".wav", "_transcript.txt"))
        if output_dir
        else None
    )
    if transcript_path and os.path.exists(transcript_path):
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                cached = f.read().strip()
                if cached:
                    logger.info(f"已載入快取轉錄: {transcript_path}")
                    return cached
        except Exception as e:
            logger.warning(f"讀取快取失敗: {e}")

    # 嘗試直接使用字幕 (.vtt)
    if video_url and output_dir:
        sub_txt = extract_subtitles(video_url, output_dir)
        if sub_txt:
            if transcript_path:
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write("來源：字幕文件\n\n" + sub_txt)
            return sub_txt

    # Whisper 轉錄
    logger.info("開始 Whisper 轉錄…")
    if gpu_manager.get_device().type == "cuda":
        gpu_manager.clear_gpu_memory()

    get_whisper_model()

    # 前處理 (去雜訊 / 正規化…)
    clean_vocal_path = load_and_clean_audio(audio_path)

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
        return ""

    transcripts: list[str] = []
    seg_langs: list[str] = []

    for idx, seg in enumerate(tqdm(segments, desc="轉錄進度"), 1):
        try:
            # 同一時間只允許一條執行緒呼叫 GPU 推論
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
            if len(seg_text) < 10 or is_excessive_repetition(seg_text) or is_hallucination_phrase(seg_text):
                if _debug_mode:
                    print(f"片段 {idx} 質量不佳，略過…")
                continue

            transcripts.append(seg_text)
            seg_langs.append(lang)
        except Exception as e:
            logger.error(f"片段 {idx} 轉錄錯誤: {e}")
        finally:
            try:
                os.remove(seg)
            except Exception:
                pass

    final_txt = " ".join(transcripts)

    # 儲存文字
    if transcript_path:
        try:
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write("來源：Faster‑Whisper\n")
                if track_languages:
                    stats = {l: seg_langs.count(l) for l in set(seg_langs)}
                    f.write(f"語言統計: {stats}\n")
                    f.write(format_segment_transcripts(transcripts, seg_langs))
                else:
                    f.write(final_txt)
            logger.info(f"轉錄已儲存: {transcript_path}")
        except Exception as e:
            logger.warning(f"寫入轉錄檔失敗: {e}")

    # 移除暫存資料夾（若為空）
    try:
        os.rmdir(os.path.join(os.path.dirname(clean_vocal_path), "temp_segments"))
    except OSError:
        pass

    logger.info(f"轉錄完成，共成功 {len(transcripts)} 段，語言分布: {dict((l, seg_langs.count(l)) for l in set(seg_langs))}")
    return final_txt

# ======================== 重複判斷與濾除邏輯 ========================
def is_excessive_repetition(text: str, phrase_threshold: int = 20, length_threshold: float = 0.8):
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
        phrase = " ".join(words[i:i+3])
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

def is_meaningful_text(text: str, min_length: int = 10) -> tuple[bool, str]:
    """粗略判斷文本是否『有意義』(非重複、長度足夠)。"""
    try:
        if not text.strip():
            return False, "文本為空"
        if len(text) < min_length:
            return False, f"文本過短 ({len(text)})"

        # 動態重複判定
        n = 3 if len(text) < 100 else 5 if len(text) < 500 else 10
        r = 3 if len(text) < 100 else 4 if len(text) < 500 else 5
        pattern = rf"(.{{{n},}}?)\1{{{r-1},}}"
        matches = list(re.finditer(pattern, text))
        
        if matches:
            # 計算重複內容佔總文本的比例
            repeat_ratio = sum(len(m.group(0)) for m in matches) / len(text)
            
            # 如果重複內容超過文本的 70%，判定為無意義
            if repeat_ratio > 0.7:
                repeated_examples = [m.group(1) for m in matches[:3]]  # 取前三個重複示例
                return (False, f"存在大量重複內容（佔比 {repeat_ratio:.1%}），例如：{', '.join(repeated_examples)}")
        
        return (True, "文本有效")
        
    except Exception as e:
        logger.error(f"判斷文本意義時出錯: {str(e)}")
        return (False, f"錯誤: {str(e)}")

def is_hallucination_phrase(text: str, phrases: list[str] = None, threshold: int = 5) -> bool:
    """
    檢查文本是否包含大量幻覺語句（如 Thank you for watching）。
    """
    if phrases is None:
        phrases = [
            "Thank you for watching",
            "This is the first time I've ever seen",
            "See you in the next video"
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

# ======================== 文本分析與比對 ========================
def compute_text_embedding(text: str) -> torch.Tensor | None:
    """計算文本向量，修復 meta tensor 問題。"""
    try:
        model = get_sentence_transformer()
        device = gpu_manager.get_device()
        
        with torch.no_grad():
            # 直接指定設備進行編碼
            if device.type == "cuda":
                embedding = model.encode(text, convert_to_tensor=True, device=device)
            else:
                embedding = model.encode(text, convert_to_tensor=True)
            
            # 檢查是否為 meta tensor
            if hasattr(embedding, 'is_meta') and embedding.is_meta:
                logger.warning("檢測到 meta tensor，重新編碼到 CPU")
                embedding = model.encode(text, convert_to_tensor=True, device='cpu')
            
            # 確保 tensor 在正確設備上且不是 meta
            if embedding.device != device and not embedding.is_meta:
                try:
                    # 使用 to_empty() 方法而非 to() 方法
                    if hasattr(embedding, 'to_empty'):
                        empty_tensor = torch.empty_like(embedding).to(device)
                        empty_tensor.copy_(embedding)
                        embedding = empty_tensor
                    else:
                        embedding = embedding.to(device)
                except Exception as e:
                    logger.warning(f"無法將嵌入向量移至 {device}，使用原設備: {str(e)}")
            
            return embedding
            
    except Exception as e:
        logger.error(f"計算文本嵌入向量時出錯: {str(e)}")
        return None

def text_similarity(text1: str, text2: str) -> tuple[float, bool, str]:
    """回傳 (相似度, 是否有效, 描述)。"""
    if not text1 or not text2:
        logger.warning("任一文本為空")
        return 0.0, False, "任一文本為空"

    valid1, msg1 = is_meaningful_text(text1)
    valid2, msg2 = is_meaningful_text(text2)
    if not valid1 or not valid2:
        logger.warning(f"無意義文本: {msg1 if not valid1 else ''} {msg2 if not valid2 else ''}")
        return 0.0, False, "; ".join(filter(None, [msg1 if not valid1 else "", msg2 if not valid2 else ""]))

    emb1 = compute_text_embedding(text1)
    emb2 = compute_text_embedding(text2)
    if emb1 is None or emb2 is None:
        logger.error("嵌入計算失敗")
        return 0.0, False, "嵌入計算失敗"

    with torch.no_grad():
        sim = util.pytorch_cos_sim(emb1, emb2)[0][0].item()

    len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
    adj_sim = sim * (0.7 + 0.3 * len_ratio)  # 長度差異調整

    logger.info(f"文本相似度: 原始={sim:.4f}, 調整後={adj_sim:.4f}, 長度比例={len_ratio:.2f}")
    return adj_sim, True, f"length_ratio={len_ratio:.2f}"