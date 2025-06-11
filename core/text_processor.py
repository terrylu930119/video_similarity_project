import gc
import os
import re
import torch
import whisper
import torchaudio
from math import ceil
from itertools import groupby
from collections import Counter
from utils.logger import logger
from utils.gpu_utils import gpu_manager
from core.audio_processor import load_audio
from concurrent.futures import ThreadPoolExecutor
from utils.downloader import generate_safe_filename
from sentence_transformers import SentenceTransformer, util
from utils.audio_cleaner import load_and_clean_audio
from faster_whisper import WhisperModel


# 全域模型變數
_whisper_model = None
_sentence_transformer = None

def get_whisper_model():
    """取得或載入 faster-whisper 模型"""
    global _whisper_model
    if _whisper_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("正在載入 faster-whisper medium 模型...")
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        _whisper_model = WhisperModel("medium", device=device, compute_type=compute_type)
        logger.info("Whisper 模型載入完成")
    return _whisper_model

def get_sentence_transformer():
    """
    獲取全域 SentenceTransformer 模型實例，如果未載入則進行載入
    """
    global _sentence_transformer
    if _sentence_transformer is None:
        logger.info("正在載入 SentenceTransformer 模型...")
        _sentence_transformer = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        # 如果有 GPU，將模型移至 GPU
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            logger.info("將 SentenceTransformer 模型移至 GPU")
            _sentence_transformer = _sentence_transformer.to(torch.device('cuda'))
        logger.info("SentenceTransformer 模型載入完成")
    return _sentence_transformer

def get_subtitle_language(filename: str) -> str:
    """
    從字幕文件名中提取語言代碼
    
    參數:
        filename: 字幕文件名
    
    返回:
        語言代碼，如果無法提取則返回空字符串
    """
    try:
        # 從文件名中提取語言代碼
        # 格式：safe_filename.language.vtt
        parts = filename.split('.')
        if len(parts) >= 3:
            return parts[-2]  # 返回倒數第二個部分（語言代碼）
        return ""
    except Exception as e:
        logger.error(f"從文件名提取語言代碼時出錯: {str(e)}")
        return ""

def get_preferred_subtitle(subtitle_files: list, safe_filename: str) -> str:
    """
    根據語言優先順序選擇最適合的字幕文件
    
    參數:
        subtitle_files: 字幕文件列表
        safe_filename: 安全的檔案名稱
    
    返回:
        選擇的字幕文件名，如果沒有合適的字幕則返回空字符串
    """
    # 語言優先順序
    language_priority = ['zh-Hant', 'zh-HK', 'zh-TW', 'zh', 'en', 'en-US', 'en-GB']
    
    # 首先嘗試找到完全匹配的文件名
    exact_matches = [f for f in subtitle_files if f.startswith(safe_filename)]
    if exact_matches:
        subtitle_files = exact_matches
    
    # 按優先順序檢查語言
    for lang in language_priority:
        for file in subtitle_files:
            if lang in file:
                return file
    
    # 如果沒有找到優先語言，返回第一個可用的字幕
    return subtitle_files[0] if subtitle_files else ""

def extract_video_id_from_url(url: str) -> str:
    """從 YouTube URL 中提取影片 ID 和播放清單索引"""
    try:
        # 提取影片 ID
        if 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[-1].split('?')[0]
        elif 'youtube.com/watch' in url:
            from urllib.parse import parse_qs, urlparse
            parsed_url = urlparse(url)
            video_id = parse_qs(parsed_url.query)['v'][0]
        else:
            logger.error(f"不支援的 YouTube URL 格式: {url}")
            return ""
            
        # 移除額外的參數
        video_id = video_id.split('&')[0]
        
        # 提取播放清單索引
        index_match = re.search(r'index=(\d+)', url)
        playlist_index = index_match.group(1) if index_match else ""
        
        # 組合檔案名稱：videoId_index
        return f"{video_id}_{playlist_index}" if playlist_index else video_id
        
    except Exception as e:
        logger.error(f"從 URL 提取影片 ID 時出錯: {str(e)}")
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

# legacy use: 保留舊版後處理邏輯，預設流程中不再呼叫
def post_process_transcript(text: str, language: str = None) -> str:
    """
    後處理轉錄文本，處理重複和格式化問題
    
    參數:
        text: 原始轉錄文本
        language: 語言代碼（'ja', 'zh', 'en' 等）
    """
    if not text:
        return text

    # 定義語言特定的標點符號和規則
    LANGUAGE_RULES = {
        'ja': {
            'sentence_end': '。！？',
            'comma': '、',
            'particles': 'はがでにとへもを',
            'end_mark': '。',
            'comma_mark': '、'
        },
        'zh': {
            'sentence_end': '。！？',
            'comma': '，、',
            'particles': '的地得了著過',
            'end_mark': '。',
            'comma_mark': '，'
        },
        'en': {
            'sentence_end': '.!?',
            'comma': ',',
            'particles': 'and or but in on at with to',
            'end_mark': '.',
            'comma_mark': ','
        }
    }
    
    # 獲取語言規則，如果沒有特定規則則使用英文規則
    rules = LANGUAGE_RULES.get(language, LANGUAGE_RULES['en'])
    
    def remove_consecutive_duplicates(text):
        """移除連續重複的內容"""
        # 根據語言選擇分割方式
        if language in ['ja', 'zh']:  # 中日文按字符分割
            words = list(text)
        else:  # 其他語言按空格分割
            words = text.split()
        
        # 移除連續重複，但保留有意義的重複（如擬聲詞）
        result = []
        for word, group in groupby(words):
            count = len(list(group))
            # 如果是短詞（1-2字符）且重複次數小於等於3，保留重複
            if (len(word) <= 2 and count <= 3) or count == 1:
                result.extend([word] * count)
            else:
                result.append(word)
        
        # 重新組合文本
        if language in ['ja', 'zh']:
            return ''.join(result)
        return ' '.join(result)
    
    def remove_long_duplicates(text):
        """移除長片段重複"""
        # 對於不同語言使用不同的最小長度
        min_length = 2 if language in ['ja', 'zh'] else 3
        max_length = 20
        
        for length in range(max_length, min_length, -1):
            pattern = f'(.{{{length}}})\\1+'
            text = re.sub(pattern, r'\1', text)
        return text
    
    def add_punctuation(text):
        """添加適當的標點符號"""
        # 在句子結尾添加句號
        end_pattern = f'([^{rules["sentence_end"]}\\s])([^\\w{rules["sentence_end"]}]*)$'
        text = re.sub(end_pattern, f'\\1{rules["end_mark"]}\\2', text)
        
        # 在自然停頓處添加逗號
        if language in ['ja', 'zh']:
            # 在特定助詞前添加逗號
            particle_pattern = f'([^{rules["comma"]}{rules["sentence_end"]}\\s])([{rules["particles"]}])'
            text = re.sub(particle_pattern, f'\\1{rules["comma_mark"]}\\2', text)
        else:
            # 在連接詞前添加逗號
            for particle in rules["particles"].split():
                text = re.sub(f'\\s+{particle}\\s+', f'{rules["comma_mark"]} {particle} ', text)
        
        return text
    
    # 執行處理步驟
    text = remove_consecutive_duplicates(text)
    text = remove_long_duplicates(text)
    text = add_punctuation(text)
    
    # 最後的清理
    # 移除多餘的空格
    if language in ['ja', 'zh']:
        text = re.sub(r'\s+', '', text)
    else:
        text = re.sub(r'\s+', ' ', text)
    
    # 移除重複的標點符號
    text = re.sub(f'[{rules["sentence_end"]}{rules["comma"]}]+', lambda m: m.group(0)[0], text)
    
    return text.strip()

def split_audio_for_transcription(audio_path: str, segment_duration: int = 30, overlap: int = 2, use_silence_detection: bool = True, merge_gap_threshold: int = 1000, min_segment_duration: int = 3) -> list:
    """
    將音頻分割成小片段用於轉錄，支持重疊處理和靜音斷點切割
    
    參數:
        audio_path: 音頻檔案路徑
        segment_duration: 每個片段的持續時間（秒）
        overlap: 重疊時間（秒）
        use_silence_detection: 是否使用靜音斷點切割
        merge_gap_threshold: 合併靜音段的閾值（毫秒）
        min_segment_duration: 最小片段時長（秒）
    """
    try:
        # 檢查音頻文件是否存在
        if not os.path.exists(audio_path):
            logger.error(f"音頻文件不存在: {audio_path}")
            return []
            
        # 創建臨時目錄
        temp_dir = os.path.join(os.path.dirname(audio_path), "temp_segments")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 使用 librosa 載入音頻（更節省內存）
        try:
            import librosa
            import soundfile as sf
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            duration = len(y) / sr
            logger.info(f"音頻總長度: {duration:.2f} 秒")
            
            # 計算分段
            samples_per_segment = int(segment_duration * sr)
            overlap_samples = int(overlap * sr)
            num_segments = ceil((len(y) - overlap_samples) / (samples_per_segment - overlap_samples))
            
            def process_segment(i):
                try:
                    # 計算片段的起始和結束位置
                    start = int(i * (samples_per_segment - overlap_samples))
                    end = int(min(start + samples_per_segment, len(y)))

                    # 提取片段
                    segment = y[start:end]

                    # 🔍 加這行來檢查片段是否有聲音
                    import numpy as np
                    print(f"[DEBUG] segment_{i:03d}: max_volume={np.abs(segment).max():.4f}, length={len(segment)}")

                    # 保存片段
                    segment_path = os.path.join(temp_dir, f"segment_{i:03d}.wav")
                    sf.write(segment_path, segment, sr)

                    # 顯示檔案大小
                    print(f"[DEBUG] segment_{i:03d}.wav saved, size: {os.path.getsize(segment_path)} bytes")

                    logger.info(f"已保存片段 {i+1}/{num_segments}")
                    return segment_path

                except Exception as e:
                    logger.error(f"處理片段 {i+1} 時出錯: {str(e)}")
                    return None


            
            # 根據系統資源動態調整線程數
            import psutil
            cpu_count = psutil.cpu_count()
            max_workers = min(cpu_count - 1, 4)  # 保留一個CPU核心給系統
            
            # 使用線程池並行處理片段
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                segment_files = list(executor.map(process_segment, range(num_segments)))
            
            # 過濾掉失敗的片段
            valid_segments = [f for f in segment_files if f is not None]
            
            if not valid_segments:
                logger.error("沒有成功生成的音頻片段")
                return []
                
            return valid_segments
            
        except ImportError:
            logger.warning("librosa 不可用，嘗試使用 torchaudio...")
            # 如果 librosa 不可用，回退到 torchaudio
            waveform, sr = torchaudio.load(audio_path)
            if waveform.size(0) > 1:  # 如果是多聲道，轉換為單聲道
                waveform = waveform.mean(dim=0, keepdim=True)
                
            duration = waveform.size(1) / sr
            logger.info(f"音頻總長度: {duration:.2f} 秒")
            
            # 計算分段
            samples_per_segment = segment_duration * sr
            overlap_samples = overlap * sr
            num_segments = ceil((waveform.size(1) - overlap_samples) / (samples_per_segment - overlap_samples))
            
            def process_segment(i):
                try:
                    start = int(i * (samples_per_segment - overlap_samples))
                    end = int(min(start + samples_per_segment, waveform.size(1)))
                    segment = waveform[:, start:end]
                    segment_path = os.path.join(temp_dir, f"segment_{i:03d}.wav")
                    torchaudio.save(segment_path, segment, sr)
                    logger.info(f"已保存片段 {i+1}/{num_segments}")
                    return segment_path
                except Exception as e:
                    logger.error(f"處理片段 {i+1} 時出錯: {str(e)}")
                    return None
            
            # 使用線程池並行處理片段
            with ThreadPoolExecutor(max_workers=4) as executor:
                segment_files = list(executor.map(process_segment, range(num_segments)))
            
            # 過濾掉失敗的片段
            valid_segments = [f for f in segment_files if f is not None]
            
            if not valid_segments:
                logger.error("沒有成功生成的音頻片段")
                return []
                
            return valid_segments
            
    except Exception as e:
        logger.error(f"分割音頻時出錯: {str(e)}")
        return []
    finally:
        # 清理內存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()  # 強制垃圾回收

def merge_transcripts(transcripts: list) -> str:
    """
    合併多個轉錄文本
    """
    return " ".join(filter(None, transcripts))

def transcribe_audio(audio_path: str, video_url: str = None, output_dir: str = None,
                     use_silence_detection: bool = True, merge_gap_threshold: int = 1000,
                     min_segment_duration: int = 3, use_source_separation: bool = True,
                     track_languages: bool = True) -> str:
    """
    使用 Faster-Whisper 進行語音轉錄，支援語言自動偵測與段落語系記錄。
    """
    try:
        # 若已有轉錄檔案
        if output_dir:
            transcript_path = os.path.join(output_dir, os.path.basename(audio_path).replace('.wav', '_transcript.txt'))
            if os.path.exists(transcript_path):
                try:
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            logger.info(f"找到現有轉錄文件：{transcript_path}")
                            return content.strip()
                except Exception as e:
                    logger.warning(f"讀取現有轉錄文件時出錯：{str(e)}")

        # 嘗試使用字幕
        if video_url and output_dir:
            logger.info("嘗試從字幕文件讀取...")
            subtitle_text = extract_subtitles(video_url, output_dir)
            if subtitle_text:
                logger.info("成功讀取字幕文件")
                if output_dir:
                    try:
                        with open(transcript_path, 'w', encoding='utf-8') as f:
                            f.write(f"來源：字幕文件\n\n")
                            f.write(subtitle_text)
                    except Exception as e:
                        logger.warning(f"保存字幕內容時出錯：{str(e)}")
                return subtitle_text

        logger.info("開始進行語音辨識...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = get_whisper_model()

        # 前處理音訊
        audio_path = load_and_clean_audio(audio_path)

        # 分段
        logger.info("開始分割音頻...")
        segment_files = split_audio_for_transcription(
            audio_path,
            segment_duration=120,
            overlap=2,
            use_silence_detection=use_silence_detection,
            merge_gap_threshold=merge_gap_threshold,
            min_segment_duration=min_segment_duration
        )
        if not segment_files:
            raise Exception("音頻分割失敗")

        all_transcripts = []
        per_segment_languages = []
        total_segments = len(segment_files)

        for i, segment_path in enumerate(segment_files, 1):
            logger.info(f"正在轉錄第 {i}/{total_segments} 個片段...")
            try:
                segments, info = model.transcribe(
                    segment_path,
                    beam_size=5,
                    temperature=0.0,
                    vad_filter=False,
                    compression_ratio_threshold=5.0,
                    log_prob_threshold=-2.5,
                    no_speech_threshold=0.6,
                )

                segments = list(segments)
                transcript = "".join([s.text for s in segments]).strip()
                detected_language = info.language or "未知"
                per_segment_languages.append(detected_language)

                if not transcript or len(transcript) < 10:
                    logger.warning(f"⚠️ 第 {i+1} 段內容過短（{len(transcript)} 字元），略過")
                    continue

                if is_excessive_repetition(transcript, phrase_threshold=20, length_threshold=0.8):
                    logger.warning(f"⚠️ 第 {i+1} 段內容過度重複（可能為幻覺），略過")
                    continue
                
                # 🧠【安全過濾：過度重複】
                if any(transcript.count(phrase) >= 5 for phrase in [
                    "Thank you for watching",
                    "This is the first time I've ever seen",
                    "See you in the next video"
                ]):
                    logger.warning(f"⚠️ 第 {i} 段出現大量重複語句，視為 hallucination，略過")
                    continue

                # 🚨【語言異常提醒（不過濾）】
                if detected_language not in {"zh", "en", "ja"}:
                    logger.warning(f"⚠️ 語言偵測異常（{detected_language}），請檢查內容合理性")

                # 📝 記錄有效結果
                all_transcripts.append(transcript)
                logger.info(f"第 {i} 段轉錄完成（語言: {detected_language}，長度: {len(transcript)}）")

            except Exception as e:
                logger.error(f"轉錄第 {i} 段時出錯: {str(e)}")
                continue
            finally:
                try:
                    os.remove(segment_path)
                except:
                    pass


        final_transcript = merge_transcripts(all_transcripts)

        # 儲存結果
        if output_dir:
            try:
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write("來源：Faster-Whisper 語音辨識\n")
                    if track_languages:
                        lang_counter = {lang: per_segment_languages.count(lang) for lang in set(per_segment_languages)}
                        f.write(f"語言切換統計：{lang_counter}\n")
                        for idx, (text, lang) in enumerate(zip(all_transcripts, per_segment_languages)):
                            f.write(f"\n[{lang}] 段落 {idx+1}:\n{text.strip()}\n")
                    else:
                        f.write(f"\n偵測語言（首段）: {per_segment_languages[0] if per_segment_languages else '未知'}\n\n")
                        f.write(final_transcript)
                logger.info(f"轉錄結果已儲存至：{transcript_path}")
            except Exception as e:
                logger.warning(f"儲存轉錄時出錯：{str(e)}")

        # 清理暫存
        temp_dir = os.path.join(os.path.dirname(audio_path), "temp_segments")
        try:
            os.rmdir(temp_dir)
        except:
            pass

        return final_transcript

    except Exception as e:
        logger.error(f"轉錄音訊時出錯: {str(e)}")
        return ""
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        print(f"[DEBUG] Most repeated phrase: '{max_phrase}' appears {count} times, ratio: {ratio:.2f}")
        if count >= phrase_threshold or ratio >= length_threshold:
            return True
    return False



def compute_text_embedding(text: str) -> torch.Tensor:
    """計算文本的嵌入向量"""
    try:
        # 獲取模型（不需要重新載入）
        model = get_sentence_transformer()
        
        # 計算嵌入向量
        with torch.no_grad():
            embeddings = model.encode(text, convert_to_tensor=True)
            logger.info("已完成文本嵌入向量計算")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"計算文本嵌入向量時出錯: {str(e)}")
        return None

def is_meaningful_text(text: str, min_length: int = 10) -> tuple:
    """
    判斷文本是否有意義，主要檢查：
    1. 是否為空白或長度過短
    2. 是否存在大量重複內容（根據文本長度動態調整）
    
    參數:
        text: 要分析的文本
        min_length: 最小文本長度
    
    返回:
        tuple: (是否有意義, 原因描述)
    """
    try:
        # 檢查空白
        if not text:
            return (False, "文本為空")
        
        # 移除多餘空白
        text = ' '.join(text.split())
        if not text:
            return (False, "文本僅包含空白字符")
            
        # 檢查最小長度
        if len(text) < min_length:
            return (False, f"文本長度過短 ({len(text)} 字符)")
            
        # 檢查重複模式（根據文本長度動態調整）
        text_length = len(text)
        
        # 定義重複檢查的參數
        if text_length < 100:  # 短文本
            pattern_length = 3
            repeat_times = 3
        elif text_length < 500:  # 中等文本
            pattern_length = 5
            repeat_times = 4
        else:  # 長文本
            pattern_length = 10
            repeat_times = 5
            
        # 檢查重複模式
        pattern = f'(.{{{pattern_length},}}?)\\1{{{repeat_times-1},}}'
        matches = list(re.finditer(pattern, text))
        
        if matches:
            # 計算重複內容佔總文本的比例
            total_repeated_length = sum(len(m.group(0)) for m in matches)
            repeat_ratio = total_repeated_length / text_length
            
            # 如果重複內容超過文本的 70%，判定為無意義
            if repeat_ratio > 0.7:
                repeated_examples = [m.group(1) for m in matches[:3]]  # 取前三個重複示例
                return (False, f"存在大量重複內容（佔比 {repeat_ratio:.1%}），例如：{', '.join(repeated_examples)}")
        
        return (True, "文本有效")
        
    except Exception as e:
        logger.error(f"判斷文本意義時出錯: {str(e)}")
        return (False, f"錯誤: {str(e)}")

def text_similarity(text1: str, text2: str) -> tuple:
    """
    計算文字相似度
    
    返回:
        tuple: (相似度分數, 是否有效比對, 狀態說明)
    """
    try:
        # 如果文字為空，返回 0
        if not text1 or not text2:
            logger.warning("文字為空，返回相似度 0")
            return (0.0, False, "文本為空")
            
        # 判斷兩段文本是否有意義
        text1_meaningful, text1_reason = is_meaningful_text(text1)
        text2_meaningful, text2_reason = is_meaningful_text(text2)
        
        # 如果任一文本無意義，返回詳細原因
        if not text1_meaningful or not text2_meaningful:
            reasons = []
            if not text1_meaningful:
                reasons.append(f"文本1: {text1_reason}")
            if not text2_meaningful:
                reasons.append(f"文本2: {text2_reason}")
            logger.warning(f"檢測到無意義文本: {'; '.join(reasons)}")
            return (0.0, False, "; ".join(reasons))
            
        logger.info("開始計算文本相似度...")
        
        # 計算嵌入向量
        emb1 = compute_text_embedding(text1)
        emb2 = compute_text_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return (0.0, False, "嵌入向量計算失敗")
        
        # 計算相似度
        with torch.no_grad():
            similarity = util.pytorch_cos_sim(emb1, emb2)[0][0].item()
        
        # 根據文本長度調整權重
        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
        adjusted_similarity = similarity * (0.7 + 0.3 * len_ratio)  # 長度差異影響30%的權重
        
        logger.info(f"文本相似度計算完成: 原始={similarity:.4f}, 調整後={adjusted_similarity:.4f}")
        return (float(adjusted_similarity), True, f"有效比對，長度比例={len_ratio:.2f}")
        
    except Exception as e:
        logger.error(f"計算文本相似度時出錯: {str(e)}")
        return (0.0, False, f"錯誤: {str(e)}")
    finally:
        # 確保清理所有資源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("最終清理 GPU 記憶體完成")