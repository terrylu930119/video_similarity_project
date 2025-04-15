import whisper
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from logger import logger
from gpu_utils import gpu_manager
import yt_dlp
import os
import re
from itertools import groupby
from audio_processor import load_audio
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
from math import ceil

# 全域模型變數
_whisper_model = None
_sentence_transformer = None

def get_whisper_model():
    """
    獲取全域 Whisper 模型實例，如果未載入則進行載入
    """
    global _whisper_model
    if _whisper_model is None:
        logger.info("正在載入 Whisper medium 模型...")
        _whisper_model = whisper.load_model("medium")
        # 如果有 GPU，將模型移至 GPU
        if gpu_manager.is_pytorch_cuda_available():
            logger.info("將 Whisper 模型移至 GPU")
            _whisper_model = _whisper_model.to(gpu_manager.get_pytorch_device())
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
        if gpu_manager.is_pytorch_cuda_available():
            logger.info("將 SentenceTransformer 模型移至 GPU")
            _sentence_transformer = _sentence_transformer.to(gpu_manager.get_pytorch_device())
        logger.info("SentenceTransformer 模型載入完成")
    return _sentence_transformer

def get_subtitle_language(filename: str) -> str:
    """從字幕文件名中提取語言代碼"""
    # 文件名格式：videoId_index.language.vtt
    parts = filename.split('.')
    if len(parts) >= 3:
        return parts[-2]  # 返回倒數第二個部分（語言代碼）
    return ""

def get_preferred_subtitle(subtitle_files: list, video_id: str) -> str:
    """
    根據優先順序選擇字幕文件
    優先順序：繁體中文 > 簡體中文 > 中文 > 英文 > 其他
    """
    language_priority = {
        'zh': 2,       # 中文（未指定）
        'en': 3,       # 英文
        'ja':1         # 日文
    }
    
    # 過濾出屬於當前影片的字幕文件
    video_subtitles = [f for f in subtitle_files if f.startswith(video_id)]
    
    # 將字幕文件按語言優先級排序
    sorted_files = []
    for file in video_subtitles:
        lang = get_subtitle_language(file)
        priority = language_priority.get(lang, 999)  # 其他語言優先級最低
        sorted_files.append((priority, file))
    
    sorted_files.sort()  # 按優先級排序
    return sorted_files[0][1] if sorted_files else ""

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
        video_url: YouTube 影片 URL
        output_dir: 輸出目錄
    
    返回:
        字幕文本，如果沒有字幕則返回空字符串
    """
    try:
        # 從 URL 中提取影片 ID
        video_id = extract_video_id_from_url(video_url)
        if not video_id:
            logger.error("無法從 URL 提取影片 ID")
            return ""
            
        # 檢查字幕文件
        subtitle_files = [f for f in os.listdir(output_dir) if f.startswith(video_id) and f.endswith('.vtt')]
        
        if not subtitle_files:
            logger.info("未找到字幕文件")
            return ""
            
        # 列出所有可用的字幕語言
        available_languages = [get_subtitle_language(f) for f in subtitle_files]
        logger.info(f"可用的字幕語言: {', '.join(available_languages)}")
        
        # 選擇優先語言的字幕
        preferred_subtitle = get_preferred_subtitle(subtitle_files, video_id)
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

def split_audio_for_transcription(audio_path: str, segment_duration: int = 180) -> list:
    """
    將音頻分割成小片段用於轉錄
    
    參數:
        audio_path: 音頻檔案路徑
        segment_duration: 每個片段的持續時間（秒），預設3分鐘
    """
    try:
        # 使用 audio_processor 的 load_audio 函數
        audio_data = load_audio(audio_path)
        if audio_data is None:
            return []
            
        y, sr = audio_data
        duration = len(y) / sr
        logger.info(f"音頻總長度: {duration:.2f} 秒")
        
        # 計算分段
        samples_per_segment = segment_duration * sr
        num_segments = ceil(len(y) / samples_per_segment)
        logger.info(f"將分割成 {num_segments} 個片段，每段 {segment_duration} 秒")
        
        # 創建臨時目錄
        temp_dir = os.path.join(os.path.dirname(audio_path), "temp_segments")
        os.makedirs(temp_dir, exist_ok=True)
        
        def process_segment(i):
            try:
                # 計算片段的起始和結束位置
                start = i * samples_per_segment
                end = min((i + 1) * samples_per_segment, len(y))
                
                # 提取片段
                segment = y[int(start):int(end)]
                
                # 保存片段
                segment_path = os.path.join(temp_dir, f"segment_{i:03d}.wav")
                sf.write(segment_path, segment, sr)
                logger.info(f"已保存片段 {i+1}/{num_segments}")
                return segment_path
            except Exception as e:
                logger.error(f"處理片段 {i+1} 時出錯: {str(e)}")
                return None
        
        # 使用線程池並行處理片段
        with ThreadPoolExecutor(max_workers=4) as executor:
            segment_files = list(executor.map(process_segment, range(num_segments)))
        
        # 過濾掉失敗的片段
        return [f for f in segment_files if f is not None]
        
    except Exception as e:
        logger.error(f"分割音頻時出錯: {str(e)}")
        return []

def merge_transcripts(transcripts: list) -> str:
    """
    合併多個轉錄文本
    """
    return " ".join(filter(None, transcripts))

def transcribe_audio(audio_path: str, video_url: str = None, output_dir: str = None, language: str = None) -> str:
    """
    轉錄音訊，優先順序：
    1. 檢查已存在的轉錄文件
    2. 檢查字幕文件（如果提供了影片 URL）
    3. 進行語音辨識
    """
    try:
        # 1. 首先檢查是否已有轉錄文件
        if output_dir:
            transcript_path = os.path.join(output_dir, os.path.basename(audio_path).replace('.wav', '_transcript.txt'))
            if os.path.exists(transcript_path):
                try:
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            logger.info(f"找到現有轉錄文件：{transcript_path}")
                            return content.strip()
                        else:
                            logger.info("現有轉錄文件為空，將重新轉錄")
                except Exception as e:
                    logger.warning(f"讀取現有轉錄文件時出錯：{str(e)}，將重新轉錄")

        # 2. 如果提供了影片 URL，嘗試提取字幕
        if video_url and output_dir:
            logger.info("嘗試從字幕文件讀取...")
            subtitle_text = extract_subtitles(video_url, output_dir)
            if subtitle_text:
                logger.info("成功讀取字幕文件")
                # 保存字幕內容為轉錄文件
                if output_dir:
                    try:
                        with open(transcript_path, 'w', encoding='utf-8') as f:
                            f.write(f"來源：字幕文件\n\n")
                            f.write(subtitle_text)
                        logger.info(f"已將字幕內容保存為轉錄文件：{transcript_path}")
                    except Exception as e:
                        logger.warning(f"保存字幕內容時出錯：{str(e)}")
                return subtitle_text
            logger.info("未找到可用的字幕，將使用語音辨識")

        # 3. 進行語音辨識
        logger.info("開始進行語音辨識...")
        
        # 清理 GPU 記憶體
        if gpu_manager.is_pytorch_cuda_available():
            torch.cuda.empty_cache()
        
        # 獲取模型
        model = get_whisper_model()
        
        # 分割音頻（使用較短的片段）
        logger.info("開始分割音頻...")
        segment_files = split_audio_for_transcription(audio_path, segment_duration=180)  # 3分鐘一段
        if not segment_files:
            raise Exception("音頻分割失敗")
        
        # 逐段轉錄
        all_transcripts = []
        total_segments = len(segment_files)
        first_detected_language = None
        
        for i, segment_path in enumerate(segment_files, 1):
            logger.info(f"正在轉錄第 {i}/{total_segments} 個片段...")
            try:
                result = model.transcribe(
                    segment_path,
                    task="transcribe",
                    language=language,
                    temperature=0.0,
                    best_of=5,
                    beam_size=5,
                    fp16=gpu_manager.is_pytorch_cuda_available(),
                    verbose=False,
                )
                
                transcript = result["text"].strip()
                detected_language = result.get("language", "未知")
                
                if i == 1:  # 只記錄第一個片段檢測到的語言
                    first_detected_language = detected_language
                
                # 後處理轉錄文本
                transcript = post_process_transcript(transcript, language or detected_language)
                all_transcripts.append(transcript)
                
                logger.info(f"第 {i} 個片段轉錄完成，長度: {len(transcript)} 字符")
                
            except Exception as e:
                logger.error(f"轉錄第 {i} 個片段時出錯: {str(e)}")
                continue
            
            finally:
                # 刪除臨時片段文件
                try:
                    os.remove(segment_path)
                except:
                    pass
        
        # 合併所有轉錄結果
        final_transcript = merge_transcripts(all_transcripts)
        
        # 儲存完整轉錄結果
        if output_dir:
            try:
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(f"來源：語音辨識\n")
                    f.write(f"檢測到的語言: {first_detected_language}\n\n")
                    f.write(final_transcript)
                logger.info(f"已將轉錄結果儲存至：{transcript_path}")
            except Exception as e:
                logger.warning(f"保存轉錄結果時出錯：{str(e)}")
        
        # 清理臨時目錄
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
        if gpu_manager.is_pytorch_cuda_available():
            torch.cuda.empty_cache()

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
        if gpu_manager.is_pytorch_cuda_available():
            torch.cuda.empty_cache()
            logger.info("最終清理 GPU 記憶體完成")