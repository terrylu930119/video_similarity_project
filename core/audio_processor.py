import os
import gc
import sys
import time
import psutil
import ffmpeg
import librosa
import hashlib
import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from utils.logger import logger
from functools import lru_cache
from librosa.sequence import dtw
from pydub.silence import detect_nonsilent
from scipy.spatial.distance import euclidean
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Optional, Generator, Tuple, Any

# 檢查 CUDA 可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用設備: {device}")

# 修改全局緩存為 LRU 緩存
_feature_cache = lru_cache(maxsize=10)  # 限制只緩存最近使用的10個檔案的特徵

@lru_cache(maxsize=32)
def get_optimal_chunk_size(file_size: int) -> float:
    """根據檔案大小動態調整分塊大小"""
    base_chunk_duration = 30.0
    if file_size > 1024 * 1024 * 1024:  # 1GB
        return 15.0  # 較小的分塊
    elif file_size > 512 * 1024 * 1024:  # 512MB
        return 30.0
    else:
        return 60.0  # 較大的分塊

def load_audio(audio_path: str) -> Generator[Tuple[np.ndarray, int], None, None]:
    file_size = os.path.getsize(audio_path)
    chunk_duration = get_optimal_chunk_size(file_size)
    
    try:
        stream = librosa.stream(
            audio_path,
            block_length=int(chunk_duration * 22050),
            frame_length=2048,
            hop_length=1024  # 增加 hop_length
        )
        
        for y_block in stream:
            # 檢查記憶體使用
            if psutil.virtual_memory().percent > 80:
                gc.collect()
                time.sleep(1)
            yield y_block, 22050
            
    except Exception as e:
        logger.error(f"載入音頻文件失敗 {audio_path}: {str(e)}")
        return None

def extract_audio_features(audio_path: str, embedding_size: int = 512) -> Optional[np.ndarray]:
    """使用 PyTorch 提取音頻特徵"""
    try:
        # 載入音頻
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(device)
        
        # 使用梅爾頻譜圖作為特徵
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        ).to(device)
        
        # 計算梅爾頻譜圖
        mel_features = mel_spec(waveform)
        
        # 使用平均池化來獲取固定大小的特徵向量
        features = torch.mean(mel_features, dim=2)  # 在時間維度上平均
        
        # 轉換為 numpy 數組
        features = features.cpu().numpy()
        
        # 如果特徵維度太大，使用 PCA 降維
        if features.shape[1] > embedding_size:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=embedding_size)
            features = pca.fit_transform(features.T).T
        
        return features[0]  # 返回第一個通道的特徵
        
    except Exception as e:
        logger.error(f"提取音頻特徵失敗 {audio_path}: {str(e)}")
        return None

def parallel_feature_extraction(audio_data: np.ndarray, sr: int) -> dict:
    # 減少特徵維度
    n_mfcc = 13  # 從20減少到13
    hop_length = 1024  # 增加 hop_length 減少計算量
    
    try:
        segment_length = sr * 5  # 從10秒減少到5秒
        segments = [audio_data[i:i + segment_length] for i in range(0, len(audio_data), segment_length)]

        def extract_segment_features(segment):
            if len(segment) < sr * 0.5:
                return None

            # 使用較低的採樣率計算梅爾頻譜圖
            mel_spec = librosa.feature.melspectrogram(
                y=segment, 
                sr=sr,
                n_mels=64,  # 減少梅爾頻帶數量
                hop_length=hop_length
            )
            
            # 只計算必要的特徵
            mfcc = librosa.feature.mfcc(
                y=segment, 
                sr=sr, 
                n_mfcc=n_mfcc,
                hop_length=hop_length
            )
            
            # 只計算一階差分
            mfcc_delta = librosa.feature.delta(mfcc)
            
            # 使用較低維度的色度特徵
            chroma = librosa.feature.chroma_stft(
                y=segment, 
                sr=sr,
                hop_length=hop_length,
                n_chroma=12
            )
            
            # 使用較大的 hop_length 計算節奏特徵
            onset_env = librosa.onset.onset_strength(
                y=segment, 
                sr=sr,
                hop_length=hop_length
            )
            
            # 計算 tempo
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
            
            def get_stats(feature):
                if len(feature.shape) == 1:
                    feature = feature.reshape(1, -1)
                stats = {
                    'mean': np.mean(feature, axis=1),
                    'std': np.std(feature, axis=1),
                    'max': np.max(feature, axis=1),
                    'min': np.min(feature, axis=1),
                    'median': np.median(feature, axis=1)
                }
                # 確保所有統計量具有相同的形狀
                return {k: np.array(v, dtype=np.float32) for k, v in stats.items()}
            
            return {
                'mfcc': get_stats(mfcc),
                'mfcc_delta': get_stats(mfcc_delta),
                'chroma': get_stats(chroma),
                'onset_env': onset_env.astype(np.float32),
                'tempo': float(tempo)
            }

        with ThreadPoolExecutor(max_workers=2) as executor:
            segment_features = list(executor.map(extract_segment_features, segments))
            
        # 過濾掉無效的段
        segment_features = [f for f in segment_features if f is not None]
        if not segment_features:
            logger.error("沒有有效的音頻段可供處理")
            return None

        # 合併所有段的特徵
        combined_features = {}
        for key in segment_features[0].keys():
            if key == 'onset_env':
                # 對於時間序列特徵，保留完整序列
                combined_features[key] = np.concatenate([f[key] for f in segment_features])
            elif key == 'tempo':
                # 對於單一數值特徵，取平均
                combined_features[key] = float(np.mean([f[key] for f in segment_features]))
            else:
                # 對於統計特徵，分別合併各個統計量
                combined_features[key] = {
                    stat: np.mean([f[key][stat] for f in segment_features], axis=0).astype(np.float32)
                    for stat in segment_features[0][key].keys()
                }

        return combined_features
    except Exception as e:
        logger.error(f"特徵提取失敗: {str(e)}")
        return None

def cache_features_to_disk(features: dict, cache_dir: str, file_id: str):
    """將特徵暫存到磁碟"""
    cache_path = os.path.join(cache_dir, f"{file_id}_features.npz")
    # 確保所有特徵都被正確轉換為可序列化的格式
    processed_features = {}
    for k, v in features.items():
        if isinstance(v, dict):
            processed_features[k] = np.array(v)  # 將字典轉換為對象數組
        elif isinstance(v, (float, int)):
            processed_features[k] = np.array(v)
        else:
            processed_features[k] = v  # 保持數組不變
    np.savez_compressed(cache_path, **processed_features)
    return cache_path

def load_cached_features(cache_path: str) -> dict:
    """從磁碟載入特徵"""
    with np.load(cache_path, allow_pickle=True) as data:
        features = {}
        for k in data.files:
            if isinstance(data[k], np.ndarray):
                if data[k].dtype == np.dtype('O'):
                    features[k] = data[k].item()  # 將對象數組轉換回字典
                else:
                    features[k] = data[k]  # 保持普通數組不變
            else:
                features[k] = data[k]
        return features

def compute_audio_features(audio_path: str) -> Optional[dict]:
    try:
        cache_dir = os.path.join(os.path.dirname(audio_path), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        file_id = hashlib.md5(audio_path.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{file_id}_features.npz")
        
        # 檢查緩存
        if os.path.exists(cache_path):
            return load_cached_features(cache_path)
            
        def process_chunks():
            for y_chunk, sr in load_audio(audio_path):
                chunk_features = parallel_feature_extraction(y_chunk, sr)
                if chunk_features:
                    yield chunk_features
                gc.collect()
            
        features_list = []
        total_size = 0
        
        for chunk_feature in process_chunks():
            features_list.append(chunk_feature)
            total_size += sys.getsizeof(chunk_feature)
            
            # 如果中間結果太大，寫入磁碟
            if total_size > 500 * 1024 * 1024:  # 500MB
                intermediate = combine_features(features_list)
                if intermediate:
                    cache_features_to_disk(intermediate, cache_dir, f"{file_id}_temp")
                features_list = []
                total_size = 0
        
        if not features_list:
            logger.error("沒有有效的音頻特徵可供處理")
            return None
            
        final_features = combine_features(features_list)
        if final_features is None:
            logger.error("合併特徵失敗")
            return None
        
        # 添加深度學習特徵
        dl_features = extract_audio_features(audio_path)
        if dl_features is not None:
            final_features['dl_features'] = dl_features
        
        cache_features_to_disk(final_features, cache_dir, file_id)
        return final_features
        
    except Exception as e:
        logger.error(f"提取音頻特徵時出錯: {str(e)}")
        return None

def compute_weighted_similarity(features1: dict, features2: dict) -> float:
    try:
        # 首先檢查是否為完全相同的特徵
        if features1 is features2:  # 如果是同一個對象引用
            return 1.0

        # 檢查所有特徵是否完全相同
        all_equal = True
        for key in features1.keys():
            if key not in features2:
                all_equal = False
                break
            if key == 'onset_env':
                if not np.array_equal(features1[key], features2[key]):
                    all_equal = False
                    break
            elif key == 'tempo':
                if abs(features1[key] - features2[key]) > 1e-6:
                    all_equal = False
                    break
            elif key == 'dl_features':
                if not np.array_equal(features1[key], features2[key]):
                    all_equal = False
                    break
            else:
                for stat in features1[key].keys():
                    if not np.array_equal(features1[key][stat], features2[key][stat]):
                        all_equal = False
                        break
        
        if all_equal:
            return 1.0

        similarities = []
        weights = []

        # DTW 距離計算（用於時間序列特徵）
        onset_dtw = dtw(features1['onset_env'], features2['onset_env'])[0][-1, -1]
        # 正規化 DTW 距離
        onset_sim = 1.0 / (1.0 + onset_dtw / len(features1['onset_env']))
        similarities.append(onset_sim)
        weights.append(2.0)  # 給予節奏特徵更高權重

        # 時間序列特徵的 DTW 距離
        mfcc_dtw = dtw(features1['mfcc']['mean'], features2['mfcc']['mean'])[0][-1, -1]
        # 正規化 MFCC DTW 距離
        mfcc_sim = 1.0 / (1.0 + mfcc_dtw / len(features1['mfcc']['mean']))
        similarities.append(mfcc_sim)
        weights.append(1.5)

        # 統計特徵的餘弦相似度
        for feature_name in ['mfcc', 'mfcc_delta', 'chroma']:
            for stat in ['mean', 'std']:
                if feature_name in features1 and feature_name in features2:
                    feat1 = features1[feature_name][stat]
                    feat2 = features2[feature_name][stat]
                    # 確保向量非零
                    if np.any(feat1) and np.any(feat2):
                        sim = cosine_similarity([feat1], [feat2])[0][0]
                        # 將相似度範圍從[-1,1]調整到[0,1]
                        sim = (sim + 1) / 2
                        similarities.append(sim)
                        # 給予不同特徵不同權重
                        if feature_name in ['mfcc', 'mfcc_delta']:
                            weights.append(1.2)
                        else:
                            weights.append(0.8)

        # Tempo 差異
        tempo_diff = abs(features1['tempo'] - features2['tempo'])
        tempo_sim = 1.0 / (1.0 + tempo_diff / 20.0)  # 調整 tempo 差異的敏感度
        similarities.append(tempo_sim)
        weights.append(1.5)

        # 深度學習特徵的餘弦相似度
        if 'dl_features' in features1 and 'dl_features' in features2:
            dl_sim = cosine_similarity([features1['dl_features']], [features2['dl_features']])[0][0]
            dl_sim = (dl_sim + 1) / 2  # 轉換到 [0,1] 範圍
            similarities.append(dl_sim)
            weights.append(2.0)  # 給予深度學習特徵較高權重

        # 計算加權平均
        if not similarities:  # 如果沒有有效的相似度計算結果
            return 0.0
            
        weighted_sum = sum(s * w for s, w in zip(similarities, weights))
        total_weight = sum(weights)
        final_similarity = weighted_sum / total_weight

        # 確保相似度在 [0,1] 範圍內
        final_similarity = max(0.0, min(1.0, final_similarity))

        return float(final_similarity)
    except Exception as e:
        logger.error(f"計算加權相似度時出錯: {str(e)}")
        return 0.0

def combine_features(features_list: list) -> dict:
    """合併多個音頻塊的特徵"""
    if not features_list:
        return None
    
    combined = {}
    for key in features_list[0].keys():
        if key == 'onset_env':
            combined[key] = np.concatenate([f[key] for f in features_list])
        elif key == 'tempo':
            combined[key] = float(np.mean([f[key] for f in features_list]))
        else:
            combined[key] = {
                stat: np.mean([f[key][stat] for f in features_list], axis=0)
                for stat in features_list[0][key].keys()
            }
    return combined

def get_optimal_workers(file_size: int) -> int:
    """根據檔案大小和系統資源動態調整工作線程數"""
    available_memory = psutil.virtual_memory().available
    cpu_count = os.cpu_count() or 4
    
    if file_size > 1024 * 1024 * 1024:  # 1GB
        return min(2, cpu_count)
    elif available_memory > 8 * 1024 * 1024 * 1024:  # 8GB可用記憶體
        return min(4, cpu_count)
    else:
        return 1

def audio_similarity(path1: str, path2: str) -> float:
    try:
        file_size = max(os.path.getsize(path1), os.path.getsize(path2))
        max_workers = get_optimal_workers(file_size)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            features = list(executor.map(compute_audio_features, [path1, path2]))

        features1, features2 = features
        if features1 is None or features2 is None:
            logger.error("無法提取音頻特徵")
            return 0.0

        # 計算加權相似度
        similarity = compute_weighted_similarity(features1, features2)
        logger.info(f"加權音頻相似度: {similarity:.3f}")

        return float(similarity)
    except Exception as e:
        logger.error(f"計算音頻相似度時出錯: {str(e)}")
        return 0.0

def extract_audio(video_path: str) -> str:
    try:
        if not os.path.exists(video_path):
            logger.error(f"影片檔案不存在: {video_path}")
            raise FileNotFoundError(f"影片檔案不存在: {video_path}")

        video_dir = os.path.dirname(os.path.abspath(video_path))
        audio_dir = os.path.join(video_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(audio_dir, f"{video_name}.wav")

        if not os.access(audio_dir, os.W_OK):
            logger.error(f"沒有輸出目錄的寫入權限: {audio_dir}")
            raise PermissionError(f"沒有輸出目錄的寫入權限: {audio_dir}")

        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            logger.info(f"音訊檔案已存在: {audio_path}")
            return audio_path

        logger.info(f"開始提取音訊: {video_path} -> {audio_path}")
        ffmpeg.input(video_path).output(
            audio_path,
            acodec='pcm_s16le',
            ac=1,
            ar=16000,
            format='wav',
            audio_bitrate='192k'
        ).overwrite_output().run(quiet=True, capture_stdout=True, capture_stderr=True)

        for _ in range(5):
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                logger.info(f"音訊檔案已生成: {audio_path}")
                return audio_path
            time.sleep(1)

        logger.error("音訊檔案生成失敗或檔案大小為0")
        raise RuntimeError("音訊檔案生成失敗")
    except Exception as e:
        logger.error(f"音訊提取失敗: {str(e)}")
        raise

def detect_silence_segments(audio_path: str) -> list:
    """
    檢測音頻中的靜音段，用於優化切割點
    """
    try:
        audio = AudioSegment.from_wav(audio_path)
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=300,  # 最小静音长度（毫秒）
            silence_thresh = audio.dBFS - 14 # 静音阈值（dB）
        )
        
        return nonsilent_ranges
    except Exception as e:
        logger.error(f"檢測靜音段時出錯: {str(e)}")
        return []