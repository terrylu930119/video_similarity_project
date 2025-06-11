import os
import sys
import time
import torch
import psutil
import ffmpeg
import librosa
import hashlib
import torchaudio
import torchopenl3
import numpy as np
from pydub import AudioSegment
from utils.logger import logger
from functools import lru_cache
from librosa.sequence import dtw
from sklearn.decomposition import PCA
from panns_inference.models import Cnn14
from pydub.silence import detect_nonsilent
from typing import Optional, Generator, Tuple
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from panns_inference.inference import load_audio as pann_load_audio

sys.path.insert(0, os.path.abspath('./panns_inference'))

# =============== 全局配置参数 ===============
# 音频处理参数
AUDIO_CONFIG = {
    'sample_rate': 32000,  # 统一使用32kHz采样率
    'channels': 1,
    'audio_bitrate': '192k',
    'format': 'wav',
    'codec': 'pcm_s16le'
}

# 特征提取参数
FEATURE_CONFIG = {
    'mfcc': {
        'n_mfcc': 13,
        'hop_length': 1024
    },
    'mel': {
        'n_mels': 64,
        'hop_length': 1024
    },
    'chroma': {
        'n_chroma': 12,
        'hop_length': 1024
    }
}

# 分块处理参数
CHUNK_CONFIG = {
    'small_file': 60.0,    # 小文件分块大小（秒）
    'medium_file': 30.0,   # 中等文件分块大小（秒）
    'large_file': 15.0,    # 大文件分块大小（秒）
    'file_size_threshold': {
        'large': 1024 * 1024 * 1024,  # 1GB
        'medium': 512 * 1024 * 1024   # 512MB
    }
}

# 相似度计算权重
SIMILARITY_WEIGHTS = {
    'pann': 2.5,
    'dl': 2.0,
    'onset': 2.0,
    'mfcc': 1.5,
    'mfcc_delta': 1.2,
    'chroma': 0.8,
    'tempo': 1.5,
    'openl3': 2.0
}

# 内存管理参数
MEMORY_CONFIG = {
    'max_memory_percent': 70,
    'intermediate_cache_size': 1000 * 1024 * 1024,  # 1GB
    'feature_cache_size': 10
}

# 线程池配置
THREAD_CONFIG = {
    'max_workers': {
        'large_file': 2,
        'medium_file': 4,
        'small_file': 1
    }
}

# 裁剪参数
CROP_CONFIG = {
    'min_duration': 30.0,     # 最小裁剪时长（秒）
    'max_duration': 300.0,   # 最大裁剪时长（秒）
    'overlap': 0.5,         # 重叠时长（秒）
    'silence_threshold': -14 # 静音检测阈值（dB）
}

# =============== 全局变量初始化 ===============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用設備: {device}")

# 初始化 PANN 模型
pann_model = None

# 初始化特征缓存
_feature_cache = lru_cache(maxsize=MEMORY_CONFIG['feature_cache_size'])

# =============== 工具函数 ===============
def get_optimal_chunk_size(file_size: int) -> float:
    """根據檔案大小動態調整分塊大小"""
    if file_size > CHUNK_CONFIG['file_size_threshold']['large']:
        return CHUNK_CONFIG['large_file']
    elif file_size > CHUNK_CONFIG['file_size_threshold']['medium']:
        return CHUNK_CONFIG['medium_file']
    return CHUNK_CONFIG['small_file']

def get_optimal_workers(file_size: int) -> int:
    """根據檔案大小和系統資源動態調整工作線程數"""
    available_memory = psutil.virtual_memory().available
    cpu_count = os.cpu_count() or 4
    
    if file_size > CHUNK_CONFIG['file_size_threshold']['large']:
        return min(THREAD_CONFIG['max_workers']['large_file'], cpu_count)
    elif available_memory > 8 * 1024 * 1024 * 1024:  # 8GB可用記憶體
        return min(THREAD_CONFIG['max_workers']['medium_file'], cpu_count)
    return THREAD_CONFIG['max_workers']['small_file']

def check_memory_usage():
    """檢查記憶體使用情況，必要時進行垃圾回收"""
    if psutil.virtual_memory().percent > MEMORY_CONFIG['max_memory_percent']:
        import gc
        gc.collect()
        time.sleep(1)
        return True
    return False

# 初始化 OpenL3 模型
openl3_model = None

def get_openl3_model():
    global openl3_model
    if openl3_model is None:
        openl3_model = torchopenl3.models.load_audio_embedding_model(
            input_repr="mel128",
            content_type="music",
            embedding_size=512,
            device=device
        )
        openl3_model.eval()
    return openl3_model

def extract_openl3_features(audio_path: str) -> Optional[np.ndarray]:
    try:
        # 使用 librosa 載入音頻（openl3 需 stereo）
        audio, sr = librosa.load(audio_path, sr=AUDIO_CONFIG['sample_rate'], mono=False)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)  # 強制成 (channels, samples)

        model = get_openl3_model()
        emb, _ = torchopenl3.get_audio_embedding(
            audio,
            sr,
            model=model,
            hop_size=1.0,
            center=True,
            verbose=False
        )
        # emb.shape: (T, 512)
        return np.mean(emb, axis=0).astype(np.float32)
    except Exception as e:
        logger.error(f"OpenL3 特徵提取失敗 {audio_path}: {str(e)}")
        return None
    
def get_pann_model():
    """獲取或初始化 PANN 模型"""
    global pann_model
    if pann_model is None:
        pann_model = Cnn14(sample_rate=AUDIO_CONFIG['sample_rate'], window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000)
        pann_model = pann_model.to(device)
        pann_model.eval()
    return pann_model

def extract_pann_features(audio_path: str) -> Optional[np.ndarray]:
    """使用 PANN 提取音頻特徵"""
    try:
        # 載入音頻
        waveform, sr = pann_load_audio(audio_path, target_sr=AUDIO_CONFIG['sample_rate'])
        waveform = torch.from_numpy(waveform).to(device)
        
        # 獲取模型
        model = get_pann_model()
        
        # 提取特徵
        with torch.no_grad():
            features = model(waveform.unsqueeze(0))
            features = features.cpu().numpy()
        
        return features.squeeze()[:2048]  # 保證回傳是 1D 向量
        
    except Exception as e:
        logger.error(f"PANN 特徵提取失敗 {audio_path}: {str(e)}")
        return None

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
    
def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """對 waveform 進行音量與能量正規化處理，降低編碼差異影響"""
    waveform = waveform - waveform.mean()
    waveform = waveform / waveform.abs().max().clamp(min=1e-6)
    waveform = waveform / (waveform.norm(p=2) + 1e-9)
    return waveform

def extract_audio_features(audio_path: str, embedding_size: int = 512) -> Optional[np.ndarray]:
    """使用 PyTorch 提取音頻特徵"""
    try:
        # 載入音頻
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = normalize_waveform(waveform)
        waveform = waveform.to(device)
        
        # 使用梅爾頻譜圖作為特徵
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=FEATURE_CONFIG['mel']['hop_length'],
            n_mels=FEATURE_CONFIG['mel']['n_mels']
        ).to(device)
        
        # 計算梅爾頻譜圖
        mel_features = mel_spec(waveform)
        
        # 使用平均池化來獲取固定大小的特徵向量
        features = torch.mean(mel_features, dim=2)  # 在時間維度上平均
        
        # 轉換為 numpy 數組
        features = features.cpu().numpy()
        
        # 如果特徵維度太大，使用 PCA 降維
        if features.shape[1] > embedding_size:
            pca = PCA(n_components=embedding_size)
            features = pca.fit_transform(features.T).T
        
        return features[0]  # 返回第一個通道的特徵
        
    except Exception as e:
        logger.error(f"提取音頻特徵失敗 {audio_path}: {str(e)}")
        return None

def parallel_feature_extraction(audio_data: np.ndarray, sr: int) -> dict:
    try:
        segment_length = sr * 5  # 5秒一段
        segments = [audio_data[i:i + segment_length] for i in range(0, len(audio_data), segment_length)]

        def extract_segment_features(segment):
            if len(segment) < sr * 0.5:
                return None

            # 使用較低的採樣率計算梅爾頻譜圖
            mel_spec = librosa.feature.melspectrogram(
                y=segment, 
                sr=sr,
                n_mels=FEATURE_CONFIG['mel']['n_mels'],
                hop_length=FEATURE_CONFIG['mel']['hop_length']
            )
            
            # 計算 MFCC
            mfcc = librosa.feature.mfcc(
                y=segment, 
                sr=sr, 
                n_mfcc=FEATURE_CONFIG['mfcc']['n_mfcc'],
                hop_length=FEATURE_CONFIG['mfcc']['hop_length']
            )
            
            # 計算 MFCC delta
            mfcc_delta = librosa.feature.delta(mfcc)
            
            # 計算色度特徵
            chroma = librosa.feature.chroma_stft(
                y=segment, 
                sr=sr,
                hop_length=FEATURE_CONFIG['chroma']['hop_length'],
                n_chroma=FEATURE_CONFIG['chroma']['n_chroma']
            )
            
            # 計算節奏特徵
            onset_env = librosa.onset.onset_strength(
                y=segment, 
                sr=sr,
                hop_length=FEATURE_CONFIG['mel']['hop_length']
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
                combined_features[key] = np.concatenate([f[key] for f in segment_features])
            elif key == 'tempo':
                combined_features[key] = float(np.mean([f[key] for f in segment_features]))
            else:
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
    processed_features = {}
    for k, v in features.items():
        if isinstance(v, dict):
            processed_features[k] = np.array(v)
        elif isinstance(v, (float, int)):
            processed_features[k] = np.array(v)
        else:
            processed_features[k] = v
    np.savez_compressed(cache_path, **processed_features)
    return cache_path

def load_cached_features(cache_path: str) -> dict:
    """從磁碟載入特徵"""
    with np.load(cache_path, allow_pickle=True) as data:
        features = {}
        for k in data.files:
            if isinstance(data[k], np.ndarray):
                if data[k].dtype == np.dtype('O'):
                    features[k] = data[k].item()
                else:
                    features[k] = data[k]
            else:
                features[k] = data[k]
        return features

def compute_weighted_similarity(features1: dict, features2: dict) -> float:
    try:
        # 首先檢查是否為完全相同的特徵
        if features1 is features2:
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
            elif key in ['dl_features', 'pann_features', 'openl3_features']:
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

        # 对时间序列特征进行长度归一化
        min_length = min(len(features1['onset_env']), len(features2['onset_env']))
        onset_env1 = features1['onset_env'][:min_length]
        onset_env2 = features2['onset_env'][:min_length]
        
        # DTW 距离计算（用于时间序列特征）
        onset_dtw = dtw(onset_env1, onset_env2)[0][-1, -1]
        onset_sim = 1.0 / (1.0 + onset_dtw / min_length)
        similarities.append(onset_sim)
        weights.append(SIMILARITY_WEIGHTS['onset'])

        # 时间序列特征的 DTW 距离
        min_mfcc_length = min(len(features1['mfcc']['mean']), len(features2['mfcc']['mean']))
        mfcc1 = features1['mfcc']['mean'][:min_mfcc_length]
        mfcc2 = features2['mfcc']['mean'][:min_mfcc_length]
        mfcc_dtw = dtw(mfcc1, mfcc2)[0][-1, -1]
        mfcc_sim = 1.0 / (1.0 + mfcc_dtw / min_mfcc_length)
        similarities.append(mfcc_sim)
        weights.append(SIMILARITY_WEIGHTS['mfcc'])

        # 統計特徵的餘弦相似度
        for feature_name in ['mfcc', 'mfcc_delta', 'chroma']:
            for stat in ['mean', 'std']:
                if feature_name in features1 and feature_name in features2:
                    feat1 = features1[feature_name][stat]
                    feat2 = features2[feature_name][stat]
                    if np.any(feat1) and np.any(feat2):
                        sim = cosine_similarity([feat1], [feat2])[0][0]
                        sim = (sim + 1) / 2
                        similarities.append(sim)
                        weights.append(SIMILARITY_WEIGHTS[feature_name])

        # Tempo 差異
        tempo_diff = abs(features1['tempo'] - features2['tempo'])
        tempo_sim = 1.0 / (1.0 + tempo_diff / 30.0)  # 增加容错范围
        similarities.append(tempo_sim)
        weights.append(SIMILARITY_WEIGHTS['tempo'])

        # 深度學習特徵的餘弦相似度
        if 'dl_features' in features1 and 'dl_features' in features2:
            dl_sim = cosine_similarity([features1['dl_features']], [features2['dl_features']])[0][0]
            dl_sim = (dl_sim + 1) / 2
            similarities.append(dl_sim)
            weights.append(SIMILARITY_WEIGHTS['dl'])
            
        # PANN 特徵的餘弦相似度
        if 'pann_features' in features1 and 'pann_features' in features2:
            pann_sim = cosine_similarity([features1['pann_features']], [features2['pann_features']])[0][0]
            pann_sim = (pann_sim + 1) / 2
            similarities.append(pann_sim)
            weights.append(SIMILARITY_WEIGHTS['pann'])

        # OpenL3 特徵的餘弦相似度
        if 'openl3_features' in features1 and 'openl3_features' in features2:
            l3_sim = cosine_similarity([features1['openl3_features']], [features2['openl3_features']])[0][0]
            l3_sim = (l3_sim + 1) / 2  # 轉為 [0, 1] 區間
            similarities.append(l3_sim)
            weights.append(SIMILARITY_WEIGHTS['openl3'])

        # 計算加權平均
        if not similarities:
            return 0.0
            
        weighted_sum = sum(s * w for s, w in zip(similarities, weights))
        total_weight = sum(weights)
        final_similarity = weighted_sum / total_weight

        return float(max(0.0, min(1.0, final_similarity)))
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

def crop_audio(audio_path: str, start_time: float, end_time: float, output_path: Optional[str] = None) -> str:
    """
    裁剪音频文件
    
    Args:
        audio_path: 输入音频文件路径
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）
        output_path: 输出文件路径（可选）
        
    Returns:
        str: 裁剪后的音频文件路径
    """
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
            
        # 验证裁剪参数
        duration = end_time - start_time
        if duration < CROP_CONFIG['min_duration']:
            raise ValueError(f"裁剪时长过短，最小需要 {CROP_CONFIG['min_duration']} 秒")
        if duration > CROP_CONFIG['max_duration']:
            raise ValueError(f"裁剪时长过长，最大允许 {CROP_CONFIG['max_duration']} 秒")
            
        if output_path is None:
            # 生成输出文件路径
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_dir = os.path.join(os.path.dirname(audio_path), "cropped")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{base_name}_cropped_{start_time}_{end_time}.wav")
            
        # 使用 ffmpeg 进行裁剪
        stream = ffmpeg.input(audio_path)
        stream = ffmpeg.output(
            stream,
            output_path,
            ss=start_time,
            t=duration,
            acodec=AUDIO_CONFIG['codec'],
            ac=AUDIO_CONFIG['channels'],
            ar=AUDIO_CONFIG['sample_rate'],
            format=AUDIO_CONFIG['format'],
            audio_bitrate=AUDIO_CONFIG['audio_bitrate']
        )
        ffmpeg.run(stream, quiet=True, capture_stdout=True, capture_stderr=True)
        
        # 验证输出文件
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("音频裁剪失败")
            
        logger.info(f"音频裁剪成功: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"音频裁剪失败: {str(e)}")
        raise

def compute_audio_features(audio_path: str, start_time: Optional[float] = None, end_time: Optional[float] = None) -> Optional[dict]:
    """整合所有特徵提取過程的主函數"""
    try:
        # 如果需要裁剪
        if start_time is not None and end_time is not None:
            audio_path = crop_audio(audio_path, start_time, end_time)
            
        cache_dir = os.path.join(os.path.dirname(audio_path), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        file_id = hashlib.md5(audio_path.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{file_id}_features.npz")
        
        # 檢查緩存
        if os.path.exists(cache_path):
            return load_cached_features(cache_path)
            
        def process_chunks():
            for y_chunk, sr in load_audio(audio_path):
                if check_memory_usage():
                    logger.warning("記憶體使用率過高，進行垃圾回收")
                chunk_features = parallel_feature_extraction(y_chunk, sr)
                if chunk_features:
                    yield chunk_features
            
        features_list = []
        total_size = 0
        
        for chunk_feature in process_chunks():
            features_list.append(chunk_feature)
            total_size += sys.getsizeof(chunk_feature)
            
            # 如果中間結果太大，寫入磁碟
            if total_size > MEMORY_CONFIG['intermediate_cache_size']:
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
            
        # 添加 PANN 特徵
        pann_features = extract_pann_features(audio_path)
        if pann_features is not None:
            final_features['pann_features'] = pann_features

        # 添加 OpenL3 特徵
        openl3_features = extract_openl3_features(audio_path)
        if openl3_features is not None:
            final_features['openl3_features'] = openl3_features
            
        # 緩存最終特徵
        cache_features_to_disk(final_features, cache_dir, file_id)
        return final_features
        
    except Exception as e:
        logger.error(f"提取音頻特徵時出錯: {str(e)}")
        return None

def audio_similarity(path1: str, path2: str, start_time1: Optional[float] = None, end_time1: Optional[float] = None, 
                    start_time2: Optional[float] = None, end_time2: Optional[float] = None) -> float:
    """
    计算两个音频文件的相似度，支持指定时间范围
    
    Args:
        path1: 第一个音频文件路径
        path2: 第二个音频文件路径
        start_time1: 第一个音频的开始时间（秒）
        end_time1: 第一个音频的结束时间（秒）
        start_time2: 第二个音频的开始时间（秒）
        end_time2: 第二个音频的结束时间（秒）
        
    Returns:
        float: 相似度分数（0-1之间）
    """
    try:
        file_size = max(os.path.getsize(path1), os.path.getsize(path2))
        max_workers = get_optimal_workers(file_size)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            features = list(executor.map(
                lambda x: compute_audio_features(x[0], x[1], x[2]),
                [(path1, start_time1, end_time1), (path2, start_time2, end_time2)]
            ))

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
    """
    从视频文件中提取音频
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        str: 提取的音频文件路径
    """
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
            acodec=AUDIO_CONFIG['codec'],
            ac=AUDIO_CONFIG['channels'],
            ar=AUDIO_CONFIG['sample_rate'],
            format=AUDIO_CONFIG['format'],
            audio_bitrate=AUDIO_CONFIG['audio_bitrate']
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
            silence_thresh=audio.dBFS + CROP_CONFIG['silence_threshold']  # 使用配置的静音阈值
        )
        
        return nonsilent_ranges
    except Exception as e:
        logger.error(f"檢測靜音段時出錯: {str(e)}")
        return []