import os
import sys
import gc
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    'codec': 'pcm_s16le',
    'force_gpu': True  # 强制使用 GPU
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
    'pann': 1.2,
    'dl': 2.8,
    'onset': 0.8,
    'mfcc': 1.5,
    'mfcc_delta': 0.7,
    'chroma': 1.2,
    'tempo': 1.4,
    'openl3': 2.8
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
    'min_duration': 30.0,     # 最小裁剪時長（秒）
    'max_duration': 300.0,   # 最大裁剪時長（秒）
    'overlap': 0.5,         # 重叠時長（秒）
    'silence_threshold': -14 # 靜音檢測閾值（dB）
}

# =============== 全局变量初始化 ===============
device = torch.device('cuda' if torch.cuda.is_available() and AUDIO_CONFIG['force_gpu'] else 'cpu')
logger.info(f"使用設備: {device}")

# 初始化 PANN 模型
pann_model = None

# 初始化特征缓存
_feature_cache = lru_cache(maxsize=MEMORY_CONFIG['feature_cache_size'])

# =============== 工具函数 ===============
def log_memory_usage(stage: str = ""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    logger.info(f"{stage} 記憶體使用量: {mem:.2f} MB")

def split_audio_by_duration(audio_path: str, split_sec: float = 300.0) -> list:
    """
    將音訊檔案每 split_sec 秒切一段，返回所有片段的路徑
    """
    try:
        audio = AudioSegment.from_wav(audio_path)
        duration_ms = len(audio)  # 總長度（毫秒）
        segments = []
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_dir = os.path.join(os.path.dirname(audio_path), "splits")
        os.makedirs(output_dir, exist_ok=True)

        for i, start_ms in enumerate(range(0, duration_ms, int(split_sec * 1000))):
            end_ms = min(start_ms + int(split_sec * 1000), duration_ms)
            segment = audio[start_ms:end_ms]
            output_path = os.path.join(output_dir, f"{base_name}_part{i+1}.wav")
            segment.export(output_path, format="wav")
            segments.append(output_path)

        logger.info(f"分割音訊為 {len(segments)} 段，每段約 {split_sec} 秒")
        return segments
    except Exception as e:
        logger.error(f"音訊分段失敗: {str(e)}")
        return []

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
            embedding_size=512
        )
        openl3_model = openl3_model.to(device)
        if device.type == 'cuda':
            openl3_model = torch.nn.DataParallel(openl3_model)  # 使用多 GPU 支持
        openl3_model.eval()
    return openl3_model

def extract_openl3_features_chunked(audio_path: str, chunk_sec: float = 10.0) -> Optional[np.ndarray]:
    try:
        # ====== 📦 音訊載入並統一格式 ======
        audio, sr = librosa.load(audio_path, sr=None, mono=False)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # ====== 📐 重採樣至 OpenL3 支援的標準 48kHz（或 32kHz）=======
        target_sr = 48000
        if sr != target_sr:
            audio = librosa.resample(audio[0], orig_sr=sr, target_sr=target_sr)
            audio = np.expand_dims(audio, axis=0)
            sr = target_sr

        chunk_size = int(chunk_sec * sr)
        model = get_openl3_model()
        embeddings = []

        # ====== 🧩 分段嵌入處理 ======
        for i in range(0, audio.shape[1], chunk_size):
            chunk = audio[:, i:i + chunk_size]
            if chunk.shape[1] < sr:  # 少於 1 秒略過
                continue

            with torch.no_grad():
                emb, _ = torchopenl3.get_audio_embedding(
                    chunk, sr,
                    model=model,
                    hop_size=1.0,
                    center=True,
                    verbose=False
                )

            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            if emb.size > 0:
                embeddings.append(np.mean(emb, axis=0))

            del emb, chunk
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # ====== 🧪 合併輸出結果 ======
        if not embeddings:
            return None
        return np.mean(np.stack(embeddings), axis=0).astype(np.float32)

    except Exception as e:
        logger.error(f"OpenL3 特徵提取失敗 (分段處理) {audio_path}: {str(e)}")
        return None

def get_pann_model():
    """獲取或初始化 PANN 模型"""
    global pann_model
    if pann_model is None:
        pann_model = Cnn14(
            sample_rate=AUDIO_CONFIG['sample_rate'],
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527
        ).to(device)
        pann_model.eval()  # 不使用 DataParallel，減少記憶體負擔
    return pann_model

def extract_pann_features_chunked(audio_path: str, chunk_sec: float = 10.0) -> Optional[np.ndarray]:
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != AUDIO_CONFIG['sample_rate']:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=AUDIO_CONFIG['sample_rate'])
            waveform = transform(waveform)
            sr = AUDIO_CONFIG['sample_rate']

        waveform = waveform.to(device)
        chunk_size = int(chunk_sec * sr)

        chunks = [
            waveform[:, i:i + chunk_size]
            for i in range(0, waveform.shape[1], chunk_size)
            if waveform[:, i:i + chunk_size].shape[1] >= sr  # 至少 1 秒
        ]

        if not chunks:
            logger.warning(f"PANN 無有效音訊片段: {audio_path}")
            return None

        model = get_pann_model()
        embeddings = []

        for chunk in chunks:
            with torch.no_grad():
                out = model(chunk)
                emb = out['embedding'].squeeze().detach().cpu().numpy()
                embeddings.append(emb[:2048])

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return np.mean(np.stack(embeddings), axis=0).astype(np.float32)

    except Exception as e:
        logger.error(f"PANN 特徵提取失敗（分段處理）{audio_path}: {str(e)}")
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

def extract_audio_features_chunked_streaming(audio_path: str, chunk_sec: float = 10.0, embedding_size: int = 512) -> Optional[np.ndarray]:
    try:
        y, sr = librosa.load(audio_path, sr=AUDIO_CONFIG['sample_rate'])
        chunk_size = int(chunk_sec * sr)
        mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=2048,
            hop_length=FEATURE_CONFIG['mel']['hop_length'],
            n_mels=FEATURE_CONFIG['mel']['n_mels']
        ).to(device)

        chunk_features = []

        for i in range(0, len(y), chunk_size):
            chunk = y[i:i + chunk_size]
            if len(chunk) < sr:
                continue
            waveform = torch.tensor(chunk).unsqueeze(0).to(device)
            waveform = normalize_waveform(waveform)

            mel_spec = mel_spec_transform(waveform)
            pooled = torch.mean(mel_spec, dim=2).squeeze().cpu().numpy()
            chunk_features.append(pooled)

            del mel_spec, waveform
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        if not chunk_features:
            return None

        avg_feature = np.mean(np.stack(chunk_features), axis=0)

        if avg_feature.shape[0] > embedding_size:
            pca = PCA(n_components=embedding_size)
            avg_feature = pca.fit_transform(avg_feature.reshape(1, -1))[0]

        return avg_feature.astype(np.float32)

    except Exception as e:
        logger.error(f"Streaming 特徵提取失敗: {str(e)}")
        return None

def parallel_feature_extraction(audio_data: np.ndarray, sr: int) -> Optional[dict]:

    segment_length = sr * 10  # 每段 10 秒
    segments = [audio_data[i:i + segment_length] for i in range(0, len(audio_data), segment_length)]
    if not segments:
        logger.error("音頻數據為空")
        return None

    def extract_segment_features(segment):
        try:
            if len(segment) < sr * 0.5:
                return None

            mel_spec = librosa.feature.melspectrogram(
                y=segment, sr=sr,
                n_mels=FEATURE_CONFIG['mel']['n_mels'],
                hop_length=FEATURE_CONFIG['mel']['hop_length']
            )
            mfcc = librosa.feature.mfcc(
                y=segment, sr=sr,
                n_mfcc=FEATURE_CONFIG['mfcc']['n_mfcc'],
                hop_length=FEATURE_CONFIG['mfcc']['hop_length']
            )
            mfcc_delta = librosa.feature.delta(mfcc)
            chroma = librosa.feature.chroma_stft(
                y=segment, sr=sr,
                hop_length=FEATURE_CONFIG['chroma']['hop_length'],
                n_chroma=FEATURE_CONFIG['chroma']['n_chroma']
            )
            onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

            def get_stats(feature):
                if len(feature.shape) == 1:
                    feature = feature.reshape(1, -1)
                elif len(feature.shape) > 2:
                    feature = feature.reshape(feature.shape[0], -1)
                return {
                    'mean': np.mean(feature, axis=1).astype(np.float32),
                    'std': np.std(feature, axis=1).astype(np.float32),
                    'max': np.max(feature, axis=1).astype(np.float32),
                    'min': np.min(feature, axis=1).astype(np.float32),
                    'median': np.median(feature, axis=1).astype(np.float32)
                }

            return {
                'mfcc': get_stats(mfcc),
                'mfcc_delta': get_stats(mfcc_delta),
                'chroma': get_stats(chroma),
                'mel': get_stats(mel_spec),
                'onset_env': onset_env.astype(np.float32),
                'tempo': float(tempo)
            }
        except Exception as e:
            logger.warning(f"特徵提取失敗: {str(e)}")
            return None

    combined_features = None
    with ThreadPoolExecutor(max_workers=min(len(segments), 4)) as executor:
        futures = {executor.submit(extract_segment_features, seg): i for i, seg in enumerate(segments)}
        for future in as_completed(futures):
            f = future.result()
            if f:
                combined_features = combine_features([combined_features, f]) if combined_features else f
            gc.collect()

    return combined_features

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
        if features1 is features2:
            return 1.0

        def safe_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
            a = a.astype(np.float32)
            b = b.astype(np.float32)
            min_len = min(a.size, b.size)
            if min_len == 0:
                return 0.0
            a = a.flatten()[:min_len]
            b = b.flatten()[:min_len]
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            return (sim + 1) / 2  # normalize to [0, 1]

        def safe_dtw(a: np.ndarray, b: np.ndarray, max_len=500) -> float:
            a = a[:max_len]
            b = b[:max_len]
            if len(a) == 0 or len(b) == 0:
                return 0.0
            dist = dtw(a, b)[0][-1, -1]
            return 1.0 / (1.0 + dist / len(a))

        similarities = []
        weights = []

        # Time-series DTW similarities
        if 'onset_env' in features1 and 'onset_env' in features2:
            onset_sim = safe_dtw(features1['onset_env'], features2['onset_env'])
            similarities.append(onset_sim)
            weights.append(SIMILARITY_WEIGHTS['onset'])

        if 'mfcc' in features1 and 'mfcc' in features2:
            mfcc_sim = safe_dtw(features1['mfcc']['mean'], features2['mfcc']['mean'])
            similarities.append(mfcc_sim)
            weights.append(SIMILARITY_WEIGHTS['mfcc'])

        # Statistical features cosine similarities
        for feature_name in ['mfcc', 'mfcc_delta', 'chroma']:
            if feature_name in features1 and feature_name in features2:
                for stat in ['mean', 'std']:
                    vec1 = features1[feature_name].get(stat)
                    vec2 = features2[feature_name].get(stat)
                    if vec1 is not None and vec2 is not None:
                        sim = safe_cosine_similarity(vec1, vec2)
                        similarities.append(sim)
                        weights.append(SIMILARITY_WEIGHTS.get(feature_name, 1.0))

        # Tempo difference similarity
        if 'tempo' in features1 and 'tempo' in features2:
            tempo_diff = abs(features1['tempo'] - features2['tempo'])
            tempo_sim = 1.0 / (1.0 + tempo_diff / 30.0)
            similarities.append(tempo_sim)
            weights.append(SIMILARITY_WEIGHTS['tempo'])

        # Deep learning features
        for deep_feat_name in ['dl_features', 'pann_features', 'openl3_features']:
            if deep_feat_name in features1 and deep_feat_name in features2:
                sim = safe_cosine_similarity(features1[deep_feat_name], features2[deep_feat_name])
                similarities.append(sim)
                weights.append(SIMILARITY_WEIGHTS.get(deep_feat_name, 1.0))

        if not similarities:
            return 0.0

        weighted_sum = sum(s * w for s, w in zip(similarities, weights))
        total_weight = sum(weights)
        return float(max(0.0, min(1.0, weighted_sum / total_weight)))

    except Exception as e:
        logger.error(f"計算加權相似度時出錯: {str(e)}")
        return 0.0

def combine_features(features: list) -> dict:
    """合併多個音頻塊的特徵"""
    if not features:
        return None
    
    combined = {}
    for key in features[0].keys():
        if key == 'onset_env':
            combined[key] = np.concatenate([f[key] for f in features])
        elif key == 'tempo':
            combined[key] = float(np.mean([f[key] for f in features]))
        else:
            combined[key] = {
                stat: np.mean([f[key][stat] for f in features], axis=0)
                for stat in features[0][key].keys()
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
            raise ValueError(f"裁減時間過短，最小需要 {CROP_CONFIG['min_duration']} 秒")
        if duration > CROP_CONFIG['max_duration']:
            raise ValueError(f"裁減時間過長，最大允許 {CROP_CONFIG['max_duration']} 秒")
            
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

def extract_all_deep_features(split_paths: list[str], use_openl3: bool = True) -> dict:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_segment(path: str) -> dict:
        result = {}
        try:
            logger.info(f"🧩 處理段: {os.path.basename(path)}")

            log_memory_usage("DL 前")
            dl = extract_audio_features_chunked_streaming(path)
            if dl is not None:
                result['dl'] = dl
            log_memory_usage("DL 後")

            log_memory_usage("PANN 前")
            pann = extract_pann_features_chunked(path)
            if pann is not None:
                result['pann'] = pann
            log_memory_usage("PANN 後")

            if use_openl3:
                log_memory_usage("OpenL3 前")
                l3 = extract_openl3_features_chunked(path)
                if l3 is not None:
                    result['openl3'] = l3
                log_memory_usage("OpenL3 後")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.error(f"⚠️ 處理段 {path} 時出錯: {str(e)}")
        return result

    results = []
    with ThreadPoolExecutor(max_workers=min(len(split_paths), 5)) as executor:
        futures = {executor.submit(process_segment, path): path for path in split_paths}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # 聚合各段落特徵
    merged = {}
    if any('dl' in r for r in results):
        dl_stack = [r['dl'] for r in results if 'dl' in r]
        merged['dl_features'] = np.mean(np.stack(dl_stack), axis=0).astype(np.float32)
    if any('pann' in r for r in results):
        pann_stack = [r['pann'] for r in results if 'pann' in r]
        merged['pann_features'] = np.mean(np.stack(pann_stack), axis=0).astype(np.float32)
    if use_openl3 and any('openl3' in r for r in results):
        l3_stack = [r['openl3'] for r in results if 'openl3' in r]
        merged['openl3_features'] = np.mean(np.stack(l3_stack), axis=0).astype(np.float32)

    return merged if merged else None

def compute_audio_features(audio_path: str, start_time: Optional[float] = None, end_time: Optional[float] = None) -> Optional[dict]:
    """整合所有特徵提取過程的主函數"""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        log_memory_usage("切割前")

        # 如果影片過長，先切成多段處理
        split_paths = split_audio_by_duration(audio_path, split_sec=300.0)  # 每段 5 分鐘
        if not split_paths:
            logger.error("音訊切段失敗")
            return None
        
        log_memory_usage("處理特徵中")

        cache_dir = os.path.join(os.path.dirname(audio_path), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        file_id = hashlib.md5(audio_path.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{file_id}_features.npz")
        
        # 檢查是否為完整快取（含 deep features）
        if os.path.exists(cache_path):
            cached = load_cached_features(cache_path)
            if all(k in cached for k in ['dl_features', 'pann_features', 'openl3_features']):
                logger.info("快取完整，使用快取")
                return cached
            else:
                logger.warning("快取不完整，將重新計算特徵")
        
        def process_stat_features(path: str) -> Optional[dict]:
            try:
                y, sr = librosa.load(path, sr=AUDIO_CONFIG['sample_rate'])
                return parallel_feature_extraction(y, sr)
            except Exception as e:
                logger.warning(f"統計特徵處理失敗 {path}: {e}")
                return None

        final_features = None
        with ThreadPoolExecutor(max_workers=min(len(split_paths), 5)) as executor:
            futures = [executor.submit(process_stat_features, path) for path in split_paths]
            for future in as_completed(futures):
                f = future.result()
                if f:
                    final_features = combine_features([final_features, f]) if final_features else f

        #log_memory_usage(f"段落 {path} 處理完畢")

        # 2. 深度特徵提取（整體只執行一次）
        log_memory_usage("深度特徵處理前")
        deep_features = extract_all_deep_features(split_paths)
        if deep_features is not None:
            final_features.update(deep_features)
        log_memory_usage("深度特徵處理後")
        
        # 緩存最終特徵
        cache_features_to_disk(final_features, cache_dir, file_id)
        return final_features
        
    except Exception as e:
        logger.error(f"提取音頻特徵時出錯: {str(e)}")
        return None

def audio_similarity(path1: str, path2: str, start_time1: Optional[float] = None, end_time1: Optional[float] = None, 
                    start_time2: Optional[float] = None, end_time2: Optional[float] = None) -> float:
    """
    計算兩段音頻的相似度
    """
    try:
        log_memory_usage("音訊1 開始前")
        features1 = compute_audio_features(path1, start_time1, end_time1)
        log_memory_usage("音訊1 完成")

        if features1 is None:
            logger.error("無法提取第一段音頻特徵")
            return 0.0

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log_memory_usage("音訊2 開始前")
        features2 = compute_audio_features(path2, start_time2, end_time2)
        log_memory_usage("音訊2 完成")

        if features2 is None:
            logger.error("無法提取第二段音頻特徵")
            return 0.0

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