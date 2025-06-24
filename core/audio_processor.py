import os
import gc
import time
import torch
import psutil
import ffmpeg
import librosa
import torchaudio
import torchopenl3
import numpy as np
from hashlib import sha1
from filelock import FileLock
from functools import lru_cache
from utils.logger import logger
from librosa.sequence import dtw
from sklearn.decomposition import PCA
from utils.gpu_utils import gpu_manager
from librosa.feature.rhythm import tempo
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Generator, Tuple, Dict

# =============== 全局配置参数 ===============
AUDIO_CONFIG = {
    'sample_rate': 32000,
    'channels': 1,
    'audio_bitrate': '192k',
    'format': 'wav',
    'codec': 'pcm_s16le',
    'force_gpu': True
}

FEATURE_CONFIG = {
    'mfcc': {'n_mfcc': 13, 'hop_length': 1024},
    'mel': {'n_mels': 64, 'hop_length': 1024},
    'chroma': {'n_chroma': 12, 'hop_length': 1024}
}

SIMILARITY_WEIGHTS = {
    'dl_features': 1.5,       # 比對深度學習模型提取的高層特徵，包含音色、音質等複雜特徵
    'pann_features': 1.2,     # 比對音頻場景分類特徵，用於識別環境音和背景音
    'openl3_features': 1.7,   # 比對音頻嵌入向量，捕捉音頻的語義信息
    'openl3_chunkwise': 0.5,  # chunkwise DTW 結構相似度
    'onset': 1.0,             # 比對音頻的起始點和節奏變化點
    'mfcc': 1.5,              # 比對梅爾頻率倒譜係數，主要用於音色和音質比對
    'mfcc_delta': 1.5,        # 比對 MFCC 的動態變化，反映音頻的時變特性
    'chroma': 1.5,            # 比對音高分布特徵，用於和聲和調性分析
    'tempo': 1.4              # 比對節奏速度特徵，反映音樂的節奏特性
}

THREAD_CONFIG = {'max_workers': 6}
CROP_CONFIG = {'min_duration': 30.0, 'max_duration': 300.0, 'overlap': 0.5, 'silence_threshold': -14}

# =============== 初始化模型與資源 ===============
FEATURE_CACHE_DIR = os.path.join(os.getcwd(), "feature_cache")
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

gpu_manager.initialize()
device = gpu_manager.get_device()

pann_model: Optional[torch.nn.Module] = None
openl3_model: Optional[torch.nn.Module] = None

@lru_cache(maxsize=3)
def get_mel_transform(sr: int):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=FEATURE_CONFIG['mel']['hop_length'],
        n_mels=FEATURE_CONFIG['mel']['n_mels']
    ).to(device)

@lru_cache(maxsize=1)
def get_openl3_model():
    global openl3_model
    if openl3_model is None:
        openl3_model = torchopenl3.models.load_audio_embedding_model("mel128", "music", 512)
        openl3_model = openl3_model.to(device)
        if device.type == 'cuda':
            openl3_model = torch.nn.DataParallel(openl3_model)
        openl3_model.eval()
    return openl3_model

@lru_cache(maxsize=1)
def get_pann_model():
    global pann_model
    if pann_model is None:
        from panns_inference.models import Cnn14
        pann_model = Cnn14(
            sample_rate=AUDIO_CONFIG['sample_rate'], window_size=1024, hop_size=320,
            mel_bins=64, fmin=50, fmax=14000, classes_num=527
        ).to(device)
        pann_model.eval()
    return pann_model

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

# =============== 基本工具函數 ===============
def log_memory(stage: str):
    print(f"[{stage}] Memory: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")

def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    waveform = waveform - waveform.mean()
    waveform = waveform / waveform.abs().max().clamp(min=1e-6)
    return waveform / (waveform.norm(p=2) + 1e-9)

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
    
def is_valid_vector(v):
    """檢查是否為合法的向量（非空、非物件型態、一維）"""
    return isinstance(v, np.ndarray) and v.dtype != object and v.size > 0 and v.ndim == 1

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

def get_cache_path(audio_path: str) -> str:
    """根據音訊路徑產生快取檔案路徑（使用 SHA1）"""
    basename = os.path.basename(audio_path)
    hash_id = sha1(audio_path.encode('utf-8')).hexdigest()
    return os.path.join(FEATURE_CACHE_DIR, f"{basename}_{hash_id[:10]}.npz")

def save_audio_features_to_cache(audio_path: str, features: Dict[str, any]):
    cache_path = get_cache_path(audio_path)
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):
        if os.path.exists(cache_path):
            logger.info(f"⚠️ 快取已存在，跳過儲存: {cache_path}")
            return
        try:
            np.savez_compressed(cache_path, **features)
            logger.info(f"✅ 特徵快取儲存成功: {cache_path}")
        except Exception as e:
            logger.warning(f"⚠️ 儲存特徵快取失敗: {e}")

def load_audio_features_from_cache(audio_path: str) -> Optional[Dict[str, any]]:
    try:
        cache_path = get_cache_path(audio_path)
        if not os.path.exists(cache_path):
            return None
        data = np.load(cache_path, allow_pickle=True)
        loaded = {k: data[k].item() if data[k].shape == () else data[k] for k in data}
        logger.info(f"📦 載入特徵快取成功: {cache_path}")
        return loaded
    except Exception as e:
        logger.warning(f"⚠️ 載入特徵快取失敗: {e}")
        return None

def perceptual_score(sim_score: float) -> float:
    """高相似度 → gamma 趨近 1.2；低分拉大差異"""
    gamma = 1.2 + 2.0 * (1 - sim_score)
    return min(max(sim_score ** gamma, 0.0), 1.0)

# =============== 特徵擷取與比較 ===============
def extract_dl_features(audio_path: str, chunk_sec=10.0) -> Optional[np.ndarray]:
    try:
        y, sr = librosa.load(audio_path, sr=AUDIO_CONFIG['sample_rate'])
        mel_spec_transform = get_mel_transform(sr)
        chunk_size = int(chunk_sec * sr)
        features = []
        for i in range(0, len(y), chunk_size):
            chunk = y[i:i+chunk_size]
            if len(chunk) < sr: continue
            try:
                waveform = normalize_waveform(torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device))
                with torch.no_grad():
                    mel = mel_spec_transform(waveform)
                    pooled = torch.mean(mel, dim=2).squeeze().cpu().numpy()
                features.append(pooled)
            except Exception as e:
                print(f"⚠️ DL chunk {i//chunk_size} failed: {e}")
                continue
        if not features:
            logger.warning(f"DL 特徵全部失敗: {audio_path}")
        chunks = np.stack(features)
        return np.concatenate([
            np.mean(chunks, axis=0),
            np.var(chunks, axis=0)
        ]).astype(np.float32) if features else None
    except Exception as e:
        print(f"DL feature error: {e}")
        return None

def extract_pann_features(audio_path: str, chunk_sec=10.0) -> Optional[np.ndarray]:
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != AUDIO_CONFIG['sample_rate']:
            waveform = torchaudio.transforms.Resample(sr, AUDIO_CONFIG['sample_rate'])(waveform)
            sr = AUDIO_CONFIG['sample_rate']
        model = get_pann_model()
        chunk_size = int(chunk_sec * sr)
        chunks = [waveform[:, i:i+chunk_size] for i in range(0, waveform.shape[1], chunk_size) if waveform[:, i:i+chunk_size].shape[1] >= sr]
        emb_list = []
        tag_vec_sum = None
        for c in chunks:
            with torch.no_grad():
                out = model(c.to(device))
                emb = out['embedding'].squeeze().cpu().numpy()
                tags = out['clipwise_output'].squeeze().cpu().numpy()
                emb_list.append(emb[:2048])
                tag_vec_sum = tags if tag_vec_sum is None else tag_vec_sum + tags
        if not emb_list:
            return None
        emb_mean = np.mean(np.stack(emb_list), axis=0)
        tag_vec_mean = tag_vec_sum / len(emb_list)
        return np.concatenate([emb_mean, tag_vec_mean]).astype(np.float32)
    except Exception as e:
        print(f"PANN error: {e}")
        return None

def extract_openl3_features(audio_path: str, chunk_sec=10.0) -> Optional[dict]:
    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=False)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)
        if audio.shape[0] > 1:
            audio = np.mean(audio, axis=0, keepdims=True)
        if sr != 48000:
            audio = librosa.resample(audio[0], orig_sr=sr, target_sr=48000)
            audio = np.expand_dims(audio, axis=0)
            sr = 48000
        audio = audio.astype(np.float32)
        chunk_size = int(chunk_sec * sr)
        model = get_openl3_model()
        if np.max(np.abs(audio)) < 1e-5 or audio.shape[1] < sr:
            logger.warning(f"⚠️ 音訊近乎靜音或長度不足，跳過：{audio_path}")
            return None
        emb_list = []
        for i in range(0, audio.shape[1], chunk_size):
            chunk = audio[:, i:i + chunk_size]
            if chunk.shape[1] < 4800:
                continue
            try:
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32).to(device)
                with torch.no_grad():
                    emb, _ = torchopenl3.get_audio_embedding(
                        chunk_tensor, sr,
                        model=model,
                        hop_size=1.0,
                        center=True,
                        verbose=False
                    )
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy()
                if emb.ndim == 3 and emb.shape[2] == 512:
                    emb = emb[0]
                if emb.ndim == 2 and emb.shape[1] == 512:
                    emb_list.append(np.mean(emb, axis=0))
                else:
                    logger.warning(f"⚠️ OpenL3 特徵 shape 不一致：{emb.shape}")
            except Exception as sub_e:
                logger.warning(f"❌ OpenL3 子段錯誤: {sub_e}")
                continue
        if not emb_list:
            logger.warning(f"❌ OpenL3 全部段落提取失敗：{audio_path}")
            return None
        emb_array = np.stack(emb_list)
        gpu_manager.clear_gpu_memory()
        return {
            "merged": np.concatenate([
                np.mean(emb_array, axis=0),
                np.var(emb_array, axis=0)
            ]).astype(np.float32),
            "chunkwise": emb_array.astype(np.float32)
        }
    except Exception as e:
        logger.warning(f"OpenL3 error: {e}")
        return None

def extract_statistical_features(audio_path: str, chunk_sec=10.0) -> Optional[dict]:
    try:
        y, sr = librosa.load(audio_path, sr=AUDIO_CONFIG['sample_rate'])
        chunk_size = int(chunk_sec * sr)
        features = []
        for i in range(0, len(y), chunk_size):
            seg = y[i:i + chunk_size]
            if len(seg) < sr: continue
            mel = librosa.feature.melspectrogram(y=seg, sr=sr, **FEATURE_CONFIG['mel'])
            mfcc = librosa.feature.mfcc(y=seg, sr=sr, **FEATURE_CONFIG['mfcc'])
            delta = librosa.feature.delta(mfcc)
            chroma = librosa.feature.chroma_stft(y=seg, sr=sr, **FEATURE_CONFIG['chroma'])
            onset = librosa.onset.onset_strength(y=seg, sr=sr)
            tempos = tempo(onset_envelope=onset, sr=sr, aggregate=None)
            tempo_mean = float(np.mean(tempos)) if len(tempos) else 0.0
            tempo_std = float(np.std(tempos)) if len(tempos) else 0.0
            tempo_range = float(np.max(tempos) - np.min(tempos)) if len(tempos) else 0.0

            def stats(x): return {
                'mean': np.mean(x, axis=1).astype(np.float32),
                'std': np.std(x, axis=1).astype(np.float32),
                'max': np.max(x, axis=1).astype(np.float32),
                'min': np.min(x, axis=1).astype(np.float32),
                'median': np.median(x, axis=1).astype(np.float32)
            }

            features.append({
                'mfcc': stats(mfcc),
                'mfcc_delta': stats(delta),
                'chroma': stats(chroma),
                'mel': stats(mel),
                'onset_env': onset.astype(np.float32),
                'tempo': {
                    'mean': tempo_mean,
                    'std': tempo_std,
                    'range': tempo_range
                }
            })

        return combine_features(features)
    except Exception as e:
        print(f"Stat feature error: {e}")
        return None

def combine_features(features: list) -> dict:
    if not features: return None
    combined = {}
    for key in features[0]:
        if key == 'onset_env':
            combined[key] = np.concatenate([f[key] for f in features])
        elif key == 'tempo':
            if isinstance(features[0][key], dict):
                combined[key] = {
                    subkey: float(np.mean([f[key][subkey] for f in features])) 
                    for subkey in features[0][key]
                }
            else:
                combined[key] = float(np.mean([f[key] for f in features]))
        else:
            combined[key] = {
                stat: np.mean([f[key][stat] for f in features], axis=0)
                for stat in features[0][key]
            }
    return combined

def chunkwise_dtw_sim(chunk1: np.ndarray, chunk2: np.ndarray, n_components=32) -> float:
    if chunk1.shape[0] < 2 or chunk2.shape[0] < 2:
        return 0.0
    combined = np.vstack([chunk1, chunk2])
    max_components = min(combined.shape[0], combined.shape[1], n_components)
    if max_components < 2:
        return 0.0  # 避免降維太少無法比較
    pca = PCA(n_components=max_components)
    reduced = pca.fit_transform(combined)
    r1 = reduced[:chunk1.shape[0]]
    r2 = reduced[chunk1.shape[0]:]
    cost = librosa.sequence.dtw(X=r1.T, Y=r2.T, metric='euclidean')[0]
    dtw_dist = cost[-1, -1]
    return 1 / (1 + dtw_dist / len(r1))

# =============== 主流程 ===============
def compute_audio_features(audio_path: str, use_openl3=True) -> Optional[dict]:
    # 先試圖從快取讀取
    cached = load_audio_features_from_cache(audio_path)
    if cached:
        return cached

    stat_feat, deep_feat = {}, {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_stat = pool.submit(extract_statistical_features, audio_path)
        f_deep = pool.submit(lambda: {
            'dl_features': extract_dl_features(audio_path),
            'pann_features': extract_pann_features(audio_path),
            'openl3_features': extract_openl3_features(audio_path) if use_openl3 else None
        })
        stat_feat = f_stat.result()
        deep_feat = f_deep.result()

    if not stat_feat:
        logger.warning(f"Statistical 特徵提取失敗: {audio_path}")
        return None

    for k in ['dl_features', 'pann_features', 'openl3_features']:
        if deep_feat.get(k) is None:
            logger.warning(f"Deep 特徵缺失：{k} 在 {audio_path}")

    features = {**stat_feat, **{k: v for k, v in deep_feat.items() if v is not None}}
    save_audio_features_to_cache(audio_path, features)
    return features

def compute_similarity(f1: dict, f2: dict) -> float:
    detailed_scores = {}
    def cos_sim(a, b): return (np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8) + 1) / 2
    def dtw_sim(a, b): return 1 / (1 + dtw(a[:500], b[:500])[0][-1, -1] / 500)

    scores, weights = [], []

    if 'onset_env' in f1 and 'onset_env' in f2:
        dtw_score = dtw_sim(f1['onset_env'], f2['onset_env'])
        weight = SIMILARITY_WEIGHTS.get('onset', 1.0)
        scores.append(dtw_score)
        weights.append(weight)
        detailed_scores['onset_env'] = (dtw_score, weight)

    for k in ['mfcc', 'mfcc_delta', 'chroma']:
        for stat in ['mean', 'std']:
            if k in f1 and k in f2 and stat in f1[k] and stat in f2[k]:
                score = cos_sim(f1[k][stat], f2[k][stat])
                scores.append(score)
                weights.append(SIMILARITY_WEIGHTS.get(k, 1.0))
                detailed_scores[f"{k}_{stat}"] = (score, SIMILARITY_WEIGHTS.get(k, 1.0))

    if 'tempo' in f1 and 'tempo' in f2:
        t1 = f1['tempo']
        t2 = f2['tempo']
        tempo_sim = 1 / (1 + abs(t1['mean'] - t2['mean']) / 30)
        std_sim = 1 / (1 + abs(t1['std'] - t2['std']) / 15)
        range_sim = 1 / (1 + abs(t1['range'] - t2['range']) / 30)
        combined_tempo_sim = 0.5 * tempo_sim + 0.25 * std_sim + 0.25 * range_sim
        scores.append(combined_tempo_sim)
        weights.append(SIMILARITY_WEIGHTS.get('tempo', 1.0))

    # 🔹 加入 OpenL3 chunkwise DTW 比對（若有）
    if 'openl3_features' in f1 and 'openl3_features' in f2:
        o1 = f1['openl3_features']
        o2 = f2['openl3_features']
        if isinstance(o1, dict) and 'chunkwise' in o1 and 'chunkwise' in o2:
            sim = chunkwise_dtw_sim(o1['chunkwise'], o2['chunkwise'])
            scores.append(sim)
            weights.append(SIMILARITY_WEIGHTS.get('openl3_features', 1.0))
            detailed_scores['openl3_chunkwise'] = (sim, SIMILARITY_WEIGHTS.get('openl3_chunkwise', 1.0))
            logger.info(f'✅ OpenL3 chunkwise DTW 相似度: {sim:.4f}')

    for k in ['dl_features', 'pann_features', 'openl3_features']:
        v1 = f1.get(k)
        v2 = f2.get(k)

        logger.info(f"🧪 比對特徵 {k}:")
        logger.info(f"  v1 type: {type(v1)}, shape: {getattr(v1, 'shape', None) if not isinstance(v1, dict) else 'dict'}")
        logger.info(f"  v2 type: {type(v2)}, shape: {getattr(v2, 'shape', None) if not isinstance(v2, dict) else 'dict'}")

        if v1 is None or v2 is None:
            logger.warning(f"❌ 特徵 {k} 為 None，跳過")
            continue

        try:
            if k == 'pann_features' and isinstance(v1, np.ndarray) and v1.size == v2.size:
                split = 2048
                emb1, tag1 = v1[:split], np.round(v1[split:])
                emb2, tag2 = v2[:split], np.round(v2[split:])
                sim1 = cos_sim(emb1, emb2)
                top1 = set(tag1.argsort()[-5:])
                top2 = set(tag2.argsort()[-5:])
                jaccard = len(top1 & top2) / len(top1 | top2) if top1 | top2 else 0.0
                sim = 0.7 * sim1 + 0.3 * jaccard

            elif k == 'openl3_features' and isinstance(v1, dict) and isinstance(v2, dict):
                if 'merged' in v1 and 'merged' in v2:
                    sim = cos_sim(v1['merged'], v2['merged'])
                else:
                    logger.warning(f"❌ openl3_features 缺少 merged 向量，跳過")
                    continue
            else:
                if isinstance(v1, np.ndarray) and v1.ndim == 2:
                    v1 = np.mean(v1, axis=0)
                if isinstance(v2, np.ndarray) and v2.ndim == 2:
                    v2 = np.mean(v2, axis=0)
                if not is_valid_vector(v1) or not is_valid_vector(v2):
                    logger.warning(f"❌ 特徵格式不合法: {k}")
                    continue
                sim = cos_sim(v1, v2)

            weight = SIMILARITY_WEIGHTS.get(k, 1.0)
            scores.append(sim)
            weights.append(weight)
            detailed_scores[k] = (sim, weight)
            logger.info(f"✅ 成功比對 {k}，相似度: {sim:.4f}，權重: {weight}")

        except Exception as e:
            logger.error(f"❌ 比對 {k} 發生錯誤: {e}")

    if not scores:
        logger.error("所有相似度評估皆失敗，無法進行加權")
        return 0.0

    if len(scores) != len(weights):
        logger.error(f"❗相似度與權重長度不一致: scores={len(scores)}, weights={len(weights)}")
        logger.error(f"scores={scores}")
        logger.error(f"weights={weights}")
        return 0.0

    final_score = float(np.average(scores, weights=weights))
    logger.info("📊 特徵比對詳情：")
    for name, (score, weight) in detailed_scores.items():
        logger.info(f"  🔹 {name:20s} | 相似度: {score:.4f} | 權重: {weight}")
    return final_score

def audio_similarity(path1: str, path2: str) -> float:
    log_memory("開始")
    f1 = compute_audio_features(path1)
    f2 = compute_audio_features(path2)
    log_memory("完成")
    print(f"✅ f1 keys: {list(f1.keys())}")
    print(f"✅ f2 keys: {list(f2.keys())}")
    if not f1 or not f2:
        logger.error("音頻特徵提取失敗，無法比較")
        logger.error(f"音頻文件1: {path1}")
        logger.error(f"音頻文件2: {path2}")
        return 0.0
    
    raw_audio_similarity = compute_similarity(f1, f2)
    adjusted_audio_similarity = perceptual_score(raw_audio_similarity)

    logger.info(f"音頻原始相似度: {raw_audio_similarity:.4f} ➜ 感知相似度: {adjusted_audio_similarity:.4f}")
    return adjusted_audio_similarity