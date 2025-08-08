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
from collections import OrderedDict
from sklearn.decomposition import PCA
from utils.gpu_utils import gpu_manager
from panns_inference.models import Cnn14
from librosa.feature.rhythm import tempo
from utils.downloader import ensure_pann_weights
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, Generator, Tuple, Dict, Any, List

# =============== 全局配置参数 ===============
_pann_model_loaded = False

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
    # 深度學習模型特徵
    'dl_features': 2.2,             # 利用 MelSpectrogram 統計表示整體音色輪廓（類似人耳感受）
    'pann_features': 2.2,           # 結合音訊分類模型（如環境聲音、場景、音效標籤）的特徵與分類輸出
    'openl3_features': 1.8,         # 音訊語義嵌入（整體語意、音質風格）特徵
    'openl3_chunkwise': 0.2,        # OpenL3 分段特徵的結構性比對（用 DTW 抓時間軸變化一致性）

    # 傳統統計特徵（總體設定，當細項缺失時使用）
    'mfcc': 1.2,                    # 音色輪廓（主頻能量分布），模擬聽感頻率響應
    'mfcc_delta': 1.0,              # MFCC 變化量，模擬語音/音樂的滑音、變化性
    'chroma': 1.4,                  # 音高與和聲（十二平均律對應的能量）

    # 統計細項（比對用於細節控制）
    'mfcc_mean': 1.3,               # 平均 MFCC：整體音色分布（傾向於穩定的 timbre）
    'mfcc_std': 1.0,                # MFCC 標準差：音色的起伏與變化性
    'mfcc_delta_mean': 0.8,         # 動態 MFCC 平均：整體變化趨勢（語速、連續性）
    'mfcc_delta_std': 1.2,          # 動態 MFCC 標準差：滑音、抖音、情緒起伏
    'chroma_mean': 1.0,             # 平均音高能量：主要音調特徵（旋律主色調）
    'chroma_std': 1.0,              # 音高能量變化：轉調、和聲複雜度等

    # 節奏與結構
    'onset_env': 1.4,               # 音訊起始點強度變化（打擊點、節奏感）
    'tempo': 1.3,                   # 節奏速度（BPM）與變化程度，用來比對歌曲速度與律動
}

THREAD_CONFIG = {'max_workers': 6}
CROP_CONFIG = {'min_duration': 30.0, 'max_duration': 300.0,
               'overlap': 0.5, 'silence_threshold': -14}

# =============== 初始化模型與資源 ===============
FEATURE_CACHE_DIR = os.path.join(os.getcwd(), "feature_cache")
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

gpu_manager.initialize()
device = gpu_manager.get_device()

pann_model: Optional[torch.nn.Module] = None
openl3_model: Optional[torch.nn.Module] = None

# PCA 快取類別


class PCACache:
    def __init__(self, max_items: int = 15) -> None:
        self.cache: OrderedDict[str, PCA] = OrderedDict()
        self.max_items = max_items

    def get(self, name: str) -> Optional[PCA]:
        return self.cache.get(name)

    def set(self, name: str, pca: PCA) -> None:
        self.cache[name] = pca
        self.cache.move_to_end(name)
        if len(self.cache) > self.max_items:
            evicted = self.cache.popitem(last=False)
            logger.info(f"自動清除 PCA: {evicted[0]}")

    def clear(self) -> None:
        self.cache.clear()


# 正式使用 PCACache
_pca_registry = PCACache()


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
        openl3_model = torchopenl3.models.load_audio_embedding_model(
            "mel128", "music", 512)
        openl3_model = openl3_model.to(device)
        if device.type == 'cuda':
            openl3_model = torch.nn.DataParallel(openl3_model)
        openl3_model.eval()
    return openl3_model


@lru_cache(maxsize=1)
def get_pann_model():
    global _pann_model_loaded
    checkpoint_path = ensure_pann_weights()
    model = Cnn14(
        sample_rate=AUDIO_CONFIG['sample_rate'], window_size=1024,
        hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if not _pann_model_loaded:
        logger.info("PANN 模型權重載入成功")
        _pann_model_loaded = True
    model = model.to(device)
    model.eval()
    return model


@lru_cache(maxsize=32)
def get_optimal_chunk_size(file_size: int) -> float:
    base = 30.0
    if file_size > 1 << 30:
        return 15.0
    if file_size > 512 << 20:
        return 30.0
    return 60.0

# =============== 基本工具函數 ===============


def chamfer_sim(a: np.ndarray, b: np.ndarray, top_k: int = 3) -> float:
    """
    a, b: shape = (n_chunk, dim)
    回傳 0~1，相似度越高越接近 1
    """
    # 兩兩餘弦相似度矩陣
    S = cosine_similarity(a, b)

    # 每個 chunk 取「對方前 top_k」平均，再雙向平均
    topA = np.mean(np.sort(S, axis=1)[:, -top_k:], axis=1)   # A→B
    topB = np.mean(np.sort(S, axis=0)[-top_k:, :], axis=0)   # B→A
    return float((topA.mean() + topB.mean()) / 2)


def log_memory(stage: str) -> None:
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
            hop_length=1024
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


def is_valid_vector(v: Any) -> bool:
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


def save_audio_features_to_cache(audio_path: str, features: Dict[str, any]) -> None:
    cache_path = get_cache_path(audio_path)
    lock_path = cache_path + ".lock"

    # 先檢查檔案是否已存在（避免不必要的鎖定）
    if os.path.exists(cache_path):
        logger.debug(f"快取已存在，跳過儲存: {cache_path}")
        return

    with FileLock(lock_path):
        # 再次檢查（防止在等待鎖定期間其他執行緒已建立檔案）
        if os.path.exists(cache_path):
            logger.debug(f"快取已存在，跳過儲存: {cache_path}")
            return
        try:
            np.savez_compressed(cache_path, **features)
            logger.info(f"特徵快取儲存成功: {cache_path}")
        except Exception as e:
            logger.warning(f"儲存特徵快取失敗: {e}")


def load_audio_features_from_cache(audio_path: str) -> Optional[Dict[str, any]]:
    try:
        cache_path = get_cache_path(audio_path)
        if not os.path.exists(cache_path):
            return None
        data = np.load(cache_path, allow_pickle=True)
        loaded = {k: data[k].item() if data[k].shape == ()
                  else data[k] for k in data}
        logger.info(f"載入特徵快取成功: {cache_path}")
        return loaded
    except Exception as e:
        logger.warning(f"載入特徵快取失敗: {e}")
        return None


def perceptual_score(sim_score: float) -> float:
    """高相似度 → gamma 趨近 1.2；低分拉大差異"""
    gamma = 1.2 + 1.0 * (1 - sim_score)
    return min(max(sim_score ** gamma, 0.0), 1.0)


def fit_pca_if_needed(name: str, data: np.ndarray, n_components: int) -> Optional[PCA]:
    if _pca_registry.get(name):
        return _pca_registry.get(name)
    n_samples, dim = data.shape
    n_components = min(n_components, n_samples, dim)
    if n_components < 2:
        return None
    pca = PCA(n_components=n_components)
    pca.fit(data)
    _pca_registry.set(name, pca)
    return pca


def apply_pca(name: str, vector: np.ndarray, n_components: int) -> np.ndarray:
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    pca = _pca_registry.get(name)
    if pca is None:
        return vector.squeeze()
    reduced = pca.transform(vector)
    return reduced.squeeze()


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """計算餘弦相似度，若長度不一致，取最小長度進行對齊再計算"""
    # 對齊長度
    if a.ndim == 1 and b.ndim == 1:
        min_len = min(a.shape[0], b.shape[0])
        a = a[:min_len]
        b = b[:min_len]
    # 計算並映射到 [0,1]
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return (cos + 1) / 2


def dtw_sim(a: np.ndarray, b: np.ndarray, max_length: int = 500) -> float:
    """簡化的 DTW 相似度，較短樣本對齊比較用"""
    a = a[:max_length]
    b = b[:max_length]
    cost = librosa.sequence.dtw(X=a.reshape(
        1, -1), Y=b.reshape(1, -1), metric='euclidean')[0]
    return 1 / (1 + cost[-1, -1] / len(a))

# =============== 特徵擷取與比較 ===============


def extract_dl_features(audio_path: str, chunk_sec: float = 10.0) -> Optional[np.ndarray]:
    try:
        y, sr = librosa.load(audio_path, sr=AUDIO_CONFIG['sample_rate'])
        mel_spec_transform = get_mel_transform(sr)
        chunk_size = int(chunk_sec * sr)
        feats = []

        for i in range(0, len(y), chunk_size):
            chunk = y[i:i + chunk_size]
            if len(chunk) < sr:
                continue
            try:
                wf = normalize_waveform(torch.tensor(chunk, dtype=torch.float32)
                                        .unsqueeze(0).to(device))
                with torch.no_grad():
                    # (1, n_mels, n_frame)
                    mel = mel_spec_transform(wf)
                    pooled = torch.mean(mel, dim=2).squeeze().cpu().numpy()
                feats.append(pooled)                       # 128 維
            except Exception as e:
                logger.warning(f"[DL] chunk {i//chunk_size} failed: {e}")
                continue

        if not feats:
            logger.warning(f"[DL] 全部 chunk 失敗: {audio_path}")
            return None

        return np.stack(feats).astype(np.float32)          # (n_chunk, 128)
    except Exception as e:
        logger.error(f"[DL] feature error: {e}")
        return None


def extract_pann_features(audio_path: str, chunk_sec: float = 10.0) -> Optional[np.ndarray]:
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != AUDIO_CONFIG['sample_rate']:
            waveform = torchaudio.transforms.Resample(
                sr, AUDIO_CONFIG['sample_rate'])(waveform)
            sr = AUDIO_CONFIG['sample_rate']

        model = get_pann_model()
        if model is None:
            logger.error("[PANN] 模型初始化失敗")
            return None

        chunk_size = int(chunk_sec * sr)
        chunks = [waveform[:, i:i + chunk_size]
                  for i in range(0, waveform.shape[1], chunk_size)
                  if waveform[:, i:i + chunk_size].shape[1] >= sr]

        feats = []
        for idx, c in enumerate(chunks):
            try:
                with torch.no_grad():
                    out = model(c.to(device))
                    emb = out['embedding'].squeeze().cpu().numpy()[
                        :2048]      # 2048
                    tags = out['clipwise_output'].squeeze(
                    ).cpu().numpy()       # 527
                # 2575
                feats.append(np.concatenate([emb, tags]))
            except Exception as e:
                logger.warning(f"[PANN] chunk {idx} failed: {e}")

        if not feats:
            logger.warning(f"[PANN] 沒有有效 chunk: {audio_path}")
            return None

        return np.stack(feats).astype(np.float32)         # (n_chunk, 2575)
    except Exception as e:
        logger.error(f"[PANN] 特徵提取失敗: {e}")
        return None


def extract_openl3_features(audio_path: str, chunk_sec: float = 10.0) -> Optional[Dict[str, np.ndarray]]:
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
            logger.warning(f"音訊近乎靜音或長度不足，跳過：{audio_path}")
            return None
        emb_list = []
        for i in range(0, audio.shape[1], chunk_size):
            chunk = audio[:, i:i + chunk_size]
            if chunk.shape[1] < 4800:
                continue
            try:
                chunk_tensor = torch.tensor(
                    chunk, dtype=torch.float32).to(device)
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
                    logger.warning(f"OpenL3 特徵 shape 不一致：{emb.shape}")
            except Exception as sub_e:
                logger.warning(f"OpenL3 子段錯誤: {sub_e}")
                continue
        if not emb_list:
            logger.warning(f"OpenL3 全部段落提取失敗：{audio_path}")
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


def extract_statistical_features(audio_path: str, chunk_sec: float = 10.0) -> Optional[Dict[str, Any]]:
    try:
        y, sr = librosa.load(audio_path, sr=AUDIO_CONFIG['sample_rate'])
        chunk_size = int(chunk_sec * sr)
        features = []
        for i in range(0, len(y), chunk_size):
            seg = y[i:i + chunk_size]
            if len(seg) < sr:
                continue
            mel = librosa.feature.melspectrogram(
                y=seg, sr=sr, **FEATURE_CONFIG['mel'])
            mfcc = librosa.feature.mfcc(y=seg, sr=sr, **FEATURE_CONFIG['mfcc'])
            delta = librosa.feature.delta(mfcc)
            chroma = librosa.feature.chroma_stft(
                y=seg, sr=sr, **FEATURE_CONFIG['chroma'])
            onset = librosa.onset.onset_strength(y=seg, sr=sr)
            tempos = tempo(onset_envelope=onset, sr=sr, aggregate=None)
            tempo_mean = float(np.mean(tempos)) if len(tempos) else 0.0
            tempo_std = float(np.std(tempos)) if len(tempos) else 0.0
            tempo_range = float(
                np.max(tempos) - np.min(tempos)) if len(tempos) else 0.0

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


def combine_features(features: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not features:
        return None
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


def chunkwise_dtw_sim(chunk1: np.ndarray, chunk2: np.ndarray, n_components: int = 32) -> float:
    if chunk1.shape[0] < 2 or chunk2.shape[0] < 2:
        return 0.0
    combined = np.vstack([chunk1, chunk2])
    fit_pca_if_needed('openl3_chunkwise', combined, n_components=n_components)
    r1 = apply_pca('openl3_chunkwise', chunk1, n_components=n_components)
    r2 = apply_pca('openl3_chunkwise', chunk2, n_components=n_components)
    if r1.ndim == 1 or r2.ndim == 1:  # 防止被 squeeze 掉維度
        return cos_sim(r1, r2)
    cost = librosa.sequence.dtw(X=r1.T, Y=r2.T, metric='euclidean')[0]
    dtw_dist = cost[-1, -1]
    return 1 / (1 + dtw_dist / len(r1))

# =============== 主流程 ===============


def compute_audio_features(audio_path: str, use_openl3: bool = True) -> Optional[Dict[str, Any]]:
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

    features = {**stat_feat, **{k: v for k,
                                v in deep_feat.items() if v is not None}}
    save_audio_features_to_cache(audio_path, features)
    return features


def compute_similarity(f1: Dict[str, Any], f2: Dict[str, Any]) -> float:
    detailed, scores, weights = {}, [], []

    # Onset
    if 'onset_env' in f1 and 'onset_env' in f2:
        dtw_score = dtw_sim(f1['onset_env'], f2['onset_env'])
        scores.append(dtw_score)
        weights.append(SIMILARITY_WEIGHTS['onset_env'])
        detailed['onset_env'] = (dtw_score, SIMILARITY_WEIGHTS['onset_env'])

    # 統計特徵：mfcc, mfcc_delta, chroma
    for k in ['mfcc', 'mfcc_delta', 'chroma']:
        for stat in ['mean', 'std']:
            if k in f1 and k in f2 and stat in f1[k] and stat in f2[k]:
                raw = cos_sim(f1[k][stat], f2[k][stat])
                sim = raw**2
                weight = SIMILARITY_WEIGHTS.get(
                    f"{k}_{stat}", SIMILARITY_WEIGHTS.get(k, 1.0))
                scores.append(sim)
                weights.append(weight)
                detailed[f"{k}_{stat}"] = (sim, weight)

    # Tempo
    if 'tempo' in f1 and 'tempo' in f2:
        t1, t2 = f1['tempo'], f2['tempo']
        s1 = 1 / (1 + abs(t1['mean'] - t2['mean']) / 30)
        s2 = 1 / (1 + abs(t1['std'] - t2['std']) / 15)
        s3 = 1 / (1 + abs(t1['range'] - t2['range']) / 30)
        sim = 0.5 * s1 + 0.25 * s2 + 0.25 * s3
        scores.append(sim)
        weights.append(SIMILARITY_WEIGHTS['tempo'])
        detailed['tempo'] = (sim, SIMILARITY_WEIGHTS['tempo'])

     # 加入 OpenL3 chunkwise DTW 比對（若有）
    if 'openl3_features' in f1 and 'openl3_features' in f2:
        o1 = f1['openl3_features']
        o2 = f2['openl3_features']
        if isinstance(o1, dict) and 'chunkwise' in o1 and 'chunkwise' in o2:
            sim = chunkwise_dtw_sim(o1['chunkwise'], o2['chunkwise'])
            scores.append(sim)
            weights.append(SIMILARITY_WEIGHTS.get('openl3_chunkwise', 1.0))
            detailed['openl3_chunkwise'] = (
                sim, SIMILARITY_WEIGHTS.get('openl3_chunkwise', 1.0))

    # PANN & OpenL3
    for k in ['dl_features', 'pann_features', 'openl3_features']:
        v1, v2 = f1.get(k), f2.get(k)
        if v1 is None or v2 is None:
            continue
        try:
            # OpenL3 chunkwise first
            if k == 'openl3_features':
                if isinstance(v1, dict):
                    if 'chunkwise' in v1 and 'chunkwise' in v2:
                        sim_chunk = chunkwise_dtw_sim(
                            v1['chunkwise'], v2['chunkwise'])
                        scores.append(sim_chunk)
                        weights.append(SIMILARITY_WEIGHTS.get(
                            'openl3_chunkwise', 1.0))
                        detailed['openl3_chunkwise'] = (
                            sim_chunk, SIMILARITY_WEIGHTS.get('openl3_chunkwise', 1.0))
                    if 'merged' in v1 and 'merged' in v2:
                        sim_merged = cos_sim(v1['merged'], v2['merged'])
                        scores.append(sim_merged)
                        weights.append(SIMILARITY_WEIGHTS.get(
                            'openl3_features', 1.0))
                        detailed['openl3_features'] = (
                            sim_merged, SIMILARITY_WEIGHTS.get('openl3_features', 1.0))
                    continue

            # 2D arrays via Chamfer
            if isinstance(
                    v1, np.ndarray) and v1.ndim == 2 and isinstance(
                    v2, np.ndarray) and v2.ndim == 2:
                sim = chamfer_sim(v1, v2, top_k=3)
            elif k == 'pann_features':
                split = 2048
                emb1, tag1 = v1[:split], v1[split:]
                emb2, tag2 = v2[:split], v2[split:]
                sim = 0.6 * cos_sim(emb1, emb2) + 0.4 * cos_sim(tag1, tag2)
            else:
                # OpenL3 merged fallback
                if isinstance(v1, dict) and 'merged' in v1 and 'merged' in v2:
                    sim = cos_sim(v1['merged'], v2['merged'])
                else:
                    sim = cos_sim(v1.flatten(), v2.flatten())
            weight = SIMILARITY_WEIGHTS.get(k, 1.0)
            scores.append(sim)
            weights.append(weight)
            detailed[k] = (sim, weight)
        except Exception as e:
            logger.warning(f"比對 {k} 發生錯誤: {e}")

    if not scores:
        logger.error("所有相似度評估皆失敗，無法進行加權")
        return 0.0

    final = float(np.average(scores, weights=weights))
    for name, (s, w) in detailed.items():
        logger.info(f"  {name:20s} | 相似度: {s:.4f} | 權重: {w}")
    return final


def audio_similarity(path1: str, path2: str) -> float:
    log_memory("開始")
    f1 = compute_audio_features(path1)
    f2 = compute_audio_features(path2)
    log_memory("完成")
    print(f"f1 keys: {list(f1.keys())}")
    print(f"f2 keys: {list(f2.keys())}")
    if not f1 or not f2:
        logger.error("音頻特徵提取失敗，無法比較")
        logger.error(f"音頻文件1: {path1}")
        logger.error(f"音頻文件2: {path2}")
        return 0.0

    raw_audio_similarity = compute_similarity(f1, f2)
    adjusted_audio_similarity = perceptual_score(raw_audio_similarity)

    logger.info(
        f"音頻原始相似度: {raw_audio_similarity:.4f} -> 感知相似度: {adjusted_audio_similarity:.4f}")
    return adjusted_audio_similarity
