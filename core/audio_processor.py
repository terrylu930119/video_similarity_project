# core/audio_processor.py
import os
import gc
import time
import torch
import psutil
import ffmpeg
import librosa
import traceback
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

# =============== 全局配置參數 ===============
_pann_model_loaded = False  # PANN 權重載入一次性提示開關

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
    'dl_features': 2.2,          # 以 Mel 池化而得的深度式片段嵌入（整體音色輪廓相似度）
    'pann_features': 2.2,        # PANN 嵌入 + 聲學標籤（聲音事件/語義相似度）
    'openl3_features': 1.8,      # OpenL3 merged（語音/音樂語義整體相似）
    'openl3_chunkwise': 0.2,     # OpenL3 chunkwise + DTW（時間序列對齊後的語義趨勢）
    'mfcc': 1.2,                 # MFCC 整體（多統計綜合：音色包絡）
    'mfcc_delta': 1.0,           # MFCC 一階差分（動態音色變化）
    'chroma': 1.4,               # 色度圖（和聲/音高結構）
    'mfcc_mean': 1.3,            # MFCC 均值（全局音色分布中心）
    'mfcc_std': 1.0,             # MFCC 標準差（音色變化幅度）
    'mfcc_delta_mean': 0.8,      # ΔMFCC 均值（動態趨勢平均）
    'mfcc_delta_std': 1.2,       # ΔMFCC 標準差（動態穩定性）
    'chroma_mean': 1.0,          # 色度均值（和聲配置中心）
    'chroma_std': 1.0,           # 色度標準差（和聲變化幅度）
    'onset_env': 1.4,            # Onset 強度包絡（節奏能量起伏）
    'tempo': 1.3,                # 節奏速度（BPM 與波動）
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

# =============== PCA 快取工具 ===============


class PCACache:
    """
    功能：維護一個小型 PCA 模型快取以避免重複擬合
    邊界：超出容量時自動移除最舊的項目
    """

    def __init__(self, max_items: int = 15) -> None:
        self.cache: OrderedDict[str, PCA] = OrderedDict()
        self.max_items = max_items

    def get(self, name: str) -> Optional[PCA]:
        """功能：取得名稱對應的 PCA 實例（可能為 None）"""
        return self.cache.get(name)

    def set(self, name: str, pca: PCA) -> None:
        """功能：寫入快取並維持 LRU；超容時淘汰最舊"""
        self.cache[name] = pca
        self.cache.move_to_end(name)
        if len(self.cache) > self.max_items:
            evicted = self.cache.popitem(last=False)
            logger.info(f"自動清除 PCA: {evicted[0]}")

    def clear(self) -> None:
        """功能：清空快取"""
        self.cache.clear()


_pca_registry = PCACache()

# =============== 模型載入器 ===============


@lru_cache(maxsize=3)
def get_mel_transform(sr: int):
    """
    功能：建立或快取 MelSpectrogram 轉換器
    邊界：依取樣率建立不同實例（已掛裝置，避免重建開銷）
    """
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=2048, hop_length=FEATURE_CONFIG['mel']['hop_length'],
        n_mels=FEATURE_CONFIG['mel']['n_mels']
    ).to(device)


@lru_cache(maxsize=1)
def get_openl3_model():
    """
    功能：載入並快取 OpenL3 模型（eval + 可用時 DataParallel）
    邊界：避免重複載入造成 VRAM 抖動與初始化延遲
    """
    global openl3_model
    # ──────────────── 第1階段：快取檢查 ────────────────
    if openl3_model is None:
        # ──────────────── 第2階段：載入與上裝置 ────────────────
        openl3_model = torchopenl3.models.load_audio_embedding_model("mel128", "music", 512).to(device)
        # ──────────────── 第3階段：視情包裝資料並列 ────────────────
        if device.type == 'cuda':
            openl3_model = torch.nn.DataParallel(openl3_model)
        # ──────────────── 第4階段：切換推論模式 ────────────────
        openl3_model.eval()
    return openl3_model


@lru_cache(maxsize=1)
def get_pann_model():
    """
    功能：載入並快取 PANN Cnn14 模型（權重就緒、上裝置、eval）
    邊界：僅首次載入時印出成功訊息；之後直接回傳單例
    規則：保證與 AUDIO_CONFIG 的取樣率一致，避免不匹配
    """
    global _pann_model_loaded
    # ──────────────── 第1階段：確保權重路徑 ────────────────
    checkpoint_path = ensure_pann_weights()
    # ──────────────── 第2階段：初始化模型結構 ────────────────
    model = Cnn14(sample_rate=AUDIO_CONFIG['sample_rate'], window_size=1024,
                  hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    # ──────────────── 第3階段：載入權重（map 到目標裝置） ────────────────
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    # ──────────────── 第4階段：一次性提示與狀態切換 ────────────────
    if not _pann_model_loaded:
        logger.info("PANN 模型權重載入成功")
        _pann_model_loaded = True
    # ──────────────── 第5階段：上裝置並設為 eval ────────────────
    return model.to(device).eval()


@lru_cache(maxsize=32)
def get_optimal_chunk_size(file_size: int) -> float:
    """
    功能：依檔案大小估算最佳切片秒數
    規則：大檔切短片避免 OOM；小檔可拉長以減少 overhead
    """
    if file_size > (1 << 30):
        return 15.0
    if file_size > (512 << 20):
        return 30.0
    return 60.0

# =============== 小工具函式 ===============


def log_memory(stage: str) -> None:
    """
    功能：列印記憶體使用量（RSS）
    """
    print(f"[{stage}] Memory: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")


def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """
    功能：對音訊波形做標準化處理（去均值 / 振幅歸一 / L2）
    規則：三步驟依序進行，降低幅度差與偏移對特徵影響
    """
    # ──────────────── 第1階段：去均值 ────────────────
    waveform = waveform - waveform.mean()
    # ──────────────── 第2階段：幅度正規化 ────────────────
    waveform = waveform / waveform.abs().max().clamp(min=1e-6)
    # ──────────────── 第3階段：L2 正規化 ────────────────
    return waveform / (waveform.norm(p=2) + 1e-9)


def _to_path_hash(path: str) -> str:
    """
    功能：把檔案路徑轉成短 SHA1 hash（前 10 碼）
    """
    return sha1(path.encode('utf-8')).hexdigest()[:10]


def get_cache_path(audio_path: str) -> str:
    """
    功能：取得音檔對應的特徵快取檔路徑（.npz）
    """
    return os.path.join(FEATURE_CACHE_DIR, f"{os.path.basename(audio_path)}_{_to_path_hash(audio_path)}.npz")


def save_audio_features_to_cache(audio_path: str, features: Dict[str, Any]) -> None:
    """
    功能：安全寫入特徵快取（檔鎖避免競態）；存在則略過
    邊界：寫入失敗會記錄警告，不拋例外
    """
    # ──────────────── 第1階段：定位路徑與鎖 ────────────────
    cache_path = get_cache_path(audio_path)
    lock_path = cache_path + ".lock"
    # ──────────────── 第2階段：快取已存在則返回 ────────────────
    if os.path.exists(cache_path):
        return
    # ──────────────── 第3階段：鎖定並寫入 ────────────────
    with FileLock(lock_path):
        if os.path.exists(cache_path):
            return
        try:
            np.savez_compressed(cache_path, **features)
            logger.info(f"特徵快取儲存成功: {cache_path}")
        except Exception as e:
            logger.warning(f"儲存特徵快取失敗: {e}")


def load_audio_features_from_cache(audio_path: str) -> Optional[Dict[str, Any]]:
    """
    功能：讀取特徵快取並做必要的形狀/型別修正
    邊界：不存在或失敗回 None
    """
    try:
        # ──────────────── 第1階段：定位快取檔 ────────────────
        cache_path = get_cache_path(audio_path)
        if not os.path.exists(cache_path):
            return None
        # ──────────────── 第2階段：讀入並解包 ────────────────
        data = np.load(cache_path, allow_pickle=True)
        loaded = {k: data[k].item() if data[k].shape == () else data[k] for k in data}
        # ──────────────── 第3階段：形狀/型別修正 ────────────────
        return _ensure_feature_shapes(loaded)
    except Exception as e:
        logger.warning(f"載入特徵快取失敗: {e}")
        return None


def load_audio(audio_path: str) -> Generator[Tuple[np.ndarray, int], None, None]:
    """
    功能：以串流方式讀取音訊（降低常駐記憶體）
    邊界：出錯時記錄並終止迭代（回 None）
    """
    # ──────────────── 第1階段：估算分段長度 ────────────────
    file_size = os.path.getsize(audio_path)
    chunk_duration = get_optimal_chunk_size(file_size)
    try:
        # ──────────────── 第2階段：建立串流迭代器 ────────────────
        stream = librosa.stream(audio_path, block_length=int(chunk_duration * 22050),
                                frame_length=2048, hop_length=1024)
        # ──────────────── 第3階段：逐段讀取與內存防護 ────────────────
        for y_block in stream:
            if psutil.virtual_memory().percent > 80:
                gc.collect()
                time.sleep(1)
            yield y_block, 22050
    except Exception as e:
        logger.error(f"載入音頻文件失敗 {audio_path}: {str(e)}")
        return None


def perceptual_score(sim_score: float) -> float:
    """
    功能：感知再映射，使高分更嚴格、低分更寬鬆
    規則：以動態 gamma 對原始分數做單調映射
    """
    gamma = 1.2 + 1.0 * (1 - sim_score)
    return float(min(max(sim_score ** gamma, 0.0), 1.0))


def fit_pca_if_needed(name: str, data: np.ndarray, n_components: int) -> Optional[PCA]:
    """
    功能：必要時擬合 PCA 並快取
    邊界：主成分數自動夾擠至樣本/維度上限；不足回 None
    """
    # ──────────────── 第1階段：快取檢查 ────────────────
    p = _pca_registry.get(name)
    if p is not None:
        return p
    # ──────────────── 第2階段：夾擠成分數與可行性判斷 ────────────────
    n_samples, dim = data.shape
    n_components = min(n_components, n_samples, dim)
    if n_components < 2:
        return None
    # ──────────────── 第3階段：擬合與快取 ────────────────
    pca = PCA(n_components=n_components)
    pca.fit(data)
    _pca_registry.set(name, pca)
    return pca


def apply_pca(name: str, vector: np.ndarray, n_components: int) -> np.ndarray:
    """
    功能：使用已擬合 PCA 轉換向量；若無則原樣返回
    邊界：自動處理 1D 形狀
    """
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    pca = _pca_registry.get(name)
    return (pca.transform(vector) if pca is not None else vector).squeeze()


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    功能：餘弦相似度轉 [0,1]
    邊界：自動對齊長度；空值回 0
    """
    # ──────────────── 第1階段：展平與轉型 ────────────────
    a = np.ravel(a).astype(np.float64)
    b = np.ravel(b).astype(np.float64)
    # ──────────────── 第2階段：長度對齊與邊界 ────────────────
    m = min(a.size, b.size)
    if m == 0:
        return 0.0
    a = a[:m]
    b = b[:m]
    # ──────────────── 第3階段：計算並映射 ────────────────
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float((float(np.dot(a, b) / denom) + 1.0) / 2.0)


def dtw_sim(a: np.ndarray, b: np.ndarray, max_length: int = 500) -> float:
    """
    功能：1D DTW 距離轉相似度
    邊界：長度截斷避免計算爆炸；空值回 0
    """
    a = np.ravel(a)[:max_length]
    b = np.ravel(b)[:max_length]
    if a.size == 0 or b.size == 0:
        return 0.0
    cost = librosa.sequence.dtw(X=a.reshape(1, -1), Y=b.reshape(1, -1), metric='euclidean')[0]
    return float(1.0 / (1.0 + cost[-1, -1] / len(a)))


def chamfer_sim(a: np.ndarray, b: np.ndarray, top_k: int = 3) -> float:
    """
    功能：類 Chamfer 距離（雙向 Top-K 相似度平均）
    邊界：僅適用 2D 集合；空集合回 0
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
        return 0.0
    S = cosine_similarity(a, b)
    topA = np.mean(np.sort(S, axis=1)[:, -top_k:], axis=1)
    topB = np.mean(np.sort(S, axis=0)[-top_k:, :], axis=0)
    return float((topA.mean() + topB.mean()) / 2.0)


def _as_float(x: Any) -> float:
    """
    功能：寬容轉 float；陣列取平均，失敗回 0.0
    """
    try:
        arr = np.asarray(x)
        return float(arr if arr.ndim == 0 else arr.mean())
    except Exception:
        try:
            return float(x)
        except Exception:
            return 0.0

# =============== Normalizers（集中處理形狀/型別） ===============


def normalize_onset_env(x: Any) -> np.ndarray:
    """
    功能：規整 onset 強度包絡為 1D float32
    """
    return np.asarray(x, dtype=np.float32).reshape(-1)


def normalize_tempo(x: Any) -> Dict[str, float]:
    """
    功能：將節奏資訊統一為 {mean, std, range}
    邊界：非字典輸入以單值填入 mean，其餘為 0
    """
    if isinstance(x, dict):
        return {'mean': float(x.get('mean', 0.0)),
                'std': float(x.get('std', 0.0)),
                'range': float(x.get('range', 0.0))}
    v = float(x) if not isinstance(x, (list, np.ndarray, dict)) else _as_float(x)
    return {'mean': v, 'std': 0.0, 'range': 0.0}


def normalize_stats_block(block: Any) -> Dict[str, np.ndarray]:
    """
    功能：將統計區塊整理為 1D 陣列的字典（mean/std/max/min/median）
    邊界：非字典輸入回空字典
    """
    if not isinstance(block, dict):
        return {}

    def to_vec(z: Any) -> np.ndarray: return np.asarray(z, dtype=np.float32).reshape(-1)
    out: Dict[str, np.ndarray] = {}
    for stat in ('mean', 'std', 'max', 'min', 'median'):
        if stat in block:
            out[stat] = to_vec(block[stat])
    return out


def normalize_dl(arr: Any) -> np.ndarray:
    """
    功能：將 DL 特徵轉為 >=2D 形狀（1D → (1,D)）
    """
    arr = np.asarray(arr, dtype=np.float32)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def normalize_pann(arr: Any) -> np.ndarray:
    """
    功能：將 PANN 特徵轉為 float32 陣列
    """
    return np.asarray(arr, dtype=np.float32)


def normalize_openl3(v: Any) -> Dict[str, np.ndarray]:
    """
    功能：規整 OpenL3 特徵為 {'merged':1D, 'chunkwise':2D} 子集
    邊界：非字典輸入僅輸出 merged 單向量
    """
    if isinstance(v, dict):
        out: Dict[str, np.ndarray] = {}
        if 'merged' in v:
            out['merged'] = np.asarray(v['merged'], dtype=np.float32).reshape(-1)
        if 'chunkwise' in v:
            arr = np.asarray(v['chunkwise'], dtype=np.float32)
            out['chunkwise'] = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 1 else arr
        return out
    return {'merged': np.asarray(v, dtype=np.float32).reshape(-1)}


def _ensure_feature_shapes(feats: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：對特徵字典進行型別與形狀的安全校正
    邊界：輸入為空則回空字典
    """
    # ──────────────── 第1階段：空值處理與淺拷貝 ────────────────
    if not feats:
        return {}
    out: Dict[str, Any] = dict(feats)
    # ──────────────── 第2階段：逐欄位形狀校正 ────────────────
    if 'onset_env' in out:
        out['onset_env'] = normalize_onset_env(out['onset_env'])
    if 'tempo' in out:
        out['tempo'] = normalize_tempo(out['tempo'])
    for name in ('mfcc', 'mfcc_delta', 'chroma', 'mel'):
        if name in out:
            out[name] = normalize_stats_block(out[name])
    if 'dl_features' in out:
        out['dl_features'] = normalize_dl(out['dl_features'])
    if 'pann_features' in out:
        out['pann_features'] = normalize_pann(out['pann_features'])
    if 'openl3_features' in out:
        out['openl3_features'] = normalize_openl3(out['openl3_features'])
    # ──────────────── 第3階段：回傳校正結果 ────────────────
    return out

# =============== OpenL3 輔助 ===============


def _load_mono_resample48k(path: str) -> Tuple[np.ndarray, int]:
    """
    功能：載入音訊、混單聲道並重採樣至 48kHz
    邊界：多聲道以均值混合；確保與 OpenL3 一致的取樣率
    """
    # ──────────────── 第1階段：讀檔（保留原 sr） ────────────────
    audio, sr = librosa.load(path, sr=None, mono=False)
    # ──────────────── 第2階段：混單聲道 ────────────────
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    # ──────────────── 第3階段：重採樣到 48k ────────────────
    if sr != 48000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
        sr = 48000
    return audio.astype(np.float32), sr


def _iter_chunks(y: np.ndarray, chunk: int, min_len: int) -> List[np.ndarray]:
    """
    功能：將長序列切為多段並過濾過短段
    邊界：小於 min_len 的片段會被丟棄
    """
    return [y[i:i + chunk] for i in range(0, y.shape[0], chunk)
            if y[i:i + chunk].shape[0] >= min_len]


def _openl3_embed_chunk(model, seg: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """
    功能：對單一段落產生 OpenL3 嵌入並做時間平均為 512 維
    邊界：失敗回 None
    """
    try:
        # ──────────────── 第1階段：Tensor 化並上裝置 ────────────────
        t = torch.tensor(seg[None, :], dtype=torch.float32).to(device)
        # ──────────────── 第2階段：推論取得嵌入 ────────────────
        with torch.no_grad():
            e, _ = torchopenl3.get_audio_embedding(t, sr, model=model, hop_size=1.0, center=True, verbose=False)
        # ──────────────── 第3階段：處理輸出形狀 ────────────────
        if isinstance(e, torch.Tensor):
            e = e.detach().cpu().numpy()
        if e.ndim == 3 and e.shape[-1] == 512:
            e = e[0]
        if e.ndim == 2 and e.shape[1] == 512:
            return e.mean(axis=0).astype(np.float32)
    except Exception as ex:
        logger.warning(f"OpenL3 子段錯誤: {ex}")
    return None


def extract_openl3_features(audio_path: str, chunk_sec: float = 10.0) -> Optional[Dict[str, np.ndarray]]:
    """
    功能：以 OpenL3 產生 merged 與 chunkwise 特徵
    邊界：近乎靜音或長度不足會跳過；完成後主動清理 GPU
    規則：merged = mean+var(512) → 1024 維；chunkwise = (N,512)
    """
    try:
        # ──────────────── 第1階段：載入並規整為 48k 單聲道 ────────────────
        y, sr = _load_mono_resample48k(audio_path)
        # ──────────────── 第2階段：靜音與長度檢查 ────────────────
        if np.max(np.abs(y)) < 1e-5 or y.shape[0] < sr:
            logger.warning(f"音訊近乎靜音或長度不足，跳過：{audio_path}")
            return None
        # ──────────────── 第3階段：取得模型與切段 ────────────────
        model = get_openl3_model()
        chunk = int(chunk_sec * sr)
        # ──────────────── 第4階段：逐段嵌入並彙整 ────────────────
        embs = [e for seg in _iter_chunks(y, chunk, int(0.1 * sr))
                if (e := _openl3_embed_chunk(model, seg, sr)) is not None]
        if not embs:
            logger.warning(f"OpenL3 段落提取皆失敗：{audio_path}")
            return None
        arr = np.stack(embs).astype(np.float32)
        gpu_manager.clear_gpu_memory()
        # ──────────────── 第5階段：組合輸出 ────────────────
        return {"merged": np.concatenate([arr.mean(axis=0), arr.var(axis=0)]).astype(np.float32),
                "chunkwise": arr}
    except Exception as e:
        logger.warning(f"OpenL3 error: {e}")
        return None

# =============== DL（mel pooling） ===============


def _extract_dl_chunk(mel_transform, chunk: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """
    功能：對單一片段做 Mel 轉換並時間平均
    邊界：片段小於 1 秒時丟棄；失敗回 None
    """
    if len(chunk) < sr:
        return None
    try:
        # ──────────────── 第1階段：標準化並上裝置 ────────────────
        wf = normalize_waveform(torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device))
        # ──────────────── 第2階段：Mel 轉換 + 時間平均 ────────────────
        with torch.no_grad():
            mel = mel_transform(wf)
            pooled = torch.mean(mel, dim=2).squeeze().detach().cpu().numpy()
        return pooled.astype(np.float32)
    except Exception as e:
        logger.warning(f"[DL] chunk failed: {e}")
        return None


def extract_dl_features(audio_path: str, chunk_sec: float = 10.0) -> Optional[np.ndarray]:
    """
    功能：以簡化 DL 管線（Mel 池化）萃取分段音訊嵌入
    邊界：全部片段失敗回 None
    規則：輸出形狀 (N, D)
    """
    try:
        # ──────────────── 第1階段：載入音訊 ────────────────
        y, sr = librosa.load(audio_path, sr=AUDIO_CONFIG['sample_rate'])
        mel_spec_transform = get_mel_transform(sr)
        chunk_size = int(chunk_sec * sr)
        feats = []
        # ──────────────── 第2階段：逐段處理 ────────────────
        for i in range(0, len(y), chunk_size):
            seg = y[i:i + chunk_size]
            emb = _extract_dl_chunk(mel_spec_transform, seg, sr)
            if emb is not None:
                feats.append(emb)
        # ──────────────── 第3階段：結果檢查 ────────────────
        if not feats:
            logger.warning(f"[DL] 全部 chunk 失敗: {audio_path}")
            return None
        return np.stack(feats).astype(np.float32)
    except Exception as e:
        logger.error(f"[DL] feature error: {e}")
        return None

# =============== PANN ===============


def _load_resampled_mono_torch(audio_path: str, target_sr: int) -> Tuple[torch.Tensor, int]:
    """
    功能：用 torchaudio 載入並重採樣到 target_sr（Tensor）
    邊界：維持單聲道張量形狀 (1, T)
    """
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        sr = target_sr
    return waveform, sr


def _split_waveform(waveform: torch.Tensor, sr: int, chunk_sec: float) -> List[torch.Tensor]:
    """
    功能：依指定秒數切分 Tensor 片段
    邊界：小於 1 秒的片段不保留
    """
    chunk_size = int(chunk_sec * sr)
    return [waveform[:, i:i + chunk_size] for i in range(0, waveform.shape[1], chunk_size)
            if waveform[:, i:i + chunk_size].shape[1] >= sr]


def _pann_embed(model: torch.nn.Module, c: torch.Tensor) -> Optional[np.ndarray]:
    """
    功能：對單段做 PANN 推論並拼接 embedding(2048)+tags(527)
    邊界：失敗回 None
    """
    try:
        # ──────────────── 第1階段：前向推論 ────────────────
        with torch.no_grad():
            out = model(c.to(device))
            emb = out['embedding'].squeeze().detach().cpu().numpy()[:2048].astype(np.float32)
            tags = out['clipwise_output'].squeeze().detach().cpu().numpy().astype(np.float32)
        # ──────────────── 第2階段：拼接輸出 ────────────────
        return np.concatenate([emb, tags]).astype(np.float32)
    except Exception as e:
        logger.warning(f"[PANN] chunk failed: {e}")
        return None


def extract_pann_features(audio_path: str, chunk_sec: float = 10.0) -> Optional[np.ndarray]:
    """
    功能：使用 PANN Cnn14 對音訊進行分段推論（N, 2575）
    邊界：沒有任何有效片段回 None
    規則：每段輸出 2048(embedding)+527(tags)
    """
    try:
        # ──────────────── 第1階段：載入並重採樣 ────────────────
        waveform, sr = _load_resampled_mono_torch(audio_path, AUDIO_CONFIG['sample_rate'])
        # ──────────────── 第2階段：取得模型與切段 ────────────────
        model = get_pann_model()
        chunks = _split_waveform(waveform, sr, chunk_sec)
        # ──────────────── 第3階段：逐段推論 ────────────────
        feats = [f for c in chunks if (f := _pann_embed(model, c)) is not None]
        # ──────────────── 第4階段：結果檢查 ────────────────
        if not feats:
            logger.warning(f"[PANN] 沒有有效 chunk: {audio_path}")
            return None
        return np.stack(feats).astype(np.float32)
    except Exception as e:
        logger.error(f"[PANN] 特徵提取失敗: {e}")
        return None

# =============== 傳統統計 ===============


def _stats_matrix(x: np.ndarray) -> Dict[str, np.ndarray]:
    """
    功能：回傳矩陣在軸 1 的基本統計（mean/std/max/min/median）
    """
    return {
        'mean': np.mean(x, axis=1).astype(np.float32),
        'std': np.std(x, axis=1).astype(np.float32),
        'max': np.max(x, axis=1).astype(np.float32),
        'min': np.min(x, axis=1).astype(np.float32),
        'median': np.median(x, axis=1).astype(np.float32),
    }


def _extract_segment_stats(seg: np.ndarray, sr: int) -> Optional[Dict[str, Any]]:
    """
    功能：對單段 seg 計算 MFCC/Δ/Chroma/Mel + Onset/Tempo 統計
    邊界：片段小於 1 秒回 None
    """
    if len(seg) < sr:
        return None
    mel = librosa.feature.melspectrogram(y=seg, sr=sr, **FEATURE_CONFIG['mel'])
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, **FEATURE_CONFIG['mfcc'])
    delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=seg, sr=sr, **FEATURE_CONFIG['chroma'])
    onset = librosa.onset.onset_strength(y=seg, sr=sr)
    tempos = tempo(onset_envelope=onset, sr=sr, aggregate=None)
    return {
        'mfcc': _stats_matrix(mfcc),
        'mfcc_delta': _stats_matrix(delta),
        'chroma': _stats_matrix(chroma),
        'mel': _stats_matrix(mel),
        'onset_env': onset.astype(np.float32),
        'tempo': {
            'mean': float(np.mean(tempos)) if len(tempos) else 0.0,
            'std': float(np.std(tempos)) if len(tempos) else 0.0,
            'range': float(np.max(tempos) - np.min(tempos)) if len(tempos) else 0.0
        }
    }


def _merge_onset(out: Dict[str, Any], feats: List[Dict[str, Any]]) -> None:
    """
    功能：合併所有片段的 onset_env 為單一長序列
    邊界：若無 onset_env 則不產生欄位
    """
    arrs = [f['onset_env'] for f in feats if 'onset_env' in f]
    if arrs:
        out['onset_env'] = np.concatenate(arrs).astype(np.float32)


def _merge_tempo(out: Dict[str, Any], feats: List[Dict[str, Any]]) -> None:
    """
    功能：聚合 tempo 的 mean/std/range（逐段平均）
    邊界：輸入片段未包含 tempo 時不產生欄位
    """
    if 'tempo' in feats[0] and isinstance(feats[0]['tempo'], dict):
        keys = feats[0]['tempo'].keys()
        out['tempo'] = {k: float(np.mean([f['tempo'][k] for f in feats])) for k in keys}


def _merge_stats_blocks(out: Dict[str, Any], feats: List[Dict[str, Any]], names: Tuple[str, ...]) -> None:
    """
    功能：合併指定名稱的統計特徵區塊（對每個統計量做逐段平均）
    """
    for name in names:
        if name in feats[0]:
            out[name] = {stat: np.mean([f[name][stat] for f in feats], axis=0).astype(np.float32)
                         for stat in feats[0][name]}


def combine_features(features: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    功能：將逐段特徵彙整為單一樣本的整體描述
    邊界：輸入為空回 None
    規則：切分為 onset/tempo/統計三個子步驟降低複雜度
    """
    # ──────────────── 第1階段：空集合處理 ────────────────
    if not features:
        return None
    out: Dict[str, Any] = {}
    # ──────────────── 第2階段：合併 onset 與 tempo ────────────────
    _merge_onset(out, features)
    _merge_tempo(out, features)
    # ──────────────── 第3階段：合併統計特徵 ────────────────
    _merge_stats_blocks(out, features, ('mfcc', 'mfcc_delta', 'chroma', 'mel'))
    return out


def extract_statistical_features(audio_path: str, chunk_sec: float = 10.0) -> Optional[Dict[str, Any]]:
    """
    功能：以 librosa 計算統計型特徵並彙整
    邊界：出錯回 None
    規則：逐段計算 → combine_features
    """
    try:
        # ──────────────── 第1階段：載入音訊 ────────────────
        y, sr = librosa.load(audio_path, sr=AUDIO_CONFIG['sample_rate'])
        chunk_size = int(chunk_sec * sr)
        features: List[Dict[str, Any]] = []
        # ──────────────── 第2階段：逐段特徵計算 ────────────────
        for i in range(0, len(y), chunk_size):
            seg = y[i:i + chunk_size]
            item = _extract_segment_stats(seg, sr)
            if item is not None:
                features.append(item)
        # ──────────────── 第3階段：彙整輸出 ────────────────
        return combine_features(features)
    except Exception as e:
        logger.error(f"Stat feature error: {e}")
        return None

# =============== OpenL3 chunkwise DTW ===============


def chunkwise_dtw_sim(chunk1: np.ndarray, chunk2: np.ndarray, n_components: int = 32) -> float:
    """
    功能：OpenL3 chunkwise → PCA 降維 → DTW 比較序列相似度
    邊界：長度不足退化；空集合回 0
    規則：r1/r2 若轉換後降成 1D，改用 cosine
    """
    # ──────────────── 第1階段：輸入整形與邊界 ────────────────
    if not isinstance(chunk1, np.ndarray):
        chunk1 = np.asarray(chunk1)
    if not isinstance(chunk2, np.ndarray):
        chunk2 = np.asarray(chunk2)
    if chunk1.ndim == 1:
        chunk1 = chunk1.reshape(-1, 1)
    if chunk2.ndim == 1:
        chunk2 = chunk2.reshape(-1, 1)
    if chunk1.shape[0] < 2 or chunk2.shape[0] < 2:
        return 0.0
    # ──────────────── 第2階段：擬合/取得 PCA 並降維 ────────────────
    combined = np.vstack([chunk1, chunk2])
    fit_pca_if_needed('openl3_chunkwise', combined, n_components=n_components)
    r1 = apply_pca('openl3_chunkwise', chunk1, n_components=n_components)
    r2 = apply_pca('openl3_chunkwise', chunk2, n_components=n_components)
    # ──────────────── 第3階段：退化條件與 DTW 比較 ────────────────
    if r1.ndim == 1 or r2.ndim == 1:
        return cos_sim(r1, r2)
    cost = librosa.sequence.dtw(X=r1.T, Y=r2.T, metric='euclidean')[0]
    dtw_dist = cost[-1, -1]
    return float(1.0 / (1.0 + dtw_dist / len(r1)))

# =============== 相似度：計分器 ===============


def score_onset(f1: Dict[str, Any], f2: Dict[str, Any]) -> Optional[Tuple[str, float, str]]:
    """
    功能：以 DTW 比較 onset 強度包絡的相似度
    邊界：缺任一欄位回 None
    """
    if 'onset_env' not in f1 or 'onset_env' not in f2:
        return None
    return ('onset_env', _as_float(dtw_sim(f1['onset_env'], f2['onset_env'])), 'onset_env')


def score_stats_block(name: str, f1: Dict[str, Any], f2: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """
    功能：比較統計特徵區塊（如 MFCC/Δ/Chroma）的 mean/std 相似度
    邊界：缺區塊或型別不符回空清單
    """
    res: List[Tuple[str, float, str]] = []
    b1, b2 = f1.get(name), f2.get(name)
    if not isinstance(b1, dict) or not isinstance(b2, dict):
        return res
    for stat in ('mean', 'std'):
        if stat in b1 and stat in b2:
            sim = _as_float(cos_sim(b1[stat], b2[stat]) ** 2)
            res.append((f"{name}_{stat}", sim, f"{name}_{stat}"))
    return res


def score_tempo(f1: Dict[str, Any], f2: Dict[str, Any]) -> Optional[Tuple[str, float, str]]:
    """
    功能：以 mean/std/range 綜合衡量節奏相似度
    邊界：缺資料回 None
    """
    if 'tempo' not in f1 or 'tempo' not in f2:
        return None
    t1, t2 = f1['tempo'], f2['tempo']
    s1 = 1.0 / (1.0 + abs(float(t1['mean']) - float(t2['mean'])) / 30.0)
    s2 = 1.0 / (1.0 + abs(float(t1['std']) - float(t2['std'])) / 15.0)
    s3 = 1.0 / (1.0 + abs(float(t1['range']) - float(t2['range'])) / 30.0)
    return ('tempo', _as_float(0.5 * s1 + 0.25 * s2 + 0.25 * s3), 'tempo')


def score_openl3(f1: Dict[str, Any], f2: Dict[str, Any]) -> Optional[Tuple[str, float, str]]:
    """
    功能：比較 OpenL3 merged 與 chunkwise 的相似度並取平均
    邊界：兩側皆無有效子項回 None
    """
    if 'openl3_features' not in f1 or 'openl3_features' not in f2:
        return None
    o1, o2 = f1['openl3_features'], f2['openl3_features']
    sims: List[float] = []
    if isinstance(o1, dict) and isinstance(o2, dict):
        if 'merged' in o1 and 'merged' in o2:
            sims.append(_as_float(cos_sim(o1['merged'], o2['merged'])))
        if 'chunkwise' in o1 and 'chunkwise' in o2:
            sims.append(_as_float(chunkwise_dtw_sim(o1['chunkwise'], o2['chunkwise'])))
    else:
        sims.append(_as_float(cos_sim(np.asarray(o1), np.asarray(o2))))
    if not sims:
        return None
    return ('openl3_features', float(np.mean(sims)), 'openl3_features')


def score_deep(name: str, v1: Any, v2: Any) -> Optional[Tuple[str, float, str]]:
    """
    功能：通用深度特徵相似度（2D 用 Chamfer，1D 用 cosine）
    邊界：任一為 None 回 None
    """
    if v1 is None or v2 is None:
        return None
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    if v1.ndim == 2 and v2.ndim == 2:
        sim = _as_float(chamfer_sim(v1, v2, top_k=3))
    else:
        sim = _as_float(cos_sim(v1, v2))
    return (name, sim, name)


def score_pann(f1: Dict[str, Any], f2: Dict[str, Any]) -> Optional[Tuple[str, float, str]]:
    """
    功能：PANN 特徵相似度；可拆解為 embedding/tag 加權融合
    邊界：形狀不符時退化為通用深度相似度
    """
    if 'pann_features' not in f1 or 'pann_features' not in f2:
        return None
    v1, v2 = np.asarray(f1['pann_features']), np.asarray(f2['pann_features'])
    if v1.ndim == 1 and v1.shape[0] >= 2575 and v2.ndim == 1 and v2.shape[0] >= 2575:
        split = 2048
        emb1, tag1 = v1[:split], v1[split:]
        emb2, tag2 = v2[:split], v2[split:]
        sim = _as_float(0.6 * cos_sim(emb1, emb2) + 0.4 * cos_sim(tag1, tag2))
        return ('pann_features', sim, 'pann_features')
    return score_deep('pann_features', v1, v2)

# =============== 合併相似度的累加器 ===============


class ScoreAccumulator:
    """
    功能：統一累積各模組分數與權重並計算加權平均
    邊界：無分數時回 0
    """

    def __init__(self) -> None:
        self.detailed: Dict[str, Tuple[float, float]] = {}
        self.scores: List[float] = []
        self.weights: List[float] = []

    def push(self, item: Optional[Tuple[str, float, str]]) -> None:
        """
        功能：寫入 (name, score, weight_key) 一筆分數
        邊界：None 直接忽略
        規則：weight_key > name > 1.0 的優先順序取得權重
        """
        if item is None:
            return
        name, score, weight_key = item
        w = float(SIMILARITY_WEIGHTS.get(weight_key, SIMILARITY_WEIGHTS.get(name, 1.0)))
        self.scores.append(score)
        self.weights.append(w)
        self.detailed[name] = (score, w)

    def weighted_average(self) -> float:
        """
        功能：回傳加權平均分數
        邊界：無分數回 0
        """
        if not self.scores:
            return 0.0
        return float(np.average(np.asarray(self.scores, dtype=np.float64),
                                weights=np.asarray(self.weights, dtype=np.float64)))

    def log(self) -> None:
        """
        功能：輸出詳細分項（名稱/分數/權重）
        """
        for name, (s, w) in self.detailed.items():
            logger.info(f"  {name:20s} | 相似度: {s:.4f} | 權重: {w}")


def compute_similarity(f1: Dict[str, Any], f2: Dict[str, Any]) -> float:
    """
    功能：彙整各計分器的分數並依權重做加權平均
    邊界：若無任何有效分數則回 0
    """
    # ──────────────── 第1階段：初始化累加器 ────────────────
    acc = ScoreAccumulator()
    # ──────────────── 第2階段：累積各模組分數 ────────────────
    acc.push(score_onset(f1, f2))
    for k in ('mfcc', 'mfcc_delta', 'chroma'):
        for t in score_stats_block(k, f1, f2):
            acc.push(t)
    acc.push(score_tempo(f1, f2))
    acc.push(score_openl3(f1, f2))
    acc.push(score_pann(f1, f2))
    acc.push(score_deep('dl_features', f1.get('dl_features'), f2.get('dl_features')))
    # ──────────────── 第3階段：加權融合與日誌 ────────────────
    final = acc.weighted_average()
    if final == 0.0:
        logger.error("所有相似度評估皆失敗，無法進行加權")
    acc.log()
    return final

# =============== 主流程 ===============


def _gather_deep_features(audio_path: str, use_openl3: bool) -> Dict[str, Any]:
    """
    功能：收集深度特徵（DL/PANN/OpenL3）
    邊界：OpenL3 可關閉；失敗會回 None 值
    """
    return {
        'dl_features': extract_dl_features(audio_path),
        'pann_features': extract_pann_features(audio_path),
        'openl3_features': extract_openl3_features(audio_path) if use_openl3 else None
    }


def compute_audio_features(audio_path: str, use_openl3: bool = True) -> Optional[Dict[str, Any]]:
    """
    功能：整合抽取統計與深度特徵，並做型態校正與快取
    邊界：統計特徵為 None 時終止；部分深度特徵可缺失
    規則：先查快取 → 並行抽取 → 形狀修正 → 寫入快取
    """
    # ──────────────── 第1階段：檢查快取 ────────────────
    cached = load_audio_features_from_cache(audio_path)
    if cached is not None:
        return cached
    # ──────────────── 第2階段：並行提取統計與深度特徵 ────────────────
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_stat = pool.submit(extract_statistical_features, audio_path)
        f_deep = pool.submit(_gather_deep_features, audio_path, use_openl3)
        stat_feat = f_stat.result()
        deep_feat = f_deep.result()
    # ──────────────── 第3階段：錯誤與缺失處理 ────────────────
    if stat_feat is None:
        logger.warning(f"Statistical 特徵提取失敗: {audio_path}")
        return None
    for k in ('dl_features', 'pann_features', 'openl3_features'):
        if deep_feat.get(k) is None:
            logger.warning(f"Deep 特徵缺失：{k} 在 {audio_path}")
    # ──────────────── 第4階段：合併、校正與快取 ────────────────
    features: Dict[str, Any] = {**stat_feat, **{k: v for k, v in deep_feat.items() if v is not None}}
    features = _ensure_feature_shapes(features)
    save_audio_features_to_cache(audio_path, features)
    return features


def audio_similarity(path1: str, path2: str) -> float:
    """
    功能：音訊相似度主流程（特徵→加權→感知校正）
    邊界：任一側特徵無效則回 0
    規則：log 兩次內存以便觀測
    """
    try:
        # ──────────────── 第1階段：提取特徵並記錄內存 ────────────────
        log_memory("開始")
        f1 = compute_audio_features(path1)
        f2 = compute_audio_features(path2)
        log_memory("完成")
        # ──────────────── 第2階段：型別檢查 ────────────────
        if not isinstance(f1, dict) or not isinstance(f2, dict):
            logger.error(f"音頻特徵型別錯誤: f1={type(f1)}, f2={type(f2)}")
            return 0.0
        # ──────────────── 第3階段：融合並感知校正 ────────────────
        raw = compute_similarity(f1, f2)
        adj = perceptual_score(raw)
        logger.info(f"音頻原始相似度: {raw:.4f} -> 感知相似度: {adj:.4f}")
        return adj
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"計算音頻相似度時出錯: {e}\n{tb}")
        return 0.0


def extract_audio(video_path: str) -> str:
    """
    功能：從影片中萃取音訊為 WAV（單聲道/32kHz/PCM16）
    邊界：具備重試機制；無寫入權限或產出失敗會擲出例外
    規則：音訊快取存在則直接返回
    """
    try:
        # ──────────────── 第1階段：輸入與目錄檢查 ────────────────
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"影片檔案不存在: {video_path}")
        video_dir = os.path.dirname(os.path.abspath(video_path))
        audio_dir = os.path.join(video_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = os.path.join(audio_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.wav")
        if not os.access(audio_dir, os.W_OK):
            raise PermissionError(f"沒有輸出目錄的寫入權限: {audio_dir}")
        # ──────────────── 第2階段：快取命中 ────────────────
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            return audio_path
        # ──────────────── 第3階段：FFmpeg 轉出音訊 ────────────────
        ffmpeg.input(video_path).output(
            audio_path,
            acodec=AUDIO_CONFIG['codec'],
            ac=AUDIO_CONFIG['channels'],
            ar=AUDIO_CONFIG['sample_rate'],
            format=AUDIO_CONFIG['format'],
            audio_bitrate=AUDIO_CONFIG['audio_bitrate']
        ).overwrite_output().run(quiet=True, capture_stdout=True, capture_stderr=True)
        # ──────────────── 第4階段：重試檢查輸出 ────────────────
        for _ in range(5):
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                return audio_path
            time.sleep(1)
        # ──────────────── 第5階段：超時失敗 ────────────────
        raise RuntimeError("音訊檔案生成失敗")
    except Exception as e:
        logger.error(f"音訊提取失敗: {str(e)}")
        raise
