import os
import gc
import time
import json
import math
import torch
import psutil
import ffmpeg
import librosa
import torchaudio
import torchopenl3
import numpy as np
import traceback
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
# 功能：定義音訊處理、特徵擷取與執行緒的預設行為與權重；集中化便於調參與 A/B 測試。
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

# 權重說明：值越大，對總體音訊相似度的影響越高。
SIMILARITY_WEIGHTS = {
    'dl_features': 2.2,
    'pann_features': 2.2,
    'openl3_features': 1.8,
    'openl3_chunkwise': 0.2,

    'mfcc': 1.2,
    'mfcc_delta': 1.0,
    'chroma': 1.4,

    'mfcc_mean': 1.3,
    'mfcc_std': 1.0,
    'mfcc_delta_mean': 0.8,
    'mfcc_delta_std': 1.2,
    'chroma_mean': 1.0,
    'chroma_std': 1.0,

    'onset_env': 1.4,
    'tempo': 1.3,
}

THREAD_CONFIG = {'max_workers': 6}
CROP_CONFIG = {'min_duration': 30.0, 'max_duration': 300.0,
               'overlap': 0.5, 'silence_threshold': -14}

# =============== 初始化模型與資源 ===============
# 功能：建立特徵快取目錄、初始化 GPU 管理器與全域模型（延遲載入）。
FEATURE_CACHE_DIR = os.path.join(os.getcwd(), "feature_cache")
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

gpu_manager.initialize()
device = gpu_manager.get_device()

pann_model: Optional[torch.nn.Module] = None
openl3_model: Optional[torch.nn.Module] = None

# =============== 小工具 ===============
# 功能：各種輔助元件（PCA 快取、轉換器、路徑與記憶體工具），用以降低重工與提升穩定性。


class PCACache:
    """
    功能：簡單的 LRU 風格 PCA 模型快取，避免重複擬合浪費時間。
    欄位：
      - cache：名稱 → PCA 物件的有序字典
      - max_items：最多保留的 PCA 個數，超出會自動淘汰最舊項
    """

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


_pca_registry = PCACache()


@lru_cache(maxsize=3)
def get_mel_transform(sr: int):
    """
    功能：取得（並快取）MelSpectrogram 轉換器
    為什麼：避免在多段音訊/分段處理時重建轉換器導致 GPU/CPU 開銷。
    回傳：已遷移至對應 device 的 torchaudio 模型
    """
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=FEATURE_CONFIG['mel']['hop_length'],
        n_mels=FEATURE_CONFIG['mel']['n_mels']
    ).to(device)


@lru_cache(maxsize=1)
def get_openl3_model():
    """
    功能：延遲載入 OpenL3 音訊嵌入模型，並在 CUDA 可用時使用 DataParallel。
    為什麼：OpenL3 初始化昂貴；快取模型以提升多次請求的效能穩定性。
    """
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
    """
    功能：載入 PANN(Cnn14) 權重並切到 eval 模式。
    邊界：首次載入時記錄成功訊息，避免每段音訊重複刷 log。
    """
    global _pann_model_loaded
    checkpoint_path = ensure_pann_weights()
    model = Cnn14(sample_rate=AUDIO_CONFIG['sample_rate'], window_size=1024,
                  hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
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
    """
    功能：根據檔案大小估算串流讀取的 block 長度（秒）
    為什麼：避免一次性讀入過大音訊；在記憶體與延遲間取平衡。
    """
    if file_size > (1 << 30):
        return 15.0
    if file_size > (512 << 20):
        return 30.0
    return 60.0


def log_memory(stage: str) -> None:
    """
    功能：快速追蹤流程中各階段的 RSS 記憶體；方便發現洩漏或尖峰。
    """
    print(f"[{stage}] Memory: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")


def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """
    功能：標準化波形，防止極端幅度/偏移影響特徵穩定性
    步驟：去均值 → 以最大絕對值歸一 → L2 正規化
    """
    # ──────────────── 第1階段：去均值 ────────────────
    waveform = waveform - waveform.mean()
    # ──────────────── 第2階段：幅度歸一 ────────────────
    waveform = waveform / waveform.abs().max().clamp(min=1e-6)
    # ──────────────── 第3階段：L2 正規化 ────────────────
    return waveform / (waveform.norm(p=2) + 1e-9)


def _to_path_hash(path: str) -> str:
    """功能：將路徑穩定映射為短雜湊，用於快取檔名避免過長。"""
    return sha1(path.encode('utf-8')).hexdigest()[:10]


def get_cache_path(audio_path: str) -> str:
    """功能：根據來源檔名 + 路徑雜湊組合出唯一快取檔路徑。"""
    basename = os.path.basename(audio_path)
    return os.path.join(FEATURE_CACHE_DIR, f"{basename}_{_to_path_hash(audio_path)}.npz")


def save_audio_features_to_cache(audio_path: str, features: Dict[str, any]) -> None:
    """
    功能：以檔案鎖確保快取寫入具原子性；避免多進程重複寫入/競態。
    """
    # ──────────────── 第1階段：路徑與鎖檔計算 ────────────────
    cache_path = get_cache_path(audio_path)
    lock_path = cache_path + ".lock"
    if os.path.exists(cache_path):
        return

    # ──────────────── 第2階段：原子化寫入 ────────────────
    with FileLock(lock_path):
        if os.path.exists(cache_path):
            return
        try:
            np.savez_compressed(cache_path, **features)
            logger.info(f"特徵快取儲存成功: {cache_path}")
        except Exception as e:
            logger.warning(f"儲存特徵快取失敗: {e}")


def load_audio_features_from_cache(audio_path: str) -> Optional[Dict[str, any]]:
    """
    功能：讀取並驗證快取；同時規整特徵的型別與 shape，避免之後比較出現錯誤。
    邊界：若快取內容非 dict 或解析失敗，回傳 None 以觸發重新計算。
    """
    try:
        cache_path = get_cache_path(audio_path)
        if not os.path.exists(cache_path):
            return None
        data = np.load(cache_path, allow_pickle=True)
        loaded = {k: data[k].item() if data[k].shape == () else data[k] for k in data}
        loaded = _ensure_feature_shapes(loaded)
        if not isinstance(loaded, dict):
            logger.warning(f"快取內容型別異常：{type(loaded)}，捨棄此快取")
            return None
        logger.info(f"載入特徵快取成功: {cache_path}")
        return loaded
    except Exception as e:
        logger.warning(f"載入特徵快取失敗: {e}")
        return None


def load_audio(audio_path: str) -> Generator[Tuple[np.ndarray, int], None, None]:
    """
    功能：以 librosa.stream 分塊讀取音訊，控制記憶體峰值。
    回傳：逐塊 y_block, sr；發生例外時回傳 None 結束生成器。
    """
    # ──────────────── 第1階段：估算合適的 chunk 長度 ────────────────
    file_size = os.path.getsize(audio_path)
    chunk_duration = get_optimal_chunk_size(file_size)
    try:
        # ──────────────── 第2階段：建立串流與塊迭代 ────────────────
        stream = librosa.stream(
            audio_path,
            block_length=int(chunk_duration * 22050),
            frame_length=2048,
            hop_length=1024
        )
        for y_block in stream:
            # 高記憶體壓力下，讓 GC 有機會回收並短暫退讓
            if psutil.virtual_memory().percent > 80:
                gc.collect()
                time.sleep(1)
            yield y_block, 22050
    except Exception as e:
        logger.error(f"載入音頻文件失敗 {audio_path}: {str(e)}")
        return None


def perceptual_score(sim_score: float) -> float:
    """
    功能：感知映射，將線性相似度壓縮為人耳更直覺的曲線（gamma 隨相似度自適應）。
    """
    gamma = 1.2 + 1.0 * (1 - sim_score)
    return float(min(max(sim_score ** gamma, 0.0), 1.0))


def fit_pca_if_needed(name: str, data: np.ndarray, n_components: int) -> Optional[PCA]:
    """
    功能：若尚無對應 PCA，則以合併資料進行擬合並快取。
    邊界：樣本數或維度不足時不擬合（回傳 None）。
    """
    p = _pca_registry.get(name)
    if p is not None:
        return p
    n_samples, dim = data.shape
    n_components = min(n_components, n_samples, dim)
    if n_components < 2:
        return None
    pca = PCA(n_components=n_components)
    pca.fit(data)
    _pca_registry.set(name, pca)
    return pca


def apply_pca(name: str, vector: np.ndarray, n_components: int) -> np.ndarray:
    """
    功能：將向量用已擬合的 PCA 做降維，若不存在則回傳原向量。
    """
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    pca = _pca_registry.get(name)
    if pca is None:
        return vector.squeeze()
    reduced = pca.transform(vector)
    return reduced.squeeze()


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """回傳 [0,1] 的純量餘弦相似度；先攤平成 1D 並對齊長度。"""
    a = np.ravel(a).astype(np.float64)
    b = np.ravel(b).astype(np.float64)
    min_len = min(a.size, b.size)
    if min_len == 0:
        return 0.0
    a = a[:min_len]
    b = b[:min_len]
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    cos = float(np.dot(a, b) / denom)
    return float((cos + 1.0) / 2.0)


def dtw_sim(a: np.ndarray, b: np.ndarray, max_length: int = 500) -> float:
    """
    功能：以 DTW 比較兩組 1D 時序（截斷至 max_length），回傳 [0,1] 相似度。
    邊界：空向量直接回 0；距離做 1/(1+d) 線性壓縮。
    """
    a = np.ravel(a)[:max_length]
    b = np.ravel(b)[:max_length]
    if a.size == 0 or b.size == 0:
        return 0.0
    cost = librosa.sequence.dtw(X=a.reshape(1, -1), Y=b.reshape(1, -1), metric='euclidean')[0]
    return float(1.0 / (1.0 + cost[-1, -1] / len(a)))


def chamfer_sim(a: np.ndarray, b: np.ndarray, top_k: int = 3) -> float:
    """a, b: (n_chunk, dim) -> [0,1]
    功能：對 chunk-wise 向量使用餘弦相似度的雙向最近鄰均值，近似 Chamfer 距離。
    """
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    if not isinstance(b, np.ndarray):
        b = np.asarray(b)
    if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
        return 0.0
    S = cosine_similarity(a, b)  # (na, nb)
    topA = np.mean(np.sort(S, axis=1)[:, -top_k:], axis=1)   # A→B
    topB = np.mean(np.sort(S, axis=0)[-top_k:, :], axis=0)   # B→A
    return float((topA.mean() + topB.mean()) / 2.0)


def _as_float(x: Any) -> float:
    """把各種 ndarray / list / scalar 轉為 float；用 mean 壓成純量。"""
    try:
        arr = np.asarray(x)
        if arr.ndim == 0:
            return float(arr)
        return float(arr.mean())
    except Exception:
        try:
            return float(x)  # 退而求其次
        except Exception:
            return 0.0


def _ensure_feature_shapes(feats: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：規整各種特徵的型別/shape，避免後續比較時出現布林歧義或異形狀。
    說明：對常見鍵位（onset_env/tempo/統計特徵/DL/PANN/OpenL3）做型別校正。
    """
    if feats is None:
        return {}
    out: Dict[str, Any] = dict(feats)

    # onset_env -> 1D float
    if 'onset_env' in out:
        out['onset_env'] = np.asarray(out['onset_env'], dtype=np.float32).reshape(-1)

    # tempo -> dict(mean/std/range)
    if 'tempo' in out:
        t = out['tempo']
        if isinstance(t, dict):
            out['tempo'] = {
                'mean': float(t.get('mean', 0.0)),
                'std': float(t.get('std', 0.0)),
                'range': float(t.get('range', 0.0)),
            }
        else:
            out['tempo'] = {'mean': float(t), 'std': 0.0, 'range': 0.0}

    # 統計特徵
    for name in ['mfcc', 'mfcc_delta', 'chroma', 'mel']:
        if name in out and isinstance(out[name], dict):
            for stat in ['mean', 'std', 'max', 'min', 'median']:
                if stat in out[name]:
                    out[name][stat] = np.asarray(out[name][stat], dtype=np.float32).reshape(-1)

    # dl_features: chunk-wise (n,128) 或 (128,) -> (n,128)
    if 'dl_features' in out:
        arr = np.asarray(out['dl_features'], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out['dl_features'] = arr

    # pann_features: 通常 (n,2575)；允許 (2575,)
    if 'pann_features' in out:
        arr = np.asarray(out['pann_features'], dtype=np.float32)
        out['pann_features'] = arr

    # openl3_features: 允許 dict 或 ndarray
    if 'openl3_features' in out:
        v = out['openl3_features']
        if isinstance(v, dict):
            if 'merged' in v:
                out['openl3_features']['merged'] = np.asarray(v['merged'], dtype=np.float32).reshape(-1)
            if 'chunkwise' in v:
                arr = np.asarray(v['chunkwise'], dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                out['openl3_features']['chunkwise'] = arr
        else:
            # 舊快取或簡化版：直接視為 merged 向量
            out['openl3_features'] = {
                'merged': np.asarray(v, dtype=np.float32).reshape(-1)
            }

    return out

# =============== 特徵擷取 ===============
# 功能：從原始音訊萃取多種特徵（DL/PANN/OpenL3/統計），並盡量容錯以保流程穩定。


def extract_dl_features(audio_path: str, chunk_sec: float = 10.0) -> Optional[np.ndarray]:
    """
    功能：以 Mel 頻譜 + 簡單池化取得學習式表徵（128 維），分段處理以兼顧長音檔。
    回傳：(n_chunk, 128) 或 None
    """
    try:
        # ──────────────── 第1階段：載入音訊與參數準備 ────────────────
        y, sr = librosa.load(audio_path, sr=AUDIO_CONFIG['sample_rate'])
        mel_spec_transform = get_mel_transform(sr)
        chunk_size = int(chunk_sec * sr)
        feats: List[np.ndarray] = []

        # ──────────────── 第2階段：逐段特徵提取 ────────────────
        for i in range(0, len(y), chunk_size):
            chunk = y[i:i + chunk_size]
            if len(chunk) < sr:
                continue
            try:
                wf = normalize_waveform(torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device))
                with torch.no_grad():
                    mel = mel_spec_transform(wf)  # (1, n_mels, n_frame)
                    pooled = torch.mean(mel, dim=2).squeeze().detach().cpu().numpy()
                feats.append(pooled.astype(np.float32))  # 128
            except Exception as e:
                logger.warning(f"[DL] chunk {i//chunk_size} failed: {e}")
                continue

        # ──────────────── 第3階段：彙整與回傳 ────────────────
        if len(feats) == 0:
            logger.warning(f"[DL] 全部 chunk 失敗: {audio_path}")
            return None

        return np.stack(feats).astype(np.float32)  # (n_chunk, 128)
    except Exception as e:
        logger.error(f"[DL] feature error: {e}")
        return None


def extract_pann_features(audio_path: str, chunk_sec: float = 10.0) -> Optional[np.ndarray]:
    """
    功能：使用 PANN(Cnn14) 取得 (embedding 2048 + tags 527) 的 2575 維表徵；分段處理。
    回傳：(n_chunk, 2575) 或 None
    """
    try:
        # ──────────────── 第1階段：載入與重採樣 ────────────────
        waveform, sr = torchaudio.load(audio_path)
        if sr != AUDIO_CONFIG['sample_rate']:
            waveform = torchaudio.transforms.Resample(sr, AUDIO_CONFIG['sample_rate'])(waveform)
            sr = AUDIO_CONFIG['sample_rate']

        # ──────────────── 第2階段：模型初始化 ────────────────
        model = get_pann_model()
        if model is None:
            logger.error("[PANN] 模型初始化失敗")
            return None

        # ──────────────── 第3階段：切塊並前向運算 ────────────────
        chunk_size = int(chunk_sec * sr)
        chunks = [waveform[:, i:i + chunk_size]
                  for i in range(0, waveform.shape[1], chunk_size)
                  if waveform[:, i:i + chunk_size].shape[1] >= sr]

        feats: List[np.ndarray] = []
        for idx, c in enumerate(chunks):
            try:
                with torch.no_grad():
                    out = model(c.to(device))
                    emb = out['embedding'].squeeze().detach().cpu().numpy()[:2048].astype(np.float32)   # 2048
                    tags = out['clipwise_output'].squeeze().detach().cpu().numpy().astype(np.float32)    # 527
                feats.append(np.concatenate([emb, tags]).astype(np.float32))  # 2575
            except Exception as e:
                logger.warning(f"[PANN] chunk {idx} failed: {e}")

        # ──────────────── 第4階段：彙整與回傳 ────────────────
        if len(feats) == 0:
            logger.warning(f"[PANN] 沒有有效 chunk: {audio_path}")
            return None

        return np.stack(feats).astype(np.float32)  # (n_chunk, 2575)
    except Exception as e:
        logger.error(f"[PANN] 特徵提取失敗: {e}")
        return None


def extract_openl3_features(audio_path: str, chunk_sec: float = 10.0) -> Optional[Dict[str, np.ndarray]]:
    """
    功能：以 OpenL3 取得 512 維表徵（chunkwise + merged: mean|var），對多聲道做混合單聲道。
    回傳：{"merged": (1024,), "chunkwise": (n,512)} 或 None
    """
    try:
        # ──────────────── 第1階段：載入與聲道處理 ────────────────
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

        # ──────────────── 第2階段：靜音/長度檢查 ────────────────
        if np.max(np.abs(audio)) < 1e-5 or audio.shape[1] < sr:
            logger.warning(f"音訊近乎靜音或長度不足，跳過：{audio_path}")
            return None

        # ──────────────── 第3階段：逐段嵌入提取 ────────────────
        model = get_openl3_model()
        emb_list: List[np.ndarray] = []
        chunk_size = int(chunk_sec * sr)

        for i in range(0, audio.shape[1], chunk_size):
            chunk = audio[:, i:i + chunk_size]
            if chunk.shape[1] < 4800:
                continue
            try:
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32).to(device)
                with torch.no_grad():
                    emb, _ = torchopenl3.get_audio_embedding(
                        chunk_tensor, sr, model=model,
                        hop_size=1.0, center=True, verbose=False
                    )
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                if emb.ndim == 3 and emb.shape[2] == 512:
                    emb = emb[0]
                if emb.ndim == 2 and emb.shape[1] == 512:
                    emb_list.append(np.mean(emb, axis=0).astype(np.float32))
                else:
                    logger.warning(f"OpenL3 特徵 shape 不一致：{emb.shape}")
            except Exception as sub_e:
                logger.warning(f"OpenL3 子段錯誤: {sub_e}")
                continue

        # ──────────────── 第4階段：彙整與回傳 ────────────────
        if len(emb_list) == 0:
            logger.warning(f"OpenL3 全部段落提取失敗：{audio_path}")
            return None

        emb_array = np.stack(emb_list).astype(np.float32)
        gpu_manager.clear_gpu_memory()
        return {
            "merged": np.concatenate([emb_array.mean(axis=0), emb_array.var(axis=0)]).astype(np.float32),
            "chunkwise": emb_array.astype(np.float32)
        }
    except Exception as e:
        logger.warning(f"OpenL3 error: {e}")
        return None


def extract_statistical_features(audio_path: str, chunk_sec: float = 10.0) -> Optional[Dict[str, Any]]:
    """
    功能：擷取 MFCC/Mel/Chroma/Onset/Tempo 等統計特徵；對每段計算後取平均彙整。
    回傳：dict（含各統計量與節奏資訊）或 None
    """
    try:
        # ──────────────── 第1階段：讀檔與分段 ────────────────
        y, sr = librosa.load(audio_path, sr=AUDIO_CONFIG['sample_rate'])
        chunk_size = int(chunk_sec * sr)
        features: List[Dict[str, Any]] = []

        # ──────────────── 第2階段：逐段特徵統計 ────────────────
        for i in range(0, len(y), chunk_size):
            seg = y[i:i + chunk_size]
            if len(seg) < sr:
                continue
            mel = librosa.feature.melspectrogram(y=seg, sr=sr, **FEATURE_CONFIG['mel'])
            mfcc = librosa.feature.mfcc(y=seg, sr=sr, **FEATURE_CONFIG['mfcc'])
            delta = librosa.feature.delta(mfcc)
            chroma = librosa.feature.chroma_stft(y=seg, sr=sr, **FEATURE_CONFIG['chroma'])
            onset = librosa.onset.onset_strength(y=seg, sr=sr)
            tempos = tempo(onset_envelope=onset, sr=sr, aggregate=None)
            tempo_mean = float(np.mean(tempos)) if len(tempos) else 0.0
            tempo_std = float(np.std(tempos)) if len(tempos) else 0.0
            tempo_range = float(np.max(tempos) - np.min(tempos)) if len(tempos) else 0.0

            def stats(x: np.ndarray) -> Dict[str, np.ndarray]:
                return {
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

        # ──────────────── 第3階段：多段合併 ────────────────
        return combine_features(features)
    except Exception as e:
        logger.error(f"Stat feature error: {e}")
        return None


def combine_features(features: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    功能：將多段統計特徵聚合成單一結構（平均/串接等策略）。
    回傳：聚合後的 dict 或 None
    """
    if len(features) == 0:
        return None
    combined: Dict[str, Any] = {}
    for key in features[0]:
        if key == 'onset_env':
            combined[key] = np.concatenate([f[key] for f in features]).astype(np.float32)
        elif key == 'tempo':
            if isinstance(features[0][key], dict):
                combined[key] = {
                    subkey: float(np.mean([f[key][subkey] for f in features]))
                    for subkey in features[0][key]
                }
            else:
                combined[key] = float(np.mean([f[key] for f in features]))
        else:
            combined[key] = {stat: np.mean([f[key][stat] for f in features], axis=0).astype(np.float32)
                             for stat in features[0][key]}
    return combined


def chunkwise_dtw_sim(chunk1: np.ndarray, chunk2: np.ndarray, n_components: int = 32) -> float:
    """
    功能：對兩組 chunk-wise 嵌入做 PCA 降維後，以 DTW 比較序列相似度。
    邊界：長度不足時改以 cos_sim 退化；完全空則回 0。
    """
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
    combined = np.vstack([chunk1, chunk2])
    fit_pca_if_needed('openl3_chunkwise', combined, n_components=n_components)
    r1 = apply_pca('openl3_chunkwise', chunk1, n_components=n_components)
    r2 = apply_pca('openl3_chunkwise', chunk2, n_components=n_components)
    if r1.ndim == 1 or r2.ndim == 1:
        return cos_sim(r1, r2)
    cost = librosa.sequence.dtw(X=r1.T, Y=r2.T, metric='euclidean')[0]
    dtw_dist = cost[-1, -1]
    return float(1.0 / (1.0 + dtw_dist / len(r1)))

# =============== 主流程 ===============
# 功能：特徵計算、相似度融合與對外 API。


def compute_audio_features(audio_path: str, use_openl3: bool = True) -> Optional[Dict[str, Any]]:
    """
    功能：計算單檔音訊的多模特徵並合併（含快取）
    流程：讀取快取 → 併行提取（統計/深度）→ 正規化/校正 → 寫入快取 → 回傳
    回傳：特徵 dict 或 None
    """
    # ──────────────── 第1階段：快取檢查 ────────────────
    cached = load_audio_features_from_cache(audio_path)
    if cached is not None:
        return cached

    # ──────────────── 第2階段：並行提取 ────────────────
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_stat = pool.submit(extract_statistical_features, audio_path)
        f_deep = pool.submit(lambda: {
            'dl_features': extract_dl_features(audio_path),
            'pann_features': extract_pann_features(audio_path),
            'openl3_features': extract_openl3_features(audio_path) if use_openl3 else None
        })
        stat_feat = f_stat.result()
        deep_feat = f_deep.result()

    # ──────────────── 第3階段：結果檢查 ────────────────
    if stat_feat is None:
        logger.warning(f"Statistical 特徵提取失敗: {audio_path}")
        return None

    for k in ['dl_features', 'pann_features', 'openl3_features']:
        if deep_feat.get(k) is None:
            logger.warning(f"Deep 特徵缺失：{k} 在 {audio_path}")

    # ──────────────── 第4階段：合併與型別校正 ────────────────
    features: Dict[str, Any] = {**stat_feat, **{k: v for k, v in deep_feat.items() if v is not None}}
    features = _ensure_feature_shapes(features)

    if not isinstance(features, dict):
        logger.error(f"特徵合併結果型別異常：{type(features)}")
        return None

    # ──────────────── 第5階段：寫入快取 ────────────────
    save_audio_features_to_cache(audio_path, features)
    return features


def compute_similarity(f1: Dict[str, Any], f2: Dict[str, Any]) -> float:
    """
    功能：融合多種特徵的相似度，並以預設權重加權成單一分數。
    流程：onset → 統計特徵 → tempo → OpenL3 chunkwise → DL/PANN/merged → 加權平均
    回傳：0~1 的整體相似度
    """
    detailed: Dict[str, Tuple[float, float]] = {}
    scores: List[float] = []
    weights: List[float] = []

    # ──────────────── 第1階段：onset DTW ────────────────
    if 'onset_env' in f1 and 'onset_env' in f2:
        dtw_score = _as_float(dtw_sim(f1['onset_env'], f2['onset_env']))
        scores.append(dtw_score)
        weights.append(SIMILARITY_WEIGHTS['onset_env'])
        detailed['onset_env'] = (dtw_score, SIMILARITY_WEIGHTS['onset_env'])

    # ──────────────── 第2階段：統計特徵（MFCC/MFCCΔ/Chroma） ────────────────
    for k in ['mfcc', 'mfcc_delta', 'chroma']:
        for stat in ['mean', 'std']:
            if k in f1 and k in f2 and isinstance(f1[k], dict) and isinstance(f2[k], dict):
                if stat in f1[k] and stat in f2[k]:
                    raw = cos_sim(f1[k][stat], f2[k][stat])
                    sim = _as_float(raw ** 2)
                    weight = SIMILARITY_WEIGHTS.get(f"{k}_{stat}", SIMILARITY_WEIGHTS.get(k, 1.0))
                    scores.append(sim)
                    weights.append(weight)
                    detailed[f"{k}_{stat}"] = (sim, weight)

    # ──────────────── 第3階段：節奏（tempo） ────────────────
    if 'tempo' in f1 and 'tempo' in f2:
        t1, t2 = f1['tempo'], f2['tempo']
        s1 = 1.0 / (1.0 + abs(float(t1['mean']) - float(t2['mean'])) / 30.0)
        s2 = 1.0 / (1.0 + abs(float(t1['std']) - float(t2['std'])) / 15.0)
        s3 = 1.0 / (1.0 + abs(float(t1['range']) - float(t2['range'])) / 30.0)
        sim = _as_float(0.5 * s1 + 0.25 * s2 + 0.25 * s3)
        scores.append(sim)
        weights.append(SIMILARITY_WEIGHTS['tempo'])
        detailed['tempo'] = (sim, SIMILARITY_WEIGHTS['tempo'])

    # ──────────────── 第4階段：OpenL3 chunkwise DTW ────────────────
    if 'openl3_features' in f1 and 'openl3_features' in f2:
        o1, o2 = f1['openl3_features'], f2['openl3_features']
        has_c1 = isinstance(o1, dict) and ('chunkwise' in o1)
        has_c2 = isinstance(o2, dict) and ('chunkwise' in o2)
        if has_c1 and has_c2:
            sim = _as_float(chunkwise_dtw_sim(o1['chunkwise'], o2['chunkwise']))
            scores.append(sim)
            weights.append(SIMILARITY_WEIGHTS.get('openl3_chunkwise', 1.0))
            detailed['openl3_chunkwise'] = (sim, SIMILARITY_WEIGHTS.get('openl3_chunkwise', 1.0))

    # ──────────────── 第5階段：DL / PANN / OpenL3 merged ────────────────
    for k in ['dl_features', 'pann_features', 'openl3_features']:
        v1, v2 = f1.get(k), f2.get(k)
        if v1 is None or v2 is None:
            continue
        try:
            if k == 'openl3_features':
                if isinstance(v1, dict) and isinstance(v2, dict):
                    sim_list: List[float] = []
                    if 'merged' in v1 and 'merged' in v2:
                        sim_list.append(_as_float(cos_sim(v1['merged'], v2['merged'])))
                    if 'chunkwise' in v1 and 'chunkwise' in v2:
                        sim_list.append(_as_float(chunkwise_dtw_sim(v1['chunkwise'], v2['chunkwise'])))
                    if len(sim_list) == 0:
                        continue
                    sim = float(np.mean(sim_list))
                else:
                    sim = _as_float(cos_sim(np.asarray(v1), np.asarray(v2)))
            elif k == 'pann_features':
                v1 = np.asarray(v1)
                v2 = np.asarray(v2)
                if v1.ndim == 1 and v1.shape[0] >= 2575:
                    split = 2048
                    emb1, tag1 = v1[:split], v1[split:]
                    emb2, tag2 = v2[:split], v2[split:]
                    sim = _as_float(0.6 * cos_sim(emb1, emb2) + 0.4 * cos_sim(tag1, tag2))
                elif v1.ndim == 2 and v2.ndim == 2:
                    sim = _as_float(chamfer_sim(v1, v2, top_k=3))
                else:
                    sim = _as_float(cos_sim(v1, v2))
            else:  # dl_features
                v1 = np.asarray(v1)
                v2 = np.asarray(v2)
                if v1.ndim == 2 and v2.ndim == 2:
                    sim = _as_float(chamfer_sim(v1, v2, top_k=3))
                else:
                    sim = _as_float(cos_sim(v1, v2))

            weight = float(SIMILARITY_WEIGHTS.get(k, 1.0))
            scores.append(sim)
            weights.append(weight)
            detailed[k] = (sim, weight)
        except Exception as e:
            logger.warning(f"比對 {k} 發生錯誤: {e}")

    # ──────────────── 第6階段：加權與回傳 ────────────────
    if len(scores) == 0:
        logger.error("所有相似度評估皆失敗，無法進行加權")
        return 0.0

    final = float(np.average(np.asarray(scores, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))
    for name, (s, w) in detailed.items():
        logger.info(f"  {name:20s} | 相似度: {s:.4f} | 權重: {w}")
    return final


def audio_similarity(path1: str, path2: str) -> float:
    """
    功能：對外 API；輸入兩個音訊路徑，回傳感知映射後的音訊相似度。
    流程：記憶體 log → 特徵擷取 → 相似度融合 → 感知映射 → 回傳
    邊界：任何例外皆捕捉並回傳 0.0，同時輸出 traceback 便於除錯。
    """
    try:
        # ──────────────── 第1階段：資源觀測 ────────────────
        log_memory("開始")

        # ──────────────── 第2階段：特徵擷取 ────────────────
        f1 = compute_audio_features(path1)
        f2 = compute_audio_features(path2)
        log_memory("完成")

        if not isinstance(f1, dict) or not isinstance(f2, dict):
            logger.error(f"音頻特徵型別錯誤: f1={type(f1)}, f2={type(f2)}")
            return 0.0

        # ──────────────── 第3階段：融合與感知映射 ────────────────
        raw_audio_similarity = compute_similarity(f1, f2)
        adjusted_audio_similarity = perceptual_score(raw_audio_similarity)
        logger.info(f"音頻原始相似度: {raw_audio_similarity:.4f} -> 感知相似度: {adjusted_audio_similarity:.4f}")
        return adjusted_audio_similarity

    except Exception as e:
        # ──────────────── 第4階段：錯誤處理 ────────────────
        # 關鍵：印出完整 traceback 定位是哪一行觸發
        tb = traceback.format_exc()
        logger.error(f"計算音頻相似度時出錯: {e}\n{tb}")
        return 0.0

# =============== 視需要：音訊擷取 ===============
# 功能：從影片檔萃取單聲道 WAV；提供 CLI/前處理使用。


def extract_audio(video_path: str) -> str:
    """
    功能：以 ffmpeg 從影片提取音訊，輸出到同層的 audio/ 目錄。
    邊界：檢查路徑存在與寫入權限，並多次輪詢確認輸出檔存在且非空。
    回傳：生成的 .wav 路徑；發生錯誤時丟出例外。
    """
    try:
        # ──────────────── 第1階段：輸入檢查 ────────────────
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

        # ──────────────── 第2階段：快取命中 ────────────────
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            logger.info(f"音訊檔案已存在: {audio_path}")
            return audio_path

        # ──────────────── 第3階段：ffmpeg 提取 ────────────────
        logger.info(f"開始提取音訊: {video_path} -> {audio_path}")
        ffmpeg.input(video_path).output(
            audio_path,
            acodec=AUDIO_CONFIG['codec'],
            ac=AUDIO_CONFIG['channels'],
            ar=AUDIO_CONFIG['sample_rate'],
            format=AUDIO_CONFIG['format'],
            audio_bitrate=AUDIO_CONFIG['audio_bitrate']
        ).overwrite_output().run(quiet=True, capture_stdout=True, capture_stderr=True)

        # ──────────────── 第4階段：輪詢確認輸出 ────────────────
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
