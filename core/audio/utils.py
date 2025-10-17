"""
音訊處理工具函式

此模組提供音訊處理相關的工具函式，包括：
- 音訊預處理與標準化
- 特徵快取管理
- 相似度計算工具
- 資料型別轉換
"""

import os
import gc
import time
import torch
import psutil
import numpy as np
import ffmpeg
from hashlib import sha1
from filelock import FileLock
from functools import lru_cache
from typing import Optional, Generator, Tuple, Dict, Any
from utils.logger import logger
from utils.gpu_utils import gpu_manager
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict

# =============== 全局配置參數 ===============
from .config import AUDIO_CONFIG

FEATURE_CACHE_DIR = os.path.join(os.getcwd(), "feature_cache")
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

gpu_manager.initialize()
device = gpu_manager.get_device()

# =============== PCA 快取工具 ===============

class PCACache:
    """PCA 模型快取管理器。

    功能：
        - 維護 PCA 模型快取避免重複擬合
        - 使用 LRU 策略管理快取容量
        - 自動清理過期的快取項目

    Attributes:
        cache (OrderedDict[str, PCA]): PCA 模型快取
        max_items (int): 最大快取項目數
    """

    def __init__(self, max_items: int = 15) -> None:
        self.cache: OrderedDict[str, PCA] = OrderedDict()
        self.max_items = max_items

    def get(self, name: str) -> Optional[PCA]:
        """取得名稱對應的 PCA 實例。

        Args:
            name (str): PCA 模型名稱

        Returns:
            Optional[PCA]: PCA 實例，不存在時回傳 None
        """
        return self.cache.get(name)

    def set(self, name: str, pca: PCA) -> None:
        """寫入快取並維持 LRU 策略。

        Args:
            name (str): PCA 模型名稱
            pca (PCA): PCA 實例

        Note:
            - 超出容量時會自動移除最舊的項目
            - 會記錄被移除的項目
        """
        self.cache[name] = pca
        self.cache.move_to_end(name)
        if len(self.cache) > self.max_items:
            evicted = self.cache.popitem(last=False)
            logger.info(f"自動清除 PCA: {evicted[0]}")

    def clear(self) -> None:
        """清空快取。

        Note:
            - 會移除所有快取的 PCA 模型
        """
        self.cache.clear()


_pca_registry = PCACache()


def log_memory(stage: str) -> None:
    """記錄記憶體使用量。

    功能：
        - 列印當前進程的記憶體使用量
        - 使用 RSS（常駐集大小）指標
        - 用於監控記憶體使用情況

    Args:
        stage (str): 處理階段名稱
    """
    print(f"[{stage}] Memory: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")


def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """對音訊波形進行標準化處理。

    功能：
        - 去除波形的均值偏移
        - 進行振幅正規化
        - 應用 L2 正規化
        - 降低幅度差和偏移對特徵的影響

    Args:
        waveform (torch.Tensor): 輸入音訊波形

    Returns:
        torch.Tensor: 標準化後的波形

    Note:
        - 三步驟依序進行：去均值 → 振幅歸一 → L2 正規化
        - 會避免除零錯誤
    """
    # ──────────────── 第1階段：去均值 ────────────────
    waveform = waveform - waveform.mean()
    # ──────────────── 第2階段：幅度正規化 ────────────────
    waveform = waveform / waveform.abs().max().clamp(min=1e-6)
    # ──────────────── 第3階段：L2 正規化 ────────────────
    return waveform / (waveform.norm(p=2) + 1e-9)


def _to_path_hash(path: str) -> str:
    """將檔案路徑轉換為短 SHA1 雜湊值。

    功能：
        - 將檔案路徑轉換為 SHA1 雜湊值
        - 取前 10 個字元作為短識別碼
        - 用於生成唯一的快取檔案名

    Args:
        path (str): 檔案路徑

    Returns:
        str: 10 字元的 SHA1 雜湊值
    """
    return sha1(path.encode('utf-8')).hexdigest()[:10]


def get_cache_path(audio_path: str) -> str:
    """取得音訊檔案對應的特徵快取路徑。

    功能：
        - 根據音訊檔案路徑生成快取檔案路徑
        - 使用檔案名和路徑雜湊值確保唯一性
        - 使用 .npz 格式儲存壓縮的特徵資料

    Args:
        audio_path (str): 音訊檔案路徑

    Returns:
        str: 快取檔案路徑
    """
    return os.path.join(FEATURE_CACHE_DIR, f"{os.path.basename(audio_path)}_{_to_path_hash(audio_path)}.npz")


def save_audio_features_to_cache(audio_path: str, features: Dict[str, Any]) -> None:
    """安全寫入特徵快取。

    功能：
        - 使用檔案鎖避免並行寫入競態
        - 檢查快取是否已存在，存在則跳過
        - 使用壓縮格式儲存特徵資料
        - 處理寫入錯誤並記錄警告

    Args:
        audio_path (str): 音訊檔案路徑
        features (Dict[str, Any]): 特徵資料字典

    Note:
        - 會記錄快取儲存成功或失敗的訊息
        - 寫入失敗不會拋出例外
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
    """讀取特徵快取並修正形狀和型別。

    功能：
        - 從快取檔案讀取特徵資料
        - 進行必要的形狀和型別修正
        - 處理載入錯誤
        - 確保特徵資料的完整性

    Args:
        audio_path (str): 音訊檔案路徑

    Returns:
        Optional[Dict[str, Any]]: 特徵資料字典，不存在或失敗時回傳 None

    Note:
        - 會處理單一元素的陣列
        - 會記錄載入失敗的警告
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
    """以串流方式讀取音訊。

    功能：
        - 使用串流方式讀取音訊降低記憶體使用
        - 根據檔案大小動態調整切片長度
        - 監控記憶體使用量並進行清理
        - 處理讀取錯誤

    Args:
        audio_path (str): 音訊檔案路徑

    Yields:
        Tuple[np.ndarray, int]: (音訊片段, 取樣率)

    Note:
        - 會根據檔案大小估算最佳切片長度
        - 記憶體使用超過 80% 時會進行清理
        - 讀取失敗時會記錄錯誤
    """
    # ──────────────── 第1階段：估算分段長度 ────────────────
    file_size = os.path.getsize(audio_path)
    chunk_duration = get_optimal_chunk_size(file_size)
    try:
        # ──────────────── 第2階段：建立串流迭代器 ────────────────
        import librosa
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


@lru_cache(maxsize=32)
def get_optimal_chunk_size(file_size: int) -> float:
    """根據檔案大小估算最佳切片秒數。

    功能：
        - 根據檔案大小動態調整切片長度
        - 大檔案使用較短的切片避免記憶體不足
        - 小檔案使用較長的切片減少處理開銷

    Args:
        file_size (int): 檔案大小（位元組）

    Returns:
        float: 建議的切片長度（秒）

    Note:
        - 大於 1GB：15 秒
        - 大於 512MB：30 秒
        - 其他：60 秒
    """
    if file_size > (1 << 30):
        return 15.0
    if file_size > (512 << 20):
        return 30.0
    return 60.0


def perceptual_score(sim_score: float) -> float:
    """感知再映射相似度分數。

    功能：
        - 使用動態 gamma 值進行感知再映射
        - 使高分更嚴格、低分更寬鬆
        - 提升相似度分數的感知一致性

    Args:
        sim_score (float): 原始相似度分數

    Returns:
        float: 映射後的相似度分數

    Note:
        - gamma = 1.2 + 1.0 * (1 - sim_score)
        - 結果會限制在 [0, 1] 範圍內
    """
    gamma = 1.2 + 1.0 * (1 - sim_score)
    return float(min(max(sim_score ** gamma, 0.0), 1.0))


def fit_pca_if_needed(name: str, data: np.ndarray, n_components: int) -> Optional[PCA]:
    """必要時擬合 PCA 並快取。

    功能：
        - 檢查快取中是否已有對應的 PCA 模型
        - 若無則擬合新的 PCA 模型並快取
        - 自動調整主成分數避免維度問題
        - 處理樣本數不足的情況

    Args:
        name (str): PCA 模型名稱
        data (np.ndarray): 訓練資料
        n_components (int): 期望的主成分數

    Returns:
        Optional[PCA]: PCA 模型，樣本數不足時回傳 None

    Note:
        - 主成分數會自動調整為 min(n_components, n_samples, n_features)
        - 樣本數少於 2 時會回傳 None
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
    """使用已擬合的 PCA 轉換向量。

    功能：
        - 使用快取中的 PCA 模型轉換向量
        - 若無對應模型則原樣返回
        - 自動處理 1D 向量的形狀

    Args:
        name (str): PCA 模型名稱
        vector (np.ndarray): 要轉換的向量
        n_components (int): 主成分數

    Returns:
        np.ndarray: 轉換後的向量

    Note:
        - 會自動將 1D 向量重塑為 2D
        - 轉換後會壓縮回 1D 形狀
    """
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    pca = _pca_registry.get(name)
    return (pca.transform(vector) if pca is not None else vector).squeeze()


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """計算餘弦相似度並映射到 [0,1] 範圍。

    功能：
        - 計算兩個向量的餘弦相似度
        - 將結果從 [-1,1] 映射到 [0,1]
        - 自動對齊向量長度
        - 處理空向量和除零情況

    Args:
        a (np.ndarray): 第一個向量
        b (np.ndarray): 第二個向量

    Returns:
        float: 映射後的相似度分數（0-1）

    Note:
        - 會將向量展平並轉換為 float64
        - 空向量時回傳 0.0
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
    """計算 1D DTW 距離並轉換為相似度。

    功能：
        - 使用動態時間規整計算序列距離
        - 將距離轉換為相似度分數
        - 限制序列長度避免計算爆炸
        - 處理空序列

    Args:
        a (np.ndarray): 第一個序列
        b (np.ndarray): 第二個序列
        max_length (int, optional): 最大序列長度。預設為 500。

    Returns:
        float: 相似度分數（0-1）

    Note:
        - 會將序列截斷到最大長度
        - 空序列時回傳 0.0
    """
    a = np.ravel(a)[:max_length]
    b = np.ravel(b)[:max_length]
    if a.size == 0 or b.size == 0:
        return 0.0
    import librosa
    cost = librosa.sequence.dtw(X=a.reshape(1, -1), Y=b.reshape(1, -1), metric='euclidean')[0]
    return float(1.0 / (1.0 + cost[-1, -1] / len(a)))


def chamfer_sim(a: np.ndarray, b: np.ndarray, top_k: int = 3) -> float:
    """計算類 Chamfer 距離相似度。

    功能：
        - 計算兩個 2D 集合的雙向 Top-K 相似度
        - 使用餘弦相似度矩陣
        - 取平均得到最終相似度分數
        - 適用於點雲或特徵集合比對

    Args:
        a (np.ndarray): 第一個 2D 集合
        b (np.ndarray): 第二個 2D 集合
        top_k (int, optional): Top-K 數量。預設為 3。

    Returns:
        float: 相似度分數（0-1）

    Note:
        - 僅適用於 2D 陣列
        - 空集合時回傳 0.0
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
    """寬容轉換為 float 類型。

    功能：
        - 將任意類型轉換為 float
        - 陣列類型會取平均值
        - 轉換失敗時回傳 0.0
        - 處理各種邊界情況

    Args:
        x (Any): 要轉換的值

    Returns:
        float: 轉換後的浮點數
    """
    try:
        arr = np.asarray(x)
        return float(arr if arr.ndim == 0 else arr.mean())
    except Exception:
        try:
            return float(x)
        except Exception:
            return 0.0


def _ensure_feature_shapes(feats: Dict[str, Any]) -> Dict[str, Any]:
    """對特徵字典進行型別與形狀的安全校正。

    功能：
        - 對特徵字典進行型別和形狀校正
        - 應用各種正規化函式
        - 確保特徵格式的一致性
        - 處理空輸入的情況

    Args:
        feats (Dict[str, Any]): 輸入的特徵字典

    Returns:
        Dict[str, Any]: 校正後的特徵字典

    Note:
        - 會對各種特徵類型應用對應的正規化函式
        - 空輸入會回傳空字典
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


# =============== Normalizers（集中處理形狀/型別） ===============

def normalize_onset_env(x: Any) -> np.ndarray:
    """規整 onset 強度包絡為 1D float32 陣列。

    功能：
        - 將 onset 強度包絡轉換為標準格式
        - 確保為 1D float32 陣列
        - 用於特徵標準化

    Args:
        x (Any): 輸入的 onset 強度包絡

    Returns:
        np.ndarray: 1D float32 陣列
    """
    return np.asarray(x, dtype=np.float32).reshape(-1)


def normalize_tempo(x: Any) -> Dict[str, float]:
    """將節奏資訊統一為標準格式。

    功能：
        - 將節奏資訊標準化為 {mean, std, range} 格式
        - 處理字典和非字典輸入
        - 確保所有值為 float 類型

    Args:
        x (Any): 輸入的節奏資訊

    Returns:
        Dict[str, float]: 標準化的節奏資訊字典

    Note:
        - 非字典輸入會將值填入 mean，std 和 range 為 0
    """
    if isinstance(x, dict):
        return {'mean': float(x.get('mean', 0.0)),
                'std': float(x.get('std', 0.0)),
                'range': float(x.get('range', 0.0))}
    v = float(x) if not isinstance(x, (list, np.ndarray, dict)) else _as_float(x)
    return {'mean': v, 'std': 0.0, 'range': 0.0}


def normalize_stats_block(block: Any) -> Dict[str, np.ndarray]:
    """將統計區塊整理為標準格式。

    功能：
        - 將統計特徵區塊整理為 1D 陣列字典
        - 包含 mean、std、max、min、median 等統計量
        - 確保所有陣列為 float32 類型

    Args:
        block (Any): 輸入的統計區塊

    Returns:
        Dict[str, np.ndarray]: 標準化的統計字典

    Note:
        - 非字典輸入會回傳空字典
        - 所有陣列會轉換為 1D float32 格式
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
    """將深度學習特徵轉換為 >=2D 形狀。

    功能：
        - 將深度學習特徵轉換為至少 2D 的形狀
        - 1D 陣列會重塑為 (1, D) 形狀
        - 確保與後續處理相容

    Args:
        arr (Any): 輸入的深度學習特徵

    Returns:
        np.ndarray: 轉換後的特徵陣列
    """
    arr = np.asarray(arr, dtype=np.float32)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def normalize_pann(arr: Any) -> np.ndarray:
    """將 PANN 特徵轉換為 float32 陣列。

    功能：
        - 將 PANN 特徵轉換為 float32 類型
        - 確保特徵格式的一致性
        - 用於特徵標準化

    Args:
        arr (Any): 輸入的 PANN 特徵

    Returns:
        np.ndarray: float32 格式的特徵陣列
    """
    return np.asarray(arr, dtype=np.float32)


def normalize_openl3(v: Any) -> Dict[str, np.ndarray]:
    """規整 OpenL3 特徵為標準格式。

    功能：
        - 將 OpenL3 特徵規整為標準格式
        - 包含 merged（1D）和 chunkwise（2D）子集
        - 確保所有陣列為 float32 類型

    Args:
        v (Any): 輸入的 OpenL3 特徵

    Returns:
        Dict[str, np.ndarray]: 標準化的 OpenL3 特徵字典

    Note:
        - 非字典輸入僅會輸出 merged 單向量
        - chunkwise 會確保為 2D 形狀
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


def extract_audio(video_path: str) -> str:
    """從影片中萃取音訊為 WAV（單聲道/32kHz/PCM16）。

    功能：
        - 從影片檔案中提取音訊
        - 轉換為標準格式（單聲道/32kHz/PCM16）
        - 提供快取機制
        - 具備重試機制

    Args:
        video_path (str): 影片檔案路徑

    Returns:
        str: 音訊檔案路徑

    Raises:
        FileNotFoundError: 影片檔案不存在
        PermissionError: 沒有輸出目錄的寫入權限
        RuntimeError: 音訊檔案生成失敗

    Note:
        - 具備重試機制
        - 音訊快取存在則直接返回
        - 無寫入權限或產出失敗會擲出例外
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
