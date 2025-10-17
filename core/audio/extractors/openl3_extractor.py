"""
OpenL3 特徵提取器

此模組提供 OpenL3 音訊特徵提取功能，包括：
- OpenL3 模型載入與管理
- 音訊特徵提取
- 模型快取機制
"""

import torch
import torchopenl3
import numpy as np
from functools import lru_cache
from typing import List, Optional, Tuple, Dict
from utils.logger import logger
from utils.gpu_utils import gpu_manager

# =============== 全局變數 ===============
openl3_model: Optional[torch.nn.Module] = None


@lru_cache(maxsize=1)
def get_openl3_model():
    """載入並快取 OpenL3 模型。

    功能：
        - 載入 OpenL3 音訊嵌入模型
        - 使用 LRU 快取避免重複載入
        - 支援 GPU 加速和 DataParallel
        - 設定為評估模式

    Returns:
        torch.nn.Module: OpenL3 模型實例

    Note:
        - 使用 mel128 配置和 music 模式
        - 輸出維度為 512
        - 會自動檢測並使用 GPU 加速
    """
    global openl3_model
    # ──────────────── 第1階段：快取檢查 ────────────────
    if openl3_model is None:
        # ──────────────── 第2階段：載入與上裝置 ────────────────
        openl3_model = torchopenl3.models.load_audio_embedding_model("mel128", "music", 512).to(gpu_manager.get_device())
        # ──────────────── 第3階段：視情包裝資料並列 ────────────────
        if gpu_manager.get_device().type == 'cuda':
            openl3_model = torch.nn.DataParallel(openl3_model)
        # ──────────────── 第4階段：切換推論模式 ────────────────
        openl3_model.eval()
    return openl3_model


def _load_mono_resample48k(path: str) -> Tuple[np.ndarray, int]:
    """載入音訊並重採樣至 48kHz 單聲道。

    功能：
        - 載入音訊檔案並轉換為單聲道
        - 重採樣至 48kHz 以符合 OpenL3 要求
        - 多聲道音訊會以均值混合
        - 確保與 OpenL3 模型相容

    Args:
        path (str): 音訊檔案路徑

    Returns:
        Tuple[np.ndarray, int]: (音訊資料, 取樣率)

    Note:
        - 會自動處理多聲道轉單聲道
        - 確保取樣率為 48kHz
    """
    import librosa
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
    """將長序列切分為多個片段。

    功能：
        - 將長音訊序列切分為多個片段
        - 過濾掉過短的片段
        - 確保每個片段都有足夠的長度

    Args:
        y (np.ndarray): 音訊序列
        chunk (int): 片段長度
        min_len (int): 最小片段長度

    Returns:
        List[np.ndarray]: 有效的音訊片段列表
    """
    return [y[i:i + chunk] for i in range(0, y.shape[0], chunk)
            if y[i:i + chunk].shape[0] >= min_len]


def _openl3_embed_chunk(model, seg: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """對單一片段產生 OpenL3 嵌入。

    功能：
        - 對單一音訊片段產生 OpenL3 嵌入
        - 進行時間平均得到 512 維特徵
        - 處理各種輸出形狀
        - 處理推論錯誤

    Args:
        model: OpenL3 模型
        seg (np.ndarray): 音訊片段
        sr (int): 取樣率

    Returns:
        Optional[np.ndarray]: 512 維嵌入向量，失敗時回傳 None

    Note:
        - 會處理 3D 和 2D 輸出形狀
        - 推論失敗會記錄警告
    """
    try:
        # ──────────────── 第1階段：Tensor 化並上裝置 ────────────────
        t = torch.tensor(seg[None, :], dtype=torch.float32).to(gpu_manager.get_device())
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
    """提取 OpenL3 特徵。

    功能：
        - 使用 OpenL3 模型提取音訊特徵
        - 產生 merged 和 chunkwise 兩種特徵
        - 處理靜音和長度不足的情況
        - 自動清理 GPU 記憶體

    Args:
        audio_path (str): 音訊檔案路徑
        chunk_sec (float, optional): 片段長度（秒）。預設為 10.0。

    Returns:
        Optional[Dict[str, np.ndarray]]: OpenL3 特徵字典，失敗時回傳 None

    Note:
        - merged: mean+var(512) → 1024 維
        - chunkwise: (N, 512) 形狀
        - 會跳過靜音或過短的音訊
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
