"""
PANN 特徵提取器

此模組提供 PANN 音訊特徵提取功能，包括：
- PANN Cnn14 模型載入與管理
- 音訊特徵提取
- 模型快取機制
"""

import torch
import torchaudio
import numpy as np
from functools import lru_cache
from typing import List, Optional, Tuple
from utils.logger import logger
from utils.gpu_utils import gpu_manager
from utils.downloader import ensure_pann_weights
from panns_inference.models import Cnn14

# =============== 全局配置參數 ===============
from ..config import AUDIO_CONFIG

_pann_model_loaded = False  # PANN 權重載入一次性提示開關


@lru_cache(maxsize=1)
def get_pann_model():
    """載入並快取 PANN Cnn14 模型。

    功能：
        - 載入 PANN Cnn14 音訊分類模型
        - 使用預訓練權重進行初始化
        - 設定為評估模式
        - 確保與音訊配置的取樣率一致

    Returns:
        torch.nn.Module: PANN 模型實例

    Note:
        - 僅首次載入時會記錄成功訊息
        - 使用 32kHz 取樣率配置
        - 支援 527 個音訊事件類別
    """
    global _pann_model_loaded
    # ──────────────── 第1階段：確保權重路徑 ────────────────
    checkpoint_path = ensure_pann_weights()
    # ──────────────── 第2階段：初始化模型結構 ────────────────
    model = Cnn14(sample_rate=AUDIO_CONFIG['sample_rate'], window_size=1024,
                  hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    # ──────────────── 第3階段：載入權重（map 到目標裝置） ────────────────
    checkpoint = torch.load(checkpoint_path, map_location=gpu_manager.get_device())
    model.load_state_dict(checkpoint['model'])
    # ──────────────── 第4階段：一次性提示與狀態切換 ────────────────
    if not _pann_model_loaded:
        logger.info("PANN 模型權重載入成功")
        _pann_model_loaded = True
    # ──────────────── 第5階段：上裝置並設為 eval ────────────────
    return model.to(gpu_manager.get_device()).eval()


def _load_resampled_mono_torch(audio_path: str, target_sr: int) -> Tuple[torch.Tensor, int]:
    """使用 torchaudio 載入並重採樣音訊。

    功能：
        - 使用 torchaudio 載入音訊檔案
        - 重採樣到目標取樣率
        - 維持單聲道張量形狀 (1, T)
        - 確保與 PANN 模型相容

    Args:
        audio_path (str): 音訊檔案路徑
        target_sr (int): 目標取樣率

    Returns:
        Tuple[torch.Tensor, int]: (音訊張量, 取樣率)
    """
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        sr = target_sr
    return waveform, sr


def _split_waveform(waveform: torch.Tensor, sr: int, chunk_sec: float) -> List[torch.Tensor]:
    """將音訊張量切分為片段。

    功能：
        - 根據指定秒數切分音訊張量
        - 過濾掉過短的片段
        - 保持張量形狀 (1, T)

    Args:
        waveform (torch.Tensor): 音訊張量
        sr (int): 取樣率
        chunk_sec (float): 片段長度（秒）

    Returns:
        List[torch.Tensor]: 有效的音訊片段列表

    Note:
        - 小於 1 秒的片段不會保留
    """
    chunk_size = int(chunk_sec * sr)
    return [waveform[:, i:i + chunk_size] for i in range(0, waveform.shape[1], chunk_size)
            if waveform[:, i:i + chunk_size].shape[1] >= sr]


def _pann_embed(model: torch.nn.Module, c: torch.Tensor) -> Optional[np.ndarray]:
    """對單一片段進行 PANN 推論。

    功能：
        - 使用 PANN 模型對音訊片段進行推論
        - 提取 embedding（2048 維）和 tags（527 維）
        - 拼接成完整的特徵向量
        - 處理推論錯誤

    Args:
        model (torch.nn.Module): PANN 模型
        c (torch.Tensor): 音訊片段張量

    Returns:
        Optional[np.ndarray]: 拼接後的特徵向量（2575 維），失敗時回傳 None

    Note:
        - 輸出維度：2048（embedding）+ 527（tags）= 2575
    """
    try:
        # ──────────────── 第1階段：前向推論 ────────────────
        with torch.no_grad():
            out = model(c.to(gpu_manager.get_device()))
            emb = out['embedding'].squeeze().detach().cpu().numpy()[:2048].astype(np.float32)
            tags = out['clipwise_output'].squeeze().detach().cpu().numpy().astype(np.float32)
        # ──────────────── 第2階段：拼接輸出 ────────────────
        return np.concatenate([emb, tags]).astype(np.float32)
    except Exception as e:
        logger.warning(f"[PANN] chunk failed: {e}")
        return None


def extract_pann_features(audio_path: str, chunk_sec: float = 10.0) -> Optional[np.ndarray]:
    """提取 PANN 特徵。

    功能：
        - 使用 PANN Cnn14 模型提取音訊特徵
        - 將音訊分段進行推論
        - 組合所有有效片段的特徵
        - 處理載入和推論錯誤

    Args:
        audio_path (str): 音訊檔案路徑
        chunk_sec (float, optional): 片段長度（秒）。預設為 10.0。

    Returns:
        Optional[np.ndarray]: 特徵矩陣 (N, 2575)，失敗時回傳 None

    Note:
        - 每段輸出：2048（embedding）+ 527（tags）= 2575 維
        - 沒有任何有效片段時會回傳 None
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
