"""
深度學習特徵提取器

此模組提供深度學習音訊特徵提取功能，包括：
- Mel 頻譜特徵提取
- 深度學習模型載入與管理
- 特徵快取機制
"""

import torch
import torchaudio
import numpy as np
from functools import lru_cache
from typing import Optional
from utils.logger import logger
from utils.gpu_utils import gpu_manager
from ..utils import normalize_waveform

# =============== 配置參數 ===============
from ..config import AUDIO_CONFIG, FEATURE_CONFIG


@lru_cache(maxsize=3)
def get_mel_transform(sr: int):
    """建立或快取 MelSpectrogram 轉換器。

    功能：
        - 根據取樣率建立 MelSpectrogram 轉換器
        - 使用 LRU 快取避免重複建立
        - 自動掛載到適當的裝置
        - 確保與音訊配置的取樣率一致

    Args:
        sr (int): 音訊取樣率

    Returns:
        torchaudio.transforms.MelSpectrogram: MelSpectrogram 轉換器

    Note:
        - 會根據取樣率建立不同的實例
        - 已預先掛載到目標裝置
    """
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=2048, hop_length=FEATURE_CONFIG['mel']['hop_length'],
        n_mels=FEATURE_CONFIG['mel']['n_mels']
    ).to(gpu_manager.get_device())


def _extract_dl_chunk(mel_transform, chunk: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """對單一片段提取深度學習特徵。

    功能：
        - 對音訊片段進行 Mel 頻譜轉換
        - 進行時間維度的平均池化
        - 標準化波形並應用變換
        - 處理過短片段和錯誤

    Args:
        mel_transform: Mel 頻譜轉換器
        chunk (np.ndarray): 音訊片段
        sr (int): 取樣率

    Returns:
        Optional[np.ndarray]: 特徵向量，失敗時回傳 None

    Note:
        - 片段小於 1 秒時會被丟棄
        - 會進行波形標準化
    """
    if len(chunk) < sr:
        return None
    try:
        # ──────────────── 第1階段：標準化並上裝置 ────────────────
        wf = normalize_waveform(torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(gpu_manager.get_device()))
        # ──────────────── 第2階段：Mel 轉換 + 時間平均 ────────────────
        with torch.no_grad():
            mel = mel_transform(wf)
            pooled = torch.mean(mel, dim=2).squeeze().detach().cpu().numpy()
        return pooled.astype(np.float32)
    except Exception as e:
        logger.warning(f"[DL] chunk failed: {e}")
        return None


def extract_dl_features(audio_path: str, chunk_sec: float = 10.0) -> Optional[np.ndarray]:
    """提取深度學習特徵。

    功能：
        - 使用簡化的深度學習管線提取特徵
        - 基於 Mel 頻譜和時間池化
        - 將音訊分段處理
        - 組合所有有效片段的特徵

    Args:
        audio_path (str): 音訊檔案路徑
        chunk_sec (float, optional): 片段長度（秒）。預設為 10.0。

    Returns:
        Optional[np.ndarray]: 特徵矩陣 (N, D)，失敗時回傳 None

    Note:
        - 輸出形狀為 (N, D)，N 為片段數，D 為特徵維度
        - 全部片段失敗時會回傳 None
    """
    try:
        import librosa
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
