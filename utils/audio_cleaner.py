# utils/audio_cleaner.py
"""
檔案用途：音訊清理與預處理工具

此模組提供音訊檔案清理功能，包括：
- 靜音檢測與修剪
- 音訊格式標準化（單聲道、指定取樣率）
- 音量正規化
- 音訊品質優化

主要功能：
- trim_silence_pydub: 使用 pydub 進行靜音修剪
- load_and_clean_audio: 完整的音訊清理流程
"""

import os
import torch
import torchaudio
import numpy as np
from typing import Optional
from pydub import AudioSegment, silence
import torchaudio.transforms as transform

# =============== 靜音修剪工具 ===============


def trim_silence_pydub(waveform: torch.Tensor, sr: int, min_silence_len: int = 300,
                       silence_thresh_db: int = -40) -> torch.Tensor:
    """使用 pydub 根據靜音切割音訊並合併語音段，避免語句開頭/結尾被剪掉。

    功能：
        - 將 PyTorch tensor 轉換為 pydub AudioSegment
        - 使用 silence.split_on_silence 檢測並分割靜音段落
        - 合併非靜音段落，保留少量前後緩衝

    Args:
        waveform (torch.Tensor): 音訊波形資料 (1, samples)
        sr (int): 取樣率
        min_silence_len (int, optional): 最小靜音長度（毫秒）。預設為 300。
        silence_thresh_db (int, optional): 靜音閾值（分貝）。預設為 -40。

    Returns:
        torch.Tensor: 清理後的音訊波形，保持原始維度 (1, samples)

    Note:
        - 如果無法分割出有效段落，會回傳原始音訊
        - 保留 100ms 的前後語音緩衝以避免切掉重要內容
    """
    samples: np.ndarray = waveform.squeeze().numpy()
    audio: AudioSegment = AudioSegment(
        samples.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )

    chunks: list[AudioSegment] = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh_db,
        keep_silence=100  # 保留一點前後語音緩衝
    )

    if not chunks:
        return waveform  # 如果切不出來，保留原始音訊

    combined: AudioSegment = sum(chunks)
    cleaned: np.ndarray = np.array(combined.get_array_of_samples()).astype(np.float32) / 32768.0
    return torch.tensor(cleaned).unsqueeze(0)

# =============== 音訊清理流程 ===============


def load_and_clean_audio(audio_path: str, output_path: Optional[str] = None,
                         sample_rate: int = 16000, use_silence_detection: bool = False,
                         min_silence_len: int = 300, silence_thresh_db: int = -40) -> str:
    """清理音訊並轉為 Whisper 最佳格式：單聲道、指定取樣率、正規化音量、去除靜音。

    功能：
        - 載入音訊檔案並轉換為 PyTorch tensor
        - 轉換為單聲道（如果是多聲道）
        - 重取樣至指定取樣率（預設 16kHz，適合 Whisper）
        - 音量正規化至 [-1, 1] 範圍
        - 可選的靜音檢測與修剪
        - 保存為標準 WAV 格式

    Args:
        audio_path (str): 輸入音訊檔案路徑
        output_path (Optional[str], optional): 輸出檔案路徑。預設為 None（自動生成）。
        sample_rate (int, optional): 目標取樣率。預設為 16000。
        use_silence_detection (bool, optional): 是否啟用靜音檢測。預設為 False。
        min_silence_len (int, optional): 最小靜音長度（毫秒）。預設為 300。
        silence_thresh_db (int, optional): 靜音閾值（分貝）。預設為 -40。

    Returns:
        str: 清理後的音訊檔案路徑

    Raises:
        Exception: 當音訊處理失敗時，會記錄錯誤並回傳原始路徑

    Note:
        - 輸出檔案格式為 WAV，適合後續的語音識別處理
        - 如果處理失敗，會回傳原始檔案路徑
    """
    try:
        waveform, sr = torchaudio.load(audio_path)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != sample_rate:
            resampler: transform.Resample = transform.Resample(sr, sample_rate)
            waveform = resampler(waveform)
            sr = sample_rate

        waveform = waveform / waveform.abs().max()

        if use_silence_detection:
            waveform = trim_silence_pydub(waveform, sr, min_silence_len, silence_thresh_db)

        waveform = torch.clamp(waveform, -1.0, 1.0).to(torch.float32)

        if output_path is None:
            base, _ = os.path.splitext(audio_path)
            output_path = f"{base}_clean.wav"

        torchaudio.save(output_path, waveform, sr)
        return output_path

    except Exception as e:
        print(f"[音訊清理失敗]: {str(e)}")
        return audio_path
