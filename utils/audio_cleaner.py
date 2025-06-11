import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pydub import AudioSegment, silence

def trim_silence_pydub(waveform: torch.Tensor, sr: int, min_silence_len=300, silence_thresh_db=-40) -> torch.Tensor:
    """
    使用 pydub 根據靜音切割音訊並合併語音段，避免語句開頭/結尾被剪掉。
    """
    samples = waveform.squeeze().numpy()
    audio = AudioSegment(
        samples.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )

    chunks = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh_db,
        keep_silence=100  # 保留一點前後語音緩衝
    )

    if not chunks:
        return waveform  # 如果切不出來，保留原始音訊

    combined = sum(chunks)
    cleaned = np.array(combined.get_array_of_samples()).astype(np.float32) / 32768.0
    return torch.tensor(cleaned).unsqueeze(0)

def load_and_clean_audio(audio_path: str, output_path: str = None, sample_rate: int = 16000,
                         use_silence_detection: bool = False, min_silence_len: int = 300, silence_thresh_db: int = -40) -> str:
    """
    清理音訊並轉為 Whisper 最佳格式：單聲道、指定取樣率、正規化音量、去除靜音。

    參數:
        audio_path: 原始音訊檔案路徑
        output_path: 處理後輸出的音訊檔案路徑（若為 None 則自動生成 *_clean.wav）
        sample_rate: Whisper/FasterWhisper 最佳推薦取樣率（預設為 16000）
        use_silence_detection: 是否使用靜音檢測（預設為 True）
        min_silence_len: 靜音長度判斷門檻（ms）
        silence_thresh_db: 靜音音量閾值（dB）

    返回:
        處理後音訊的儲存路徑
    """
    try:
        waveform, sr = torchaudio.load(audio_path)

        # 轉為單聲道
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 重取樣
        if sr != sample_rate:
            resampler = T.Resample(sr, sample_rate)
            waveform = resampler(waveform)
            sr = sample_rate

        # 音量正規化
        waveform = waveform / waveform.abs().max()

        # 去除靜音段
        if use_silence_detection:
            waveform = trim_silence_pydub(waveform, sr, min_silence_len, silence_thresh_db)

        # 限制儲存格式與範圍
        waveform = torch.clamp(waveform, -1.0, 1.0).to(torch.float32)

        # 自動生成輸出檔名
        if output_path is None:
            base, _ = os.path.splitext(audio_path)
            output_path = f"{base}_clean.wav"

        torchaudio.save(output_path, waveform, sr)
        return output_path

    except Exception as e:
        print(f"[音訊清理失敗]: {str(e)}")
        return audio_path
