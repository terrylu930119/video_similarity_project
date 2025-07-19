import os
import torch
import torchaudio
import numpy as np
from typing import Optional
from pydub import AudioSegment, silence
import torchaudio.transforms as transform

# =============== 靜音修剪工具 ===============
def trim_silence_pydub(waveform: torch.Tensor, sr: int, min_silence_len: int = 300, silence_thresh_db: int = -40) -> torch.Tensor:
    """
    使用 pydub 根據靜音切割音訊並合併語音段，避免語句開頭/結尾被剪掉。
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
def load_and_clean_audio(audio_path: str, output_path: Optional[str] = None, sample_rate: int = 16000, use_silence_detection: bool = False,
                         min_silence_len: int = 300,silence_thresh_db: int = -40) -> str:
    """
    清理音訊並轉為 Whisper 最佳格式：單聲道、指定取樣率、正規化音量、去除靜音。
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
