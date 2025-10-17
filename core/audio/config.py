"""
音訊處理配置模組

此模組提供音訊處理相關的統一配置參數，避免重複定義。
"""

# =============== 音訊配置 ===============
AUDIO_CONFIG = {
    'sample_rate': 32000,
    'channels': 1,
    'audio_bitrate': '192k',
    'format': 'wav',
    'codec': 'pcm_s16le',
    'force_gpu': True
}

# =============== 特徵配置 ===============
FEATURE_CONFIG = {
    'mfcc': {'n_mfcc': 13, 'hop_length': 1024},
    'mel': {'n_mels': 64, 'hop_length': 1024},
    'chroma': {'n_chroma': 12, 'hop_length': 1024}
}

# =============== 相似度權重 ===============
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

# =============== 其他配置 ===============
THREAD_CONFIG = {'max_workers': 6}
CROP_CONFIG = {'min_duration': 30.0, 'max_duration': 300.0, 'overlap': 0.5, 'silence_threshold': -14}
