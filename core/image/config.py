"""
影像處理配置模組

此模組統一管理影像處理相關的所有配置參數，包括：
- 採樣參數
- pHash 特徵提取參數
- DTW 對齊參數
- 時間聚合參數
- 相似度計算權重
- 批次處理參數
"""

# =============== 採樣配置 ===============
SAMPLING_CONFIG = {
    # 影片長度分級採樣間隔（秒）
    'short_video_interval': 1,      # ≤60秒：每1秒採樣
    'medium_video_interval': 2,     # 60-300秒：每2秒採樣  
    'long_video_interval': 3,       # >300秒：每3秒採樣
    
    # pHash 閾值（根據影片長度調整）
    'short_video_threshold': 0.55,  # 短影片閾值
    'medium_video_threshold': 0.6,  # 中影片閾值
    'long_video_threshold': 0.65,   # 長影片閾值
    
    # 截尾均值參數
    'truncated_mean_min_length': 12,  # 最小長度
    'truncated_mean_ratio': 0.8,      # 截尾比例
    'truncated_mean_min_k': 6,        # 最小 k 值
}

# =============== pHash 特徵提取配置 ===============
PHASH_CONFIG = {
    # 圖像預處理參數
    'gaussian_blur_kernel': (5, 5),  # 高斯模糊核大小
    'gaussian_blur_sigma': 1,        # 高斯模糊標準差
    'resize_size': 64,               # 圖像縮放尺寸
    
    # Canny 邊緣檢測參數
    'canny_low_threshold': 50,       # Canny 低閾值
    'canny_high_threshold': 150,     # Canny 高閾值
    
    # DCT 特徵提取參數
    'dct_low_freq_size': 32,         # DCT 低頻部分大小
    'gray_threshold_factor': 0.3,    # 灰度閾值因子
    'phash_feature_dim': 10240,      # pHash 特徵維度 (1024+1024+8192)
    
    # 特徵維度
    'gray_hash_size': 1024,          # 灰度哈希大小 (32*32)
    'edge_hash_size': 1024,          # 邊緣哈希大小 (32*32)
    'hsv_hash_size': 8192,           # HSV 哈希大小 (64*64*2)
}

# =============== DTW 對齊配置 ===============
DTW_CONFIG = {
    # 對齊品質權重計算
    'alignment_weight_base': 0.5,      # 對齊權重基礎值
    'alignment_weight_scale': 0.5,     # 對齊權重縮放係數
    
    # 覆蓋率權重計算  
    'coverage_weight_base': 0.6,       # 覆蓋率權重基礎值
    'coverage_weight_scale': 0.4,      # 覆蓋率權重縮放係數
    'coverage_threshold': 0.70,        # 覆蓋率閾值
    
    # 對齊品質判斷閾值
    'high_quality_coverage_threshold': 0.80,  # 高品質覆蓋率閾值
    'high_quality_alignment_threshold': 0.70, # 高品質對齊閾值
    
    # DTW 路徑處理
    'max_pairs': 400,                 # 最大配對數量
}

# =============== 時間聚合配置 ===============
TEMPORAL_SMOOTHING_CONFIG = {
    'enabled': True,                  # 是否啟用時間聚合
    'method': 'ema',                  # 聚合方法：'ema' 或 'majority'
    'ema_alpha': 0.7,                 # EMA 平滑係數（0.6-0.8）
    'majority_window': 3,             # Majority voting 窗口大小
}

# =============== 相似度計算權重 ===============
SIMILARITY_WEIGHTS = {
    # pHash 特徵權重
    'gray_weight': 0.5,               # 灰度特徵權重
    'edge_weight': 0.3,               # 邊緣特徵權重
    'hsv_weight': 0.2,                # HSV 特徵權重
    
    # 最終相似度融合權重
    'phash_weight': 0.3,              # pHash 相似度權重
    'deep_weight': 0.7,               # 深度特徵相似度權重
    
    # 對齊後相似度權重（高品質對齊）
    'high_quality_primary_weight': 0.70,    # 主要分數權重
    'high_quality_secondary_weight': 0.30,  # 次要分數權重
    
    # 對齊後相似度權重（低品質對齊）
    'low_quality_deep_weight': 0.60,        # 深度特徵權重
    'low_quality_phash_weight': 0.40,       # pHash 權重
}

# =============== 批次處理配置 ===============
BATCH_CONFIG = {
    'default_batch_size': 64,         # 預設批次大小
    'gpu_batch_size': 32,             # GPU 批次大小（較小避免記憶體不足）
    'image_processing_chunk_size': 64, # 圖像處理塊大小
    'max_workers': 8,                 # 最大工作進程數
}

# =============== 深度特徵提取配置 ===============
DEEP_FEATURE_CONFIG = {
    # 模型配置
    'model_name': 'mobilenet_v3_large',
    'pretrained_weights': 'IMAGENET1K_V1',
    
    # 圖像預處理參數
    'resize_size': 256,               # 圖像縮放尺寸
    'crop_size': 224,                 # 中心裁切尺寸
    'normalize_mean': [0.485, 0.456, 0.406],  # 標準化均值
    'normalize_std': [0.229, 0.224, 0.225],   # 標準化標準差
    
    # 快取配置
    'cache_size': 1024,               # LRU 快取大小
}