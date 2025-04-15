import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
import os
from logger import logger
from gpu_utils import gpu_manager
import time
import signal
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# 全局變量
_image_model = None
_transform = None
_model_loaded = False
_feature_cache = {}

def cleanup():
    """清理資源"""
    if gpu_manager.is_pytorch_cuda_available():
        torch.cuda.empty_cache()
        gpu_manager.clear_gpu_memory()

@lru_cache(maxsize=1024)
def compute_phash(image_path: str) -> Optional[np.ndarray]:
    """計算感知哈希值（使用GPU加速）"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
            
        # 調整大小為32x32
        img = cv2.resize(img, (32, 32))
        
        # 如果有GPU可用，使用GPU
        if gpu_manager.is_pytorch_cuda_available():
            img_tensor = torch.from_numpy(img).float().cuda()
            # 使用DCT變換
            dct = torch.fft.rfft2(img_tensor)
            # 取低頻部分
            dct_low = torch.abs(dct[:8, :8])  # 使用絕對值避免複數
            # 將中值計算移到 CPU
            dct_low_cpu = dct_low.cpu()
            med = torch.median(dct_low_cpu)
            # 比較操作在 GPU 上進行
            phash = (dct_low > med.cuda()).cpu().numpy()
        else:
            # CPU版本
            dct = cv2.dct(np.float32(img))
            dct_low = dct[:8, :8]
            med = np.median(dct_low)
            phash = dct_low > med
            
        return phash
    except Exception as e:
        logger.error(f"計算pHash時出錯 {image_path}: {str(e)}")
        return None

def get_image_model():
    """獲取預訓練的 MobileNetV3-Large 模型"""
    global _image_model, _transform, _model_loaded
    if not _model_loaded:
        start_time = time.time()
        logger.info("開始載入 MobileNetV3-Large 模型...")
        
        try:
            # 使用 Large 版本
            _image_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            # 移除最後的分類層，只保留特徵提取部分
            _image_model = torch.nn.Sequential(*list(_image_model.children())[:-1])
            
            if gpu_manager.is_pytorch_cuda_available():
                _image_model = _image_model.cuda()
                _image_model.eval()  # 設置為評估模式
            
            # 更新預處理步驟以匹配 MobileNetV3-Large 的要求
            _transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
            
            _model_loaded = True
            logger.info(f"MobileNetV3-Large 模型載入完成，耗時: {time.time() - start_time:.2f}秒")
        
        except Exception as e:
            logger.error(f"載入模型時出錯: {str(e)}")
            raise
    
    return _image_model, _transform

def compute_batch_embeddings(image_paths: List[str], batch_size: int = 64) -> Optional[np.ndarray]:
    """批次計算圖像嵌入向量"""
    try:
        model, transform = get_image_model()
        embeddings = []
        
        # 使用線程池並行載入和預處理圖像
        with ThreadPoolExecutor() as executor:
            futures = []
            for path in image_paths:
                if path in _feature_cache:
                    embeddings.append(_feature_cache[path])
                else:
                    futures.append(executor.submit(lambda p: transform(Image.open(p).convert('RGB')), path))
            
            batch_tensors = []
            for future in futures:
                try:
                    tensor = future.result()
                    batch_tensors.append(tensor)
                except Exception as e:
                    logger.error(f"處理圖像時出錯: {str(e)}")
                    continue
        
        if not batch_tensors and not embeddings:
            return None
            
        # 處理未緩存的圖像
        if batch_tensors:
            for i in range(0, len(batch_tensors), batch_size):
                batch = batch_tensors[i:i + batch_size]
                if not batch:
                    continue
                
                batch_tensor = torch.stack(batch)
                if gpu_manager.is_pytorch_cuda_available():
                    batch_tensor = batch_tensor.cuda(non_blocking=True)
                
                with torch.no_grad():
                    features = model(batch_tensor)
                    if gpu_manager.is_pytorch_cuda_available():
                        features = features.cpu()
                
                for j, feature in enumerate(features):
                    feature_np = feature.squeeze().numpy()
                    embeddings.append(feature_np)
                    _feature_cache[image_paths[i + j]] = feature_np
                
                del batch_tensor, features
                if gpu_manager.is_pytorch_cuda_available():
                    torch.cuda.empty_cache()
        
        return np.array(embeddings)
    except Exception as e:
        logger.error(f"批次計算嵌入向量時出錯: {str(e)}")
        return None

def fast_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """計算pHash的漢明距離相似度"""
    if feat1.dtype == bool and feat2.dtype == bool:
        return 1 - np.count_nonzero(feat1 != feat2) / feat1.size
    return 0

def video_similarity(frames1: List[str], frames2: List[str], 
                    video_duration: float,
                    batch_size: int = 64) -> Dict[str, float]:
    """兩階段視頻相似度比對：pHash快速篩選 + MobileNetV3精確比對"""
    try:
        # 根據視頻時長確定採樣間隔
        if video_duration <= 60:  # 1分鐘以內
            sample_interval = 1    # 每秒1幀
            phash_threshold = 0.6  # 較寬鬆的閾值
        elif video_duration <= 300:  # 1-5分鐘
            sample_interval = 2    # 每2秒1幀
            phash_threshold = 0.65 # 中等閾值
        else:  # 5分鐘以上
            sample_interval = 3    # 每3秒1幀
            phash_threshold = 0.7  # 較嚴格的閾值
            
        # 均勻採樣
        sampled_frames1 = frames1[::sample_interval]
        sampled_frames2 = frames2[::sample_interval]
        
        logger.info(f"視頻1採樣後幀數: {len(sampled_frames1)}")
        logger.info(f"視頻2採樣後幀數: {len(sampled_frames2)}")
        
        # 第一階段：pHash快速比對
        similar_pairs = []
        
        # 並行計算 pHash
        with ThreadPoolExecutor() as executor:
            phash1 = list(executor.map(compute_phash, sampled_frames1))
            phash2 = list(executor.map(compute_phash, sampled_frames2))
            
        # 過濾無效的 pHash
        valid_frames1 = [(f, p) for f, p in zip(sampled_frames1, phash1) if p is not None]
        valid_frames2 = [(f, p) for f, p in zip(sampled_frames2, phash2) if p is not None]
        
        if not valid_frames1 or not valid_frames2:
            logger.error("無法計算有效的 pHash")
            return {"similarity": 0.0}
            
        # 快速比對階段
        for frame1, p1 in valid_frames1:
            for frame2, p2 in valid_frames2:
                sim = fast_similarity(p1, p2)
                if sim >= phash_threshold:
                    similar_pairs.append((frame1, frame2))
                    
        if not similar_pairs:
            logger.info("快速比對階段未找到相似幀")
            return {"similarity": 0.0}
            
        logger.info(f"pHash 快速比對找到 {len(similar_pairs)} 對相似幀")
        
        # 第二階段：MobileNetV3-Large 精確比對
        frames_to_compare1 = [pair[0] for pair in similar_pairs]
        frames_to_compare2 = [pair[1] for pair in similar_pairs]
        
        # 載入模型（如果還未載入）
        model, transform = get_image_model()
        
        # 批次計算特徵
        embeddings1 = compute_batch_embeddings(frames_to_compare1, batch_size)
        embeddings2 = compute_batch_embeddings(frames_to_compare2, batch_size)
        
        if embeddings1 is None or embeddings2 is None:
            return {"similarity": 0.0}
            
        # 標準化特徵向量
        embeddings1 = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8)
        embeddings2 = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8)
        
        # 計算餘弦相似度矩陣
        similarity_matrix = np.dot(embeddings1, embeddings2.T)
        
        # 取每個幀的最大相似度
        max_similarities = np.max(similarity_matrix, axis=1)
        final_similarity = float(np.mean(max_similarities))
        
        logger.info(f"MobileNetV3-Large 精確比對結果: {final_similarity:.3f}")
        
        return {
            "similarity": final_similarity,
            "similar_pairs": len(similar_pairs),
            "phash_threshold": phash_threshold
        }
        
    except Exception as e:
        logger.error(f"計算視頻相似度時出錯: {str(e)}")
        return {"similarity": 0.0}