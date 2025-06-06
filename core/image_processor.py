import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from utils.logger import logger
from functools import lru_cache
from utils.gpu_utils import gpu_manager
import torchvision.transforms as transforms
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


# 全局變量
_image_model = None
_transform = None
_model_loaded = False
_feature_cache = {}

def cleanup():
    """清理資源"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_manager.clear_gpu_memory()

@lru_cache(maxsize=1024)
def compute_phash(image_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """計算多重特徵的感知哈希值"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # 1. 灰度圖特徵 - 調整模糊參數
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1)  # 增加模糊半徑，減少噪音影響
        gray = cv2.resize(gray, (64, 64))
        
        # 2. 邊緣特徵 - 調整 Canny 參數
        edges = cv2.Canny(gray, 50, 150)  # 降低閾值，檢測更多邊緣
        edges = cv2.resize(edges, (64, 64))
        
        # 3. 顏色特徵
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = cv2.resize(hsv, (64, 64))
        
        gpu_available = torch.cuda.is_available()
        
        if gpu_available:
            # 處理灰度特徵
            gray_tensor = torch.from_numpy(gray).float().cuda()
            gray_dct = torch.fft.rfft2(gray_tensor)
            gray_dct_low = torch.abs(gray_dct[:32, :32])
            gray_mean = torch.mean(gray_dct_low.cpu())
            gray_std = torch.std(gray_dct_low.cpu())
            gray_threshold = gray_mean + 0.3 * gray_std  # 降低閾值
            gray_hash = (gray_dct_low > gray_threshold.cuda()).cpu().numpy()
            
            # 處理邊緣特徵
            edge_tensor = torch.from_numpy(edges).float().cuda()
            edge_dct = torch.fft.rfft2(edge_tensor)
            edge_dct_low = torch.abs(edge_dct[:32, :32])
            edge_mean = torch.mean(edge_dct_low.cpu())
            edge_hash = (edge_dct_low > edge_mean.cuda()).cpu().numpy()
            
            # 處理顏色特徵 - 只使用 H 和 S 通道
            hsv_tensor = torch.from_numpy(hsv[:,:,:2]).float().cuda()  # 只取 H 和 S 通道
            hsv_mean = torch.mean(hsv_tensor, dim=(0, 1))
            hsv_hash = (hsv_tensor > hsv_mean.reshape(1, 1, -1)).cpu().numpy()
            
        else:
            # CPU 版本
            gray_dct = cv2.dct(np.float32(gray))
            gray_dct_low = gray_dct[:32, :32]
            gray_mean = np.mean(gray_dct_low)
            gray_std = np.std(gray_dct_low)
            gray_threshold = gray_mean + 0.3 * gray_std
            gray_hash = gray_dct_low > gray_threshold
            
            edge_dct = cv2.dct(np.float32(edges))
            edge_dct_low = edge_dct[:32, :32]
            edge_mean = np.mean(edge_dct_low)
            edge_hash = edge_dct_low > edge_mean
            
            hsv_mean = np.mean(hsv[:,:,:2], axis=(0, 1))
            hsv_hash = hsv[:,:,:2] > hsv_mean.reshape(1, 1, -1)
            
        return gray_hash, edge_hash, hsv_hash
        
    except Exception as e:
        logger.error(f"計算多重特徵哈希時出錯 {image_path}: {str(e)}")
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
            
            if torch.cuda.is_available():
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
                if torch.cuda.is_available():
                    batch_tensor = batch_tensor.cuda(non_blocking=True)
                
                with torch.no_grad():
                    features = model(batch_tensor)
                    if torch.cuda.is_available():
                        features = features.cpu()
                
                for j, feature in enumerate(features):
                    feature_np = feature.squeeze().numpy()
                    embeddings.append(feature_np)
                    _feature_cache[image_paths[i + j]] = feature_np
                
                del batch_tensor, features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return np.array(embeddings)
    except Exception as e:
        logger.error(f"批次計算嵌入向量時出錯: {str(e)}")
        return None

def fast_similarity(feat1: Tuple[np.ndarray, np.ndarray, np.ndarray], 
                   feat2: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    """計算多重特徵的綜合相似度"""
    if all(f1 is not None and f2 is not None for f1, f2 in zip(feat1, feat2)):
        # 計算各個特徵的相似度
        gray_sim = 1 - np.count_nonzero(feat1[0] != feat2[0]) / feat1[0].size
        edge_sim = 1 - np.count_nonzero(feat1[1] != feat2[1]) / feat1[1].size
        hsv_sim = 1 - np.count_nonzero(feat1[2] != feat2[2]) / feat1[2].size
        
        # 直接使用加權平均
        weights = [0.5, 0.3, 0.2]  # 灰度、邊緣和顏色特徵的權重
        weighted_sim = (gray_sim * weights[0] + 
                       edge_sim * weights[1] + 
                       hsv_sim * weights[2])
        
        return weighted_sim
    return 0

def video_similarity(frames1: List[str], frames2: List[str], 
                    video_duration: float,
                    batch_size: int = 64) -> Dict[str, float]:
    """兩階段視頻相似度比對：pHash快速篩選 + MobileNetV3精確比對"""
    try:
        # 根據視頻時長確定採樣間隔和閾值
        if video_duration <= 60:  # 1分鐘以內
            sample_interval = 1    
            phash_threshold = 0.6  # 降低閾值
        elif video_duration <= 300:  # 1-5分鐘
            sample_interval = 2    
            phash_threshold = 0.65
        else:  # 5分鐘以上
            sample_interval = 3    
            phash_threshold = 0.7
            
        # 均勻採樣
        sampled_frames1 = frames1[::sample_interval]
        sampled_frames2 = frames2[::sample_interval]
        
        logger.info(f"視頻1採樣後幀數: {len(sampled_frames1)}")
        logger.info(f"視頻2採樣後幀數: {len(sampled_frames2)}")
        
        # 第一階段：pHash快速比對
        similar_pairs = []
        phash_similarities = []  # 儲存所有幀對的 pHash 相似度
        
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
            
        # 快速比對階段，同時記錄所有 pHash 相似度
        for frame1, p1 in valid_frames1:
            max_phash_sim = 0  # 記錄當前幀的最大 pHash 相似度
            best_match = None
            
            for frame2, p2 in valid_frames2:
                sim = fast_similarity(p1, p2)
                max_phash_sim = max(max_phash_sim, sim)
                
                if sim >= phash_threshold:
                    if best_match is None or sim > best_match[1]:
                        best_match = (frame2, sim)
            
            phash_similarities.append(max_phash_sim)  # 記錄最大相似度
            if best_match is not None:
                similar_pairs.append((frame1, best_match[0]))
        
        # 計算 pHash 階段的平均相似度
        avg_phash_similarity = np.mean(phash_similarities)
        logger.info(f"pHash 平均相似度: {avg_phash_similarity:.3f}")
        
        if not similar_pairs:
            logger.info("快速比對階段未找到高相似度幀對")
            # 返回 pHash 的平均相似度作為最終結果
            return {
                "similarity": avg_phash_similarity,
                "filtered_similarity": 0.0,
                "similar_pairs": 0,
                "total_pairs": len(valid_frames1),
                "phash_threshold": phash_threshold
            }
            
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
        
        # 計算深度特徵相似度
        similarity_matrix = np.dot(embeddings1, embeddings2.T)
        max_similarities = np.max(similarity_matrix, axis=1)
        deep_similarity = float(np.mean(max_similarities))
        
        # 計算整體相似度
        # 1. 對於通過閾值的幀對使用深度特徵相似度
        # 2. 對於未通過閾值的幀對使用其 pHash 相似度
        matched_ratio = len(similar_pairs) / len(valid_frames1)

        # 如果匹配率太低，降低整體相似度
        if matched_ratio < 0.3:  # 如果匹配率低於30%
            weight_factor = matched_ratio / 0.3  # 線性降低權重
        else:
            weight_factor = 1.0

        final_similarity = weight_factor * (matched_ratio * deep_similarity + 
                                          (1 - matched_ratio) * avg_phash_similarity)
        
        logger.info(f"深度特徵相似度: {deep_similarity:.3f}")
        logger.info(f"pHash 平均相似度: {avg_phash_similarity:.3f}")
        logger.info(f"最終整體相似度: {final_similarity:.3f}")
        logger.info(f"通過篩選幀數比例: {len(similar_pairs)}/{len(valid_frames1)} = {matched_ratio:.2%}")
        
        return {
            "similarity": final_similarity,
            "deep_similarity": deep_similarity,
            "phash_similarity": avg_phash_similarity,
            "similar_pairs": len(similar_pairs),
            "total_pairs": len(valid_frames1),
            "phash_threshold": phash_threshold
        }
        
    except Exception as e:
        logger.error(f"計算視頻相似度時出錯: {str(e)}")
        return {"similarity": 0.0}