import cv2
import time
import torch
import numpy as np
from PIL import Image
from functools import lru_cache
from utils.logger import logger
from utils.gpu_utils import gpu_manager
import torchvision.transforms as transforms
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# ======================== 全域變數 ========================
_image_model: Optional[torch.nn.Module] = None
_transform: Optional[transforms.Compose] = None
_model_loaded: bool = False
_feature_cache: dict[str, np.ndarray] = {}

# =============== 特徵擷取：感知哈希（pHash） ===============
@lru_cache(maxsize=1024)
def compute_phash(image_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """計算灰度、邊緣與顏色三種特徵的感知哈希"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1)
        gray = cv2.resize(gray, (64, 64))

        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.resize(edges, (64, 64))

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = cv2.resize(hsv, (64, 64))

        if gpu_manager.get_device().type == "cuda":
            #  ──────────────── GPU 快速 DCT／閾值二值化  ────────────────
            gray_tensor = torch.from_numpy(gray).float().to(gpu_manager.get_device())
            edge_tensor = torch.from_numpy(edges).float().to(gpu_manager.get_device())
            hsv_tensor = torch.from_numpy(hsv[:, :, :2]).float().to(gpu_manager.get_device())

            gray_dct = torch.fft.rfft2(gray_tensor)
            gray_dct_low = torch.abs(gray_dct[:32, :32])
            gray_threshold = torch.mean(gray_dct_low.cpu()) + 0.3 * torch.std(gray_dct_low.cpu())
            gray_hash = (gray_dct_low > gray_threshold.to(gpu_manager.get_device())).cpu().numpy()

            edge_dct = torch.fft.rfft2(edge_tensor)
            edge_dct_low = torch.abs(edge_dct[:32, :32])
            edge_mean = torch.mean(edge_dct_low.cpu())
            edge_hash = (edge_dct_low > edge_mean.to(gpu_manager.get_device())).cpu().numpy()

            hsv_mean = torch.mean(hsv_tensor, dim=(0, 1))
            hsv_hash = (hsv_tensor > hsv_mean.reshape(1, 1, -1)).cpu().numpy()
        else:
            #  ──────────────── CPU 模式  ────────────────
            gray_dct = cv2.dct(np.float32(gray))[:32, :32]
            gray_hash = gray_dct > np.mean(gray_dct) + 0.3 * np.std(gray_dct)

            edge_dct = cv2.dct(np.float32(edges))[:32, :32]
            edge_hash = edge_dct > np.mean(edge_dct)

            hsv_mean = np.mean(hsv[:, :, :2], axis=(0, 1))
            hsv_hash = hsv[:, :, :2] > hsv_mean.reshape(1, 1, -1)

        return gray_hash, edge_hash, hsv_hash

    except Exception as e:
        logger.error(f"計算多重特徵哈希時出錯 {image_path}: {str(e)}")
        return None

# =============== 特徵擷取：深度模型（MobileNetV3） ===============
def get_image_model():
    """載入 MobileNetV3-Large 模型與前處理流程"""
    global _image_model, _transform, _model_loaded
    if not _model_loaded:
        try:
            start_time = time.time()
            logger.info("開始載入 MobileNetV3-Large 模型...")
            _image_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            _image_model = torch.nn.Sequential(*list(_image_model.children())[:-1])
            gpu_manager.initialize()
            if gpu_manager.get_device().type == "cuda":
                _image_model = _image_model.to(gpu_manager.get_device())
            _image_model.eval()

            _transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            _model_loaded = True
            logger.info(f"模型載入完成，耗時: {time.time() - start_time:.2f}秒")
        except Exception as e:
            logger.error(f"載入模型時出錯: {str(e)}")
            raise
    return _image_model, _transform

def compute_batch_embeddings(image_paths: List[str], batch_size: int = 64) -> Optional[np.ndarray]:
    """批次計算圖像深度嵌入向量"""
    try:
        model, transform = get_image_model()
        embeddings = []

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(lambda p: transform(Image.open(p).convert('RGB')), path)
                for path in image_paths if path not in _feature_cache
            ]
            for path in image_paths:
                if path in _feature_cache:
                    embeddings.append(_feature_cache[path])

            batch_tensors = []
            for future in futures:
                try:
                    batch_tensors.append(future.result())
                except Exception as e:
                    logger.error(f"處理圖像時出錯: {str(e)}")

        if batch_tensors:
            for i in range(0, len(batch_tensors), batch_size):
                batch = torch.stack(batch_tensors[i:i + batch_size])
                batch = batch.to(gpu_manager.get_device()) if gpu_manager.get_device().type == "cuda" else batch
                with torch.no_grad():
                    features = model(batch).cpu()
                for j, f in enumerate(features):
                    vec = f.squeeze().numpy()
                    embeddings.append(vec)
                    _feature_cache[image_paths[i + j]] = vec

                gpu_manager.clear_gpu_memory()

        return np.array(embeddings) if embeddings else None

    except Exception as e:
        logger.error(f"批次計算嵌入向量時出錯: {str(e)}")
        return None

# =============== 特徵比對邏輯 ===============
def fast_similarity(feat1: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    feat2: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    """快速比對 pHash 特徵的綜合相似度"""
    if all(f1 is not None and f2 is not None for f1, f2 in zip(feat1, feat2)):
        gray_sim = 1 - np.count_nonzero(feat1[0] != feat2[0]) / feat1[0].size
        edge_sim = 1 - np.count_nonzero(feat1[1] != feat2[1]) / feat1[1].size
        hsv_sim = 1 - np.count_nonzero(feat1[2] != feat2[2]) / feat1[2].size
        return gray_sim * 0.5 + edge_sim * 0.3 + hsv_sim * 0.2
    return 0

# =============== 影片相似度比對主流程 ===============
def video_similarity(frames1: List[str], frames2: List[str],
                     video_duration: float,
                     batch_size: int = 64) -> Dict[str, float]:
    """兩階段視訊相似度比對流程"""
    try:
        # 動態調整採樣與門檻
        if video_duration <= 60:
            sample_interval, phash_threshold = 1, 0.6
        elif video_duration <= 300:
            sample_interval, phash_threshold = 2, 0.65
        else:
            sample_interval, phash_threshold = 3, 0.7

        sampled_frames1 = frames1[::sample_interval]
        sampled_frames2 = frames2[::sample_interval]

        logger.info(f"採樣幀數: 視頻1={len(sampled_frames1)}，視頻2={len(sampled_frames2)}")

        #  ──────────────── 第一階段：pHash 快速比對 ────────────────
        with ThreadPoolExecutor() as executor:
            phash1 = list(executor.map(compute_phash, sampled_frames1))
            phash2 = list(executor.map(compute_phash, sampled_frames2))

        valid1 = [(f, p) for f, p in zip(sampled_frames1, phash1) if p is not None]
        valid2 = [(f, p) for f, p in zip(sampled_frames2, phash2) if p is not None]

        if not valid1 or not valid2:
            return {"similarity": 0.0}

        similar_pairs = []
        phash_similarities = []
        for f1, p1 in valid1:
            max_sim = 0
            best = None
            for f2, p2 in valid2:
                sim = fast_similarity(p1, p2)
                if sim > max_sim:
                    max_sim = sim
                    best = (f2, sim)
            phash_similarities.append(max_sim)
            if best and best[1] >= phash_threshold:
                similar_pairs.append((f1, best[0]))

        avg_phash = np.mean(phash_similarities)
        matched_ratio = len(similar_pairs) / len(valid1)

        if not similar_pairs:
            return {
                "similarity": avg_phash,
                "filtered_similarity": 0.0,
                "similar_pairs": 0,
                "total_pairs": len(valid1),
                "phash_threshold": phash_threshold
            }

        #  ──────────────── 第二階段：深度特徵比對  ────────────────
        e1 = compute_batch_embeddings([p[0] for p in similar_pairs], batch_size)
        e2 = compute_batch_embeddings([p[1] for p in similar_pairs], batch_size)

        if e1 is None or e2 is None:
            return {"similarity": 0.0}

        e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-8
        e2 /= np.linalg.norm(e2, axis=1, keepdims=True) + 1e-8

        sim_matrix = np.dot(e1, e2.T)
        deep_sim = float(np.mean(np.max(sim_matrix, axis=1)))

        weight_factor = min(1.0, matched_ratio / 0.3)
        final_sim = weight_factor * (matched_ratio * deep_sim + (1 - matched_ratio) * avg_phash)

        return {
            "similarity": final_sim,
            "deep_similarity": deep_sim,
            "phash_similarity": avg_phash,
            "similar_pairs": len(similar_pairs),
            "total_pairs": len(valid1),
            "phash_threshold": phash_threshold
        }

    except Exception as e:
        logger.error(f"計算視頻相似度時出錯: {str(e)}")
        return {"similarity": 0.0}