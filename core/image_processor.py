import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from fastdtw import fastdtw
from functools import lru_cache
from utils.logger import logger
from utils.gpu_utils import gpu_manager
from scipy.spatial.distance import cosine
import torchvision.transforms as transforms
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from skimage.metrics import structural_similarity as ssim
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# ======================== 全域變數 ========================
_image_model: Optional[torch.nn.Module] = None
_transform: Optional[transforms.Compose] = None
_model_loaded: bool = False
_feature_cache: dict[str, np.ndarray] = {}

# ======================== 小工具 =========================


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _autocrop_letterbox(pil_img: Image.Image,
                        black_thr: int = 12,
                        max_border_frac: float = 0.25,
                        subtitle_frac: float = 0.10) -> Image.Image:
    """
    自動裁掉接近黑色的上下左右邊框；若底部 10% 亮且方差大（常見字幕帶），再裁掉一段。
    目的：減少 letterbox/字幕對特徵的干擾，提升對齊後的深度相似度。
    """
    img = np.array(pil_img)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def scan_edge(axis=0):
        if axis == 0:  # top/bottom
            rows_mean = gray.mean(axis=1)
            top = 0
            while top < int(h * max_border_frac) and rows_mean[top] <= black_thr:
                top += 1
            bottom = h - 1
            while bottom > h - int(h * max_border_frac) - 1 and rows_mean[bottom] <= black_thr:
                bottom -= 1
            return top, bottom
        else:  # left/right
            cols_mean = gray.mean(axis=0)
            left = 0
            while left < int(w * max_border_frac) and cols_mean[left] <= black_thr:
                left += 1
            right = w - 1
            while right > w - int(w * max_border_frac) - 1 and cols_mean[right] <= black_thr:
                right -= 1
            return left, right

    top, bottom = scan_edge(axis=0)
    left, right = scan_edge(axis=1)

    top = max(0, min(top, h - 2))
    bottom = max(top + 1, min(bottom, h - 1))
    left = max(0, min(left, w - 2))
    right = max(left + 1, min(right, w - 1))

    cropped = img[top:bottom + 1, left:right + 1, :]

    # 底部字幕帶偵測（亮且雜訊大）
    ch, cw = cropped.shape[:2]
    band_h = max(4, int(ch * 0.10))
    band = cv2.cvtColor(cropped[ch - band_h: ch, :, :], cv2.COLOR_RGB2GRAY)
    if band.mean() >= 140 and band.std() >= 25:
        cut = max(1, int(ch * subtitle_frac))
        cropped = cropped[:ch - cut, :, :]

    return Image.fromarray(cropped)

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


def _load_one(transform, p: str):
    img = Image.open(p).convert('RGB')
    img = _autocrop_letterbox(img)  # 去黑邊/字幕（可視情況調參）
    return transform(img)


def compute_batch_embeddings(image_paths: List[str], batch_size: int = 64) -> Optional[np.ndarray]:
    """
    批次計算圖像深度嵌入向量（保序、快取安全）
    - 只對「未快取」的路徑做 I/O+轉換 的並行
    - GPU forward 依原始順序分批，寫回 cache
    - 最後嚴格按 image_paths 原順序回填
    """
    try:
        model, transform = get_image_model()
        device = gpu_manager.get_device()

        non_cached = [p for p in image_paths if p not in _feature_cache]

        if non_cached:
            chunk = max(batch_size * 2, 64)
            for s in range(0, len(non_cached), chunk):
                block = non_cached[s:s + chunk]
                with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
                    tensors = list(ex.map(lambda p: _load_one(transform, p), block))

                for i in range(0, len(tensors), batch_size):
                    batch = torch.stack(tensors[i:i + batch_size])
                    if device.type == "cuda":
                        batch = batch.to(device)
                    with torch.no_grad():
                        feats = model(batch).cpu().numpy()
                    for j, vec in enumerate(feats):
                        _feature_cache[block[i + j]] = np.squeeze(vec)

                gpu_manager.clear_gpu_memory()

        ordered = []
        for p in image_paths:
            v = _feature_cache.get(p)
            if v is None:
                logger.error(f"快取缺漏：{p}")
                return None
            ordered.append(v)
        return np.array(ordered)

    except Exception as e:
        logger.error(f"批次計算嵌入向量時出錯: {e}")
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
    return 0.0


def dtw_similarity(emb_seq1: np.ndarray, emb_seq2: np.ndarray) -> float:
    """使用 fastdtw 計算兩段嵌入序列的相似度 (0~1)"""
    distance, _ = fastdtw(emb_seq1, emb_seq2, dist=cosine)
    sim = 1.0 / (1.0 + distance / max(len(emb_seq1), len(emb_seq2)))
    return sim

# =============== 由 DTW path 取對齊配對 ===============


def _pairs_from_dtw_path(frames1: List[str], frames2: List[str],
                         path: List[Tuple[int, int]],
                         max_pairs: int = 400) -> List[Tuple[str, str]]:
    """
    從 DTW 的 (i, j) path 直接抽取對齊配對。
    - 以固定步長取樣稀疏化，避免過度密集。
    - 僅保留合法索引與非重複連續點。
    """
    if not path:
        return []
    step = max(1, len(path) // max_pairs)
    picked: List[Tuple[str, str]] = []
    last_i, last_j = -1, -1
    for k in range(0, len(path), step):
        i, j = path[k]
        if i == last_i and j == last_j:
            continue
        if 0 <= i < len(frames1) and 0 <= j < len(frames2):
            picked.append((frames1[i], frames2[j]))
            last_i, last_j = i, j
    return picked

# =============== 影片相似度比對主流程 ===============


def video_similarity(frames1: List[str], frames2: List[str],
                     video_duration: float,
                     batch_size: int = 64) -> Dict[str, float]:
    """
    兩階段視訊相似度比對流程
    - pHash: 快速粗估（保留）
    - 深度特徵：沿 DTW path 逐幀對齊，並用截尾平均穩定噪聲
    """
    try:
        # 動態調整採樣與門檻
        if video_duration <= 60:
            sample_interval, phash_threshold = 1, 0.55
        elif video_duration <= 300:
            sample_interval, phash_threshold = 2, 0.6
        else:
            sample_interval, phash_threshold = 3, 0.65

        sampled_frames1 = frames1[::sample_interval]
        sampled_frames2 = frames2[::sample_interval]

        logger.info(f"採樣幀數: 視頻1={len(sampled_frames1)}，視頻2={len(sampled_frames2)}")

        # ──────────────── 第一階段：pHash 粗估（保留作為輔助） ────────────────
        with ThreadPoolExecutor() as executor:
            phash1 = list(executor.map(compute_phash, sampled_frames1))
            phash2 = list(executor.map(compute_phash, sampled_frames2))

        valid1 = [(i, f, p) for i, (f, p) in enumerate(zip(sampled_frames1, phash1)) if p is not None]
        valid2 = [(i, f, p) for i, (f, p) in enumerate(zip(sampled_frames2, phash2)) if p is not None]
        if not valid1 or not valid2:
            return {"similarity": 0.0}

        phash_similarities = []
        for _, _, p1 in valid1:
            max_sim = 0.0
            for _, _, p2 in valid2:
                s = fast_similarity(p1, p2)
                if s > max_sim:
                    max_sim = s
            phash_similarities.append(max_sim)
        avg_phash = float(np.mean(phash_similarities)) if phash_similarities else 0.0

        # ────────────── 取得完整序列嵌入 → DTW 對齊 ──────────────
        seq_emb1 = compute_batch_embeddings(sampled_frames1, batch_size)
        seq_emb2 = compute_batch_embeddings(sampled_frames2, batch_size)
        if seq_emb1 is None or seq_emb2 is None:
            logger.warning("序列嵌入為 None，DTW 對齊略過")
            return {"similarity": avg_phash}

        seq_emb1 = _normalize_rows(seq_emb1)
        seq_emb2 = _normalize_rows(seq_emb2)

        dtw_dist, path = fastdtw(seq_emb1, seq_emb2, dist=cosine)
        sim_dtw = 1.0 / (1.0 + dtw_dist / max(len(seq_emb1), len(seq_emb2)))
        lags = [j - i for i, j in path]
        lag_frames = int(np.median(lags)) if lags else 0

        covered_i = len({i for i, _ in path})
        covered_j = len({j for _, j in path})
        coverage = min(covered_i / max(1, len(seq_emb1)), covered_j / max(1, len(seq_emb2)))

        # ─────────────── 主分數：沿 DTW path 逐幀深度相似度（截尾平均） ───────────────
        aligned_pairs = _pairs_from_dtw_path(sampled_frames1, sampled_frames2, path, max_pairs=400)
        if not aligned_pairs:
            logger.warning("DTW 對齊後無配對，回退到 avg_phash")
            deep_aligned = 0.0
            phash_aligned = avg_phash
        else:
            e1 = compute_batch_embeddings([a for a, _ in aligned_pairs], batch_size)
            e2 = compute_batch_embeddings([b for _, b in aligned_pairs], batch_size)
            if e1 is None or e2 is None:
                return {"similarity": 0.0}
            e1 = _normalize_rows(e1)
            e2 = _normalize_rows(e2)

            per_frame_cos = np.sum(e1 * e2, axis=1)
            per_frame_cos = np.clip(per_frame_cos, -1.0, 1.0)
            if len(per_frame_cos) >= 12:
                k = max(6, int(0.8 * len(per_frame_cos)))
                deep_aligned = float(np.mean(np.partition(per_frame_cos, -k)[-k:]))
            else:
                deep_aligned = float(np.mean(per_frame_cos))

            # 沿 path 的 pHash（同樣用 80% 截尾平均）
            ph_sims = []
            for a, b in aligned_pairs:
                pa = compute_phash(a)
                pb = compute_phash(b)
                if pa is not None and pb is not None:
                    ph_sims.append(fast_similarity(pa, pb))
            if ph_sims:
                ph_sims = np.asarray(ph_sims, dtype=float)
                if len(ph_sims) >= 12:
                    k2 = max(6, int(0.8 * len(ph_sims)))
                    phash_aligned = float(np.mean(np.partition(ph_sims, -k2)[-k2:]))
                else:
                    phash_aligned = float(np.mean(ph_sims))
            else:
                phash_aligned = avg_phash

        # 估計 lag 秒數（以「採樣幀代表秒」的粗估法）
        seconds_per_sample = (video_duration / max(1, len(frames1))) * sample_interval
        estimated_lag_seconds = float(lag_frames) * seconds_per_sample

        # ─────────────── 分數融合 ───────────────
        # 純位移情境：覆蓋高且 DTW 高 → 讓分數更靠近「對齊後 deep 與 pHash 的較高者」
        if coverage >= 0.90 and sim_dtw >= 0.75:
            base_sim = 0.55 * max(deep_aligned, phash_aligned) + 0.45 * min(deep_aligned, phash_aligned)
        else:
            base_sim = 0.80 * deep_aligned + 0.20 * max(sim_dtw, avg_phash)

        # 覆蓋權重（60% 覆蓋即近滿權重，避免片頭/片尾稀釋）
        weight_factor = 0.65 + 0.35 * min(1.0, coverage / 0.60)
        final_sim = base_sim * weight_factor

        # log
        logger.info(
            f"[ALIGN] lag={lag_frames} ({estimated_lag_seconds:.2f}s) "
            f"coverage={coverage:.3f} dtw={sim_dtw:.3f}"
        )
        logger.info(
            f"[SCORES] deep_aligned={deep_aligned:.3f} pHash_avg={avg_phash:.3f} "
            f"pHash_aligned={phash_aligned:.3f} base={base_sim:.3f} weight={weight_factor:.3f}"
        )
        logger.info(f"[FINAL] final={final_sim:.3f} pairs={len(aligned_pairs)}")

        return {
            "similarity": final_sim,
            "deep_similarity": deep_aligned,        # 對齊後逐幀深度分數（截尾平均）
            "phash_similarity": phash_aligned,      # 沿 path 的 pHash（截尾平均）
            "phash_avg_global": avg_phash,          # 方便觀察
            "dtw_similarity": sim_dtw,
            "matched_ratio": coverage,              # 用 coverage 直觀反映重疊
            "similar_pairs": len(aligned_pairs),
            "total_pairs": len(sampled_frames1),
            "phash_threshold": phash_threshold,
            "aligned_coverage": float(coverage),
            "estimated_lag_frames": int(lag_frames),
            "estimated_lag_seconds": float(estimated_lag_seconds),
        }

    except Exception as e:
        logger.error(f"計算視頻相似度時出錯: {e}")
        return {"similarity": 0.0}
