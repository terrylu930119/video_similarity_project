# core/image_processor.py
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
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# ======================== 全域變數 ========================
_image_model: Optional[torch.nn.Module] = None
_transform: Optional[transforms.Compose] = None
_model_loaded: bool = False
_feature_cache: dict[str, np.ndarray] = {}


# ======================== 小工具 =========================
def _normalize_rows(x: np.ndarray) -> np.ndarray:
    """正規化矩陣的每一行"""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _scan_vertical_edges(gray: np.ndarray, max_border_frac: float, black_thr: int) -> Tuple[int, int]:
    """
    掃描垂直方向的邊緣

    Args:
        gray: 灰度圖像
        max_border_frac: 最大邊框比例
        black_thr: 黑色閾值

    Returns:
        Tuple[int, int]: (top, bottom) 邊界
    """
    rows_mean = gray.mean(axis=1)
    h = gray.shape[0]

    top = 0
    while top < int(h * max_border_frac) and rows_mean[top] <= black_thr:
        top += 1

    bottom = h - 1
    while bottom > h - int(h * max_border_frac) - 1 and rows_mean[bottom] <= black_thr:
        bottom -= 1

    return top, bottom


def _scan_horizontal_edges(gray: np.ndarray, max_border_frac: float, black_thr: int) -> Tuple[int, int]:
    """
    掃描水平方向的邊緣

    Args:
        gray: 灰度圖像
        max_border_frac: 最大邊框比例
        black_thr: 黑色閾值

    Returns:
        Tuple[int, int]: (left, right) 邊界
    """
    cols_mean = gray.mean(axis=0)
    w = gray.shape[1]

    left = 0
    while left < int(w * max_border_frac) and cols_mean[left] <= black_thr:
        left += 1

    right = w - 1
    while right > w - int(w * max_border_frac) - 1 and cols_mean[right] <= black_thr:
        right -= 1

    return left, right


def _scan_edge_axis(gray: np.ndarray, axis: int, max_border_frac: float, black_thr: int) -> Tuple[int, int]:
    """
    掃描指定軸向的邊緣，找到非黑色區域的邊界

    Args:
        gray: 灰度圖像
        axis: 掃描軸向 (0: 垂直, 1: 水平)
        max_border_frac: 最大邊框比例
        black_thr: 黑色閾值

    Returns:
        Tuple[int, int]: (起始位置, 結束位置)
    """
    if axis == 0:  # top/bottom
        return _scan_vertical_edges(gray, max_border_frac, black_thr)
    else:  # left/right
        return _scan_horizontal_edges(gray, max_border_frac, black_thr)


def _detect_subtitle_band(cropped: np.ndarray, subtitle_frac: float) -> np.ndarray:
    """
    偵測並移除底部字幕帶

    Args:
        cropped: 裁切後的圖像
        subtitle_frac: 字幕帶比例

    Returns:
        np.ndarray: 移除字幕帶後的圖像
    """
    ch, cw = cropped.shape[:2]
    band_h = max(4, int(ch * 0.10))
    band = cv2.cvtColor(cropped[ch - band_h: ch, :, :], cv2.COLOR_RGB2GRAY)

    if band.mean() >= 140 and band.std() >= 25:
        cut = max(1, int(ch * subtitle_frac))
        cropped = cropped[:ch - cut, :, :]

    return cropped


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

    # 掃描邊緣
    top, bottom = _scan_edge_axis(gray, 0, max_border_frac, black_thr)
    left, right = _scan_edge_axis(gray, 1, max_border_frac, black_thr)

    # 確保邊界有效
    top = max(0, min(top, h - 2))
    bottom = max(top + 1, min(bottom, h - 1))
    left = max(0, min(left, w - 2))
    right = max(left + 1, min(right, w - 1))

    # 裁切圖像
    cropped = img[top:bottom + 1, left:right + 1, :]

    # 移除字幕帶
    cropped = _detect_subtitle_band(cropped, subtitle_frac)

    return Image.fromarray(cropped)


# =============== 特徵擷取：感知哈希（pHash） ===============
def _compute_gpu_phash(gray: np.ndarray, edges: np.ndarray,
                       hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 GPU 計算感知哈希

    Args:
        gray: 灰度圖像
        edges: 邊緣圖像
        hsv: HSV 圖像

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 三種特徵的哈希
    """
    device = gpu_manager.get_device()

    gray_tensor = torch.from_numpy(gray).float().to(device)
    edge_tensor = torch.from_numpy(edges).float().to(device)
    hsv_tensor = torch.from_numpy(hsv[:, :, :2]).float().to(device)

    # 計算 DCT 並提取低頻部分
    gray_dct = torch.fft.rfft2(gray_tensor)
    gray_dct_low = torch.abs(gray_dct[:32, :32])
    gray_threshold = torch.mean(gray_dct_low.cpu()) + 0.3 * torch.std(gray_dct_low.cpu())
    gray_hash = (gray_dct_low > gray_threshold.to(device)).cpu().numpy()

    edge_dct = torch.fft.rfft2(edge_tensor)
    edge_dct_low = torch.abs(edge_dct[:32, :32])
    edge_mean = torch.mean(edge_dct_low.cpu())
    edge_hash = (edge_dct_low > edge_mean.to(device)).cpu().numpy()

    hsv_mean = torch.mean(hsv_tensor, dim=(0, 1))
    hsv_hash = (hsv_tensor > hsv_mean.reshape(1, 1, -1)).cpu().numpy()

    return gray_hash, edge_hash, hsv_hash


def _compute_cpu_phash(gray: np.ndarray, edges: np.ndarray,
                       hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 CPU 計算感知哈希

    Args:
        gray: 灰度圖像
        edges: 邊緣圖像
        hsv: HSV 圖像

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 三種特徵的哈希
    """
    # 計算 DCT 並提取低頻部分
    gray_dct = cv2.dct(np.float32(gray))[:32, :32]
    gray_hash = gray_dct > np.mean(gray_dct) + 0.3 * np.std(gray_dct)

    edge_dct = cv2.dct(np.float32(edges))[:32, :32]
    edge_hash = edge_dct > np.mean(edge_dct)

    hsv_mean = np.mean(hsv[:, :, :2], axis=(0, 1))
    hsv_hash = hsv[:, :, :2] > hsv_mean.reshape(1, 1, -1)

    return gray_hash, edge_hash, hsv_hash


@lru_cache(maxsize=1024)
def compute_phash(image_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """計算灰度、邊緣與顏色三種特徵的感知哈希"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        # 預處理圖像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1)
        gray = cv2.resize(gray, (64, 64))

        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.resize(edges, (64, 64))

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = cv2.resize(hsv, (64, 64))

        # 根據設備選擇計算方式
        if gpu_manager.get_device().type == "cuda":
            return _compute_gpu_phash(gray, edges, hsv)
        else:
            return _compute_cpu_phash(gray, edges, hsv)

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
    """載入並預處理單張圖像"""
    img = Image.open(p).convert('RGB')
    img = _autocrop_letterbox(img)  # 去黑邊/字幕（可視情況調參）
    return transform(img)


def _process_batch_tensors(tensors: List[torch.Tensor], batch_size: int,
                           model: torch.nn.Module, device: torch.device,
                           block: List[str]) -> None:
    """
    處理批次張量，計算特徵並存入快取

    Args:
        tensors: 張量列表
        batch_size: 批次大小
        model: 模型
        device: 計算設備
        block: 對應的檔案路徑列表
    """
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i:i + batch_size])
        if device.type == "cuda":
            batch = batch.to(device)
        with torch.no_grad():
            feats = model(batch).cpu().numpy()
        for j, vec in enumerate(feats):
            _feature_cache[block[i + j]] = np.squeeze(vec)


def _load_and_process_images(non_cached: List[str], transform, batch_size: int,
                             model: torch.nn.Module, device: torch.device) -> None:
    """
    載入並處理圖像批次

    Args:
        non_cached: 未快取的路徑列表
        transform: 圖像轉換器
        batch_size: 批次大小
        model: 模型
        device: 計算設備
    """
    chunk = max(batch_size * 2, 64)
    for s in range(0, len(non_cached), chunk):
        block = non_cached[s:s + chunk]

        # 並行載入圖像
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            tensors = list(ex.map(lambda p: _load_one(transform, p), block))

        # 批次處理特徵
        _process_batch_tensors(tensors, batch_size, model, device, block)
        gpu_manager.clear_gpu_memory()


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

        # 找出未快取的路徑
        non_cached = [p for p in image_paths if p not in _feature_cache]

        if non_cached:
            _load_and_process_images(non_cached, transform, batch_size, model, device)

        # 按原順序回填結果
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
def _calculate_sampling_parameters(video_duration: float) -> Tuple[int, float]:
    """
    根據影片長度計算採樣參數

    Args:
        video_duration: 影片長度（秒）

    Returns:
        Tuple[int, float]: (採樣間隔, pHash 閾值)
    """
    if video_duration <= 60:
        return 1, 0.55
    elif video_duration <= 300:
        return 2, 0.6
    else:
        return 3, 0.65


def _find_max_similarity(phash1: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         valid2: List[Tuple[int, str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]) -> float:
    """
    找到與指定 pHash 的最大相似度

    Args:
        phash1: 第一個 pHash 特徵
        valid2: 第二個影片的有效 pHash 列表

    Returns:
        float: 最大相似度
    """
    max_sim = 0.0
    for _, _, p2 in valid2:
        s = fast_similarity(phash1, p2)
        if s > max_sim:
            max_sim = s
    return max_sim


def _extract_valid_phash_data(sampled_frames: List[str], phash_results: List[Optional[Tuple[np.ndarray,
                              np.ndarray, np.ndarray]]]) -> List[Tuple[int, str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    提取有效的 pHash 資料

    Args:
        sampled_frames: 採樣幀列表
        phash_results: pHash 計算結果列表

    Returns:
        List[Tuple[int, str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]: 有效的 pHash 資料
    """
    return [(i, f, p) for i, (f, p) in enumerate(zip(sampled_frames, phash_results)) if p is not None]


def _compute_phash_similarities(sampled_frames1: List[str], sampled_frames2: List[str]) -> Tuple[List[float], float]:
    """
    計算 pHash 相似度

    Args:
        sampled_frames1: 採樣幀列表 1
        sampled_frames2: 採樣幀列表 2

    Returns:
        Tuple[List[float], float]: (相似度列表, 平均相似度)
    """
    with ThreadPoolExecutor() as executor:
        phash1 = list(executor.map(compute_phash, sampled_frames1))
        phash2 = list(executor.map(compute_phash, sampled_frames2))

    valid1 = _extract_valid_phash_data(sampled_frames1, phash1)
    valid2 = _extract_valid_phash_data(sampled_frames2, phash2)

    if not valid1 or not valid2:
        return [], 0.0

    phash_similarities = []
    for _, _, p1 in valid1:
        max_sim = _find_max_similarity(p1, valid2)
        phash_similarities.append(max_sim)

    avg_phash = float(np.mean(phash_similarities)) if phash_similarities else 0.0
    return phash_similarities, avg_phash


def _compute_dtw_alignment(sampled_frames1: List[str], sampled_frames2: List[str],
                           batch_size: int) -> Tuple[float, List[Tuple[int, int]], float, int]:
    """
    計算 DTW 對齊

    Args:
        sampled_frames1: 採樣幀列表 1
        sampled_frames2: 採樣幀列表 2
        batch_size: 批次大小

    Returns:
        Tuple[float, List[Tuple[int, int]], float, int]: (DTW 相似度, 路徑, 覆蓋率, 延遲幀數)
    """
    seq_emb1 = compute_batch_embeddings(sampled_frames1, batch_size)
    seq_emb2 = compute_batch_embeddings(sampled_frames2, batch_size)

    if seq_emb1 is None or seq_emb2 is None:
        logger.warning("序列嵌入為 None，DTW 對齊略過")
        return 0.0, [], 0.0, 0

    seq_emb1 = _normalize_rows(seq_emb1)
    seq_emb2 = _normalize_rows(seq_emb2)

    dtw_dist, path = fastdtw(seq_emb1, seq_emb2, dist=cosine)
    sim_dtw = 1.0 / (1.0 + dtw_dist / max(len(seq_emb1), len(seq_emb2)))
    lags = [j - i for i, j in path]
    lag_frames = int(np.median(lags)) if lags else 0

    covered_i = len({i for i, _ in path})
    covered_j = len({j for _, j in path})
    coverage = min(covered_i / max(1, len(seq_emb1)), covered_j / max(1, len(seq_emb2)))

    return sim_dtw, path, coverage, lag_frames


def _calculate_truncated_mean(values: np.ndarray, min_length: int = 12, ratio: float = 0.8) -> float:
    """
    計算截尾平均

    Args:
        values: 數值陣列
        min_length: 最小長度閾值
        ratio: 截尾比例

    Returns:
        float: 截尾平均
    """
    if len(values) >= min_length:
        k = max(6, int(ratio * len(values)))
        return float(np.mean(np.partition(values, -k)[-k:]))
    else:
        return float(np.mean(values))


def _compute_deep_similarity(aligned_pairs: List[Tuple[str, str]], batch_size: int) -> float:
    """
    計算深度相似度

    Args:
        aligned_pairs: 對齊的幀對
        batch_size: 批次大小

    Returns:
        float: 深度相似度
    """
    e1 = compute_batch_embeddings([a for a, _ in aligned_pairs], batch_size)
    e2 = compute_batch_embeddings([b for _, b in aligned_pairs], batch_size)

    if e1 is None or e2 is None:
        return 0.0

    e1 = _normalize_rows(e1)
    e2 = _normalize_rows(e2)

    per_frame_cos = np.sum(e1 * e2, axis=1)
    per_frame_cos = np.clip(per_frame_cos, -1.0, 1.0)

    return _calculate_truncated_mean(per_frame_cos)


def _compute_phash_aligned_similarity(aligned_pairs: List[Tuple[str, str]], avg_phash: float) -> float:
    """
    計算對齊後的 pHash 相似度

    Args:
        aligned_pairs: 對齊的幀對
        avg_phash: 平均 pHash 相似度

    Returns:
        float: 對齊後的 pHash 相似度
    """
    ph_sims = []
    for a, b in aligned_pairs:
        pa = compute_phash(a)
        pb = compute_phash(b)
        if pa is not None and pb is not None:
            ph_sims.append(fast_similarity(pa, pb))

    if ph_sims:
        ph_sims = np.asarray(ph_sims, dtype=float)
        return _calculate_truncated_mean(ph_sims)
    else:
        return avg_phash


def _compute_aligned_similarities(aligned_pairs: List[Tuple[str, str]],
                                  batch_size: int, avg_phash: float) -> Tuple[float, float]:
    """
    計算對齊後的相似度

    Args:
        aligned_pairs: 對齊的幀對
        batch_size: 批次大小
        avg_phash: 平均 pHash 相似度

    Returns:
        Tuple[float, float]: (深度相似度, pHash 相似度)
    """
    if not aligned_pairs:
        logger.warning("DTW 對齊後無配對，回退到 avg_phash")
        return 0.0, avg_phash

    # 計算深度相似度
    deep_aligned = _compute_deep_similarity(aligned_pairs, batch_size)

    # 計算對齊後的 pHash 相似度
    phash_aligned = _compute_phash_aligned_similarity(aligned_pairs, avg_phash)

    return deep_aligned, phash_aligned


def _calculate_final_similarity(deep_aligned: float, phash_aligned: float, sim_dtw: float,
                                avg_phash: float, coverage: float) -> float:
    """
    計算最終相似度分數

    Args:
        deep_aligned: 對齊後的深度相似度
        phash_aligned: 對齊後的 pHash 相似度
        sim_dtw: DTW 相似度
        avg_phash: 平均 pHash 相似度
        coverage: 覆蓋率

    Returns:
        float: 最終相似度分數
    """
    # 純位移情境：覆蓋高且 DTW 高 → 讓分數更靠近「對齊後 deep 與 pHash 的較高者」
    if coverage >= 0.90 and sim_dtw >= 0.75:
        base_sim = 0.55 * max(deep_aligned, phash_aligned) + 0.45 * min(deep_aligned, phash_aligned)
    else:
        base_sim = 0.80 * deep_aligned + 0.20 * max(sim_dtw, avg_phash)

    # 覆蓋權重（60% 覆蓋即近滿權重，避免片頭/片尾稀釋）
    weight_factor = 0.65 + 0.35 * min(1.0, coverage / 0.60)
    final_sim = base_sim * weight_factor

    return final_sim


def _log_similarity_results(lag_frames: int, estimated_lag_seconds: float, coverage: float,
                            sim_dtw: float, deep_aligned: float, avg_phash: float,
                            phash_aligned: float, final_sim: float, aligned_pairs: List[Tuple[str, str]]) -> None:
    """
    記錄相似度計算結果

    Args:
        lag_frames: 延遲幀數
        estimated_lag_seconds: 估計延遲秒數
        coverage: 覆蓋率
        sim_dtw: DTW 相似度
        deep_aligned: 對齊後的深度相似度
        avg_phash: 平均 pHash 相似度
        phash_aligned: 對齊後的 pHash 相似度
        final_sim: 最終相似度
        aligned_pairs: 對齊的幀對
    """
    logger.info(
        f"[ALIGN] lag={lag_frames} ({estimated_lag_seconds:.2f}s) "
        f"coverage={coverage:.3f} dtw={sim_dtw:.3f}"
    )
    logger.info(
        f"[SCORES] deep_aligned={deep_aligned:.3f} pHash_avg={avg_phash:.3f} "
        f"pHash_aligned={phash_aligned:.3f} base={final_sim:.3f}"
    )
    logger.info(f"[FINAL] final={final_sim:.3f} pairs={len(aligned_pairs)}")


def _execute_phash_stage(sampled_frames1: List[str], sampled_frames2: List[str]) -> Tuple[List[float], float]:
    """
    執行第一階段：pHash 粗估

    Args:
        sampled_frames1: 採樣幀列表 1
        sampled_frames2: 採樣幀列表 2

    Returns:
        Tuple[List[float], float]: (相似度列表, 平均相似度)
    """
    phash_similarities, avg_phash = _compute_phash_similarities(sampled_frames1, sampled_frames2)
    if not phash_similarities:
        return [], 0.0
    return phash_similarities, avg_phash


def _execute_dtw_stage(sampled_frames1: List[str], sampled_frames2: List[str],
                       batch_size: int) -> Tuple[float, List[Tuple[int, int]], float, int]:
    """
    執行第二階段：DTW 對齊

    Args:
        sampled_frames1: 採樣幀列表 1
        sampled_frames2: 採樣幀列表 2
        batch_size: 批次大小

    Returns:
        Tuple[float, List[Tuple[int, int]], float, int]: (DTW 相似度, 路徑, 覆蓋率, 延遲幀數)
    """
    sim_dtw, path, coverage, lag_frames = _compute_dtw_alignment(sampled_frames1, sampled_frames2, batch_size)
    if not path:
        return 0.0, [], 0.0, 0
    return sim_dtw, path, coverage, lag_frames


def _execute_alignment_stage(sampled_frames1: List[str], sampled_frames2: List[str],
                             path: List[Tuple[int, int]], batch_size: int,
                             avg_phash: float) -> Tuple[float, float, List[Tuple[str, str]]]:
    """
    執行對齊階段：計算對齊後的相似度

    Args:
        sampled_frames1: 採樣幀列表 1
        sampled_frames2: 採樣幀列表 2
        path: DTW 路徑
        batch_size: 批次大小
        avg_phash: 平均 pHash 相似度

    Returns:
        Tuple[float, float, List[Tuple[str, str]]]: (深度相似度, pHash 相似度, 對齊幀對)
    """
    aligned_pairs = _pairs_from_dtw_path(sampled_frames1, sampled_frames2, path, max_pairs=400)
    deep_aligned, phash_aligned = _compute_aligned_similarities(aligned_pairs, batch_size, avg_phash)
    return deep_aligned, phash_aligned, aligned_pairs


def _estimate_lag_time(lag_frames: int, video_duration: float, frames1: List[str], sample_interval: int) -> float:
    """
    估計延遲時間

    Args:
        lag_frames: 延遲幀數
        video_duration: 影片長度
        frames1: 第一影片幀列表
        sample_interval: 採樣間隔

    Returns:
        float: 估計的延遲秒數
    """
    seconds_per_sample = (video_duration / max(1, len(frames1))) * sample_interval
    return float(lag_frames) * seconds_per_sample


def _create_result_dict(final_sim: float, deep_aligned: float, phash_aligned: float, avg_phash: float,
                        sim_dtw: float, coverage: float, aligned_pairs: List[Tuple[str, str]],
                        sampled_frames1: List[str], phash_threshold: float, lag_frames: int,
                        estimated_lag_seconds: float) -> Dict[str, float]:
    """
    建立結果字典

    Args:
        final_sim: 最終相似度
        deep_aligned: 對齊後的深度相似度
        phash_aligned: 對齊後的 pHash 相似度
        avg_phash: 平均 pHash 相似度
        sim_dtw: DTW 相似度
        coverage: 覆蓋率
        aligned_pairs: 對齊的幀對
        sampled_frames1: 採樣幀列表
        phash_threshold: pHash 閾值
        lag_frames: 延遲幀數
        estimated_lag_seconds: 估計延遲秒數

    Returns:
        Dict[str, float]: 結果字典
    """
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


def video_similarity(frames1: List[str], frames2: List[str],
                     video_duration: float, batch_size: int = 64) -> Dict[str, float]:
    """
    兩階段視訊相似度比對流程
    - pHash: 快速粗估（保留）
    - 深度特徵：沿 DTW path 逐幀對齊，並用截尾平均穩定噪聲
    """
    try:
        # 動態調整採樣與門檻
        sample_interval, phash_threshold = _calculate_sampling_parameters(video_duration)
        sampled_frames1 = frames1[::sample_interval]
        sampled_frames2 = frames2[::sample_interval]

        logger.info(f"採樣幀數: 視頻1={len(sampled_frames1)}，視頻2={len(sampled_frames2)}")

        # 第一階段：pHash 粗估
        phash_similarities, avg_phash = _execute_phash_stage(sampled_frames1, sampled_frames2)
        if not phash_similarities:
            return {"similarity": 0.0}

        # 第二階段：DTW 對齊
        sim_dtw, path, coverage, lag_frames = _execute_dtw_stage(sampled_frames1, sampled_frames2, batch_size)
        if not path:
            return {"similarity": avg_phash}

        # 第三階段：對齊後相似度計算
        deep_aligned, phash_aligned, aligned_pairs = _execute_alignment_stage(
            sampled_frames1, sampled_frames2, path, batch_size, avg_phash)

        # 估計延遲時間
        estimated_lag_seconds = _estimate_lag_time(lag_frames, video_duration, frames1, sample_interval)

        # 計算最終相似度
        final_sim = _calculate_final_similarity(deep_aligned, phash_aligned, sim_dtw, avg_phash, coverage)

        # 記錄日誌
        _log_similarity_results(lag_frames, estimated_lag_seconds, coverage, sim_dtw,
                                deep_aligned, avg_phash, phash_aligned, final_sim, aligned_pairs)

        # 建立結果字典
        return _create_result_dict(final_sim, deep_aligned, phash_aligned, avg_phash, sim_dtw,
                                   coverage, aligned_pairs, sampled_frames1, phash_threshold,
                                   lag_frames, estimated_lag_seconds)

    except Exception as e:
        logger.error(f"計算視頻相似度時出錯: {e}")
        return {"similarity": 0.0}
