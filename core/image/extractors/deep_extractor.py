"""
深度特徵提取器

此模組提供深度學習特徵提取功能，包括：
- MobileNetV3 模型載入與管理
- 批次特徵提取
- 特徵快取機制
- GPU 加速支援
"""

import os
import time
import torch
import numpy as np
from PIL import Image
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from utils.logger import logger
from utils.gpu_utils import gpu_manager
from ..utils import _autocrop_letterbox
from ..config import DEEP_FEATURE_CONFIG, BATCH_CONFIG

# ======================== 全域變數 ========================
_image_model: Optional[torch.nn.Module] = None
_transform: Optional[transforms.Compose] = None
_model_loaded: bool = False
_feature_cache: dict[str, np.ndarray] = {}


def get_image_model():
    """載入 MobileNetV3-Large 模型與前處理流程。

    功能：
        - 載入預訓練的 MobileNetV3-Large 模型
        - 移除最後一層分類器
        - 設定適當的前處理流程
        - 支援 GPU 加速

    Returns:
        Tuple[torch.nn.Module, transforms.Compose]: (模型, 前處理流程)

    Note:
        - 使用 ImageNet 預訓練權重
        - 會記錄載入時間
        - 支援 GPU 和 CPU 模式
    """
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
                transforms.Resize(DEEP_FEATURE_CONFIG['resize_size']),
                transforms.CenterCrop(DEEP_FEATURE_CONFIG['crop_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=DEEP_FEATURE_CONFIG['normalize_mean'],
                                     std=DEEP_FEATURE_CONFIG['normalize_std'])
            ])
            _model_loaded = True
            logger.info(f"模型載入完成，耗時: {time.time() - start_time:.2f}秒")
        except Exception as e:
            logger.error(f"載入模型時出錯: {str(e)}")
            raise
    return _image_model, _transform


def _load_one(transform, p: str):
    """載入並預處理單張圖像。

    功能：
        - 載入圖像並轉換為 RGB 格式
        - 應用自動裁切去除邊框
        - 使用指定的變換流程處理

    Args:
        transform: 圖像變換流程
        p (str): 圖像檔案路徑

    Returns:
        torch.Tensor: 處理後的圖像張量
    """
    img = Image.open(p).convert('RGB')
    img = _autocrop_letterbox(img)  # 去黑邊/字幕（可視情況調參）
    return transform(img)


def _process_batch_tensors(tensors: List[torch.Tensor], batch_size: int,
                           model: torch.nn.Module, device: torch.device,
                           block: List[str]) -> None:
    """處理批次張量並計算特徵。

    功能：
        - 將張量分批處理以節省記憶體
        - 使用模型計算特徵向量
        - 將結果存入全域快取
        - 支援 GPU 加速

    Args:
        tensors (List[torch.Tensor]): 張量列表
        batch_size (int): 批次大小
        model (torch.nn.Module): 特徵提取模型
        device (torch.device): 計算設備
        block (List[str]): 對應的檔案路徑列表
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
    """載入並處理圖像批次。

    功能：
        - 並行載入多張圖像
        - 分批處理以節省記憶體
        - 計算深度特徵並快取
        - 自動清理 GPU 記憶體

    Args:
        non_cached (List[str]): 未快取的路徑列表
        transform: 圖像轉換器
        batch_size (int): 批次大小
        model (torch.nn.Module): 特徵提取模型
        device (torch.device): 計算設備

    Note:
        - 使用 ThreadPoolExecutor 進行並行載入
        - 會自動清理 GPU 記憶體
    """
    chunk = max(batch_size * 2, BATCH_CONFIG['image_processing_chunk_size'])
    for s in range(0, len(non_cached), chunk):
        block = non_cached[s:s + chunk]

        # 並行載入圖像
        with ThreadPoolExecutor(max_workers=min(BATCH_CONFIG['max_workers'], os.cpu_count() or 4)) as ex:
            tensors = list(ex.map(lambda p: _load_one(transform, p), block))

        # 批次處理特徵
        _process_batch_tensors(tensors, batch_size, model, device, block)
        gpu_manager.clear_gpu_memory()


def compute_batch_embeddings(image_paths: List[str], batch_size: int = 64) -> Optional[np.ndarray]:
    """批次計算圖像深度嵌入向量。

    功能：
        - 批次計算多張圖像的深度特徵
        - 支援快取機制避免重複計算
        - 保持原始順序進行處理
        - 使用並行載入提升效能

    Args:
        image_paths (List[str]): 圖像檔案路徑列表
        batch_size (int, optional): 批次大小。預設為 64。

    Returns:
        Optional[np.ndarray]: 特徵向量陣列，失敗時回傳 None

    Note:
        - 只對未快取的路徑進行計算
        - 會記錄快取缺漏的錯誤
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
