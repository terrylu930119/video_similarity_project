"""
文本嵌入提取器

此模組提供文本語義嵌入提取功能，包括：
- Sentence Transformers 模型載入與管理
- 文本嵌入計算與快取
- 多語言文本處理
- 嵌入相似度計算
"""

import torch
import inspect
import threading
import numpy as np
from torch import Tensor
from utils.logger import logger
from utils.gpu_utils import gpu_manager
from typing import Dict, List, Optional, Tuple, Any
from sentence_transformers import SentenceTransformer, util
from ..utils import is_meaningful_text

# ======================== 全域變數與常數 ========================
_sentence_transformer: Optional[SentenceTransformer] = None
_st_lock = threading.Lock()


def _setup_model_device(model: torch.nn.Module, device: str, dtype: torch.dtype) -> torch.nn.Module:
    """設定模型的裝置和資料類型。

    功能：
        - 將模型移動到指定裝置
        - 設定適當的資料類型
        - 優先使用 to_empty() 方法（如果可用）

    Args:
        model (torch.nn.Module): 要設定的模型
        device (str): 目標裝置
        dtype (torch.dtype): 目標資料類型

    Returns:
        torch.nn.Module: 設定後的模型
    """
    if hasattr(model, "to_empty"):
        model.to_empty(device=device, dtype=dtype)
    else:
        model.to(device=device, dtype=dtype)
    return model


def _load_and_process_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    """載入並處理檢查點檔案。

    功能：
        - 載入 PyTorch 檢查點檔案
        - 處理 state_dict 包裝
        - 移除 DataParallel 的 'module.' 前綴

    Args:
        ckpt_path (str): 檢查點檔案路徑

    Returns:
        Dict[str, Any]: 處理後的狀態字典
    """
    state = torch.load(ckpt_path, map_location="cpu")

    # 處理 state_dict 包裝
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    # 處理 DataParallel 的 'module.' 前綴
    if isinstance(state, dict) and state and all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}

    return state


def _load_state_dict_with_assign(model: torch.nn.Module, state: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """使用 assign=True 載入狀態字典（如果可用）。

    功能：
        - 嘗試使用 assign=True 載入狀態字典
        - 回退到標準載入方法
        - 提供載入結果的詳細資訊

    Args:
        model (torch.nn.Module): 要載入的模型
        state (Dict[str, Any]): 狀態字典

    Returns:
        Tuple[List[str], List[str]]: (缺失的鍵, 意外的鍵)
    """
    sig = inspect.signature(model.load_state_dict)
    try:
        if "assign" in sig.parameters:
            missing, unexpected = model.load_state_dict(state, assign=True)
        else:
            missing, unexpected = model.load_state_dict(state)
    except TypeError:
        missing, unexpected = model.load_state_dict(state)

    return missing, unexpected


def safe_load_module(model: torch.nn.Module, ckpt_path: str,
                     device: str = None, dtype: torch.dtype = torch.float32) -> torch.nn.Module:
    """
    通用安全載入：
    1) 優先用 to_empty() 在正確 device/dtype 分配參數
    2) load_state_dict(assign=True)（PyTorch 2.4+）避免 meta no-op
    3) 兼容 DataParallel 的 'module.' 前綴
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 分配參數記憶體
    model = _setup_model_device(model, device, dtype)

    # 2) 讀取並處理檢查點
    state = _load_and_process_checkpoint(ckpt_path)

    # 3) 載入狀態字典
    missing, unexpected = _load_state_dict_with_assign(model, state)

    if missing or unexpected:
        print(f"[safe_load_module] missing={missing}, unexpected={unexpected}")

    model.eval()
    return model


def _create_cpu_model() -> SentenceTransformer:
    """在 CPU 上建立 SentenceTransformer 模型。

    功能：
        - 在 CPU 上載入 SentenceTransformer 模型
        - 進行基本健康檢查
        - 確保模型權重已實體化

    Returns:
        SentenceTransformer: CPU 上的模型實例

    Raises:
        RuntimeError: 當模型權重未實體化或包含無效值時
    """
    torch.set_grad_enabled(False)
    cpu_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="cpu")

    with torch.inference_mode():
        test_embed: Tensor = cpu_model.encode("warmup_materialize", convert_to_tensor=True)

    # 基本健康檢查
    if getattr(test_embed, "is_meta", False):
        raise RuntimeError("模型權重尚未實體化 (meta tensor)")
    if not torch.isfinite(test_embed).all():
        raise RuntimeError("模型初始化失敗：embedding 含有 NaN 或 Inf")

    return cpu_model


def _move_model_to_target_device(cpu_model: SentenceTransformer, target: str) -> SentenceTransformer:
    """將模型移動到目標裝置。

    功能：
        - 將 CPU 上的模型移動到目標裝置
        - 支援 CPU 和 CUDA 裝置

    Args:
        cpu_model (SentenceTransformer): CPU 上的模型
        target (str): 目標裝置名稱

    Returns:
        SentenceTransformer: 移動後的模型
    """
    if target == "cpu":
        return cpu_model
    else:
        return cpu_model.to(target)


def _fallback_to_cpu() -> SentenceTransformer:
    """回退到 CPU 載入模型。

    功能：
        - 當 GPU 載入失敗時回退到 CPU
        - 清理 GPU 記憶體
        - 確保模型能正常載入

    Returns:
        SentenceTransformer: CPU 上的模型實例
    """
    logger.error("載入 SentenceTransformer 失敗，回退 CPU")

    # 清顯存避免殘留
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # 最終保底：CPU 直載
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="cpu")
    model.eval()
    return model


def get_sentence_transformer() -> SentenceTransformer:
    """
    穩健載入：CPU 先實體化避免 meta → 視需要搬到 CUDA。
    失敗時清顯存並回退 CPU。
    """
    global _sentence_transformer
    if _sentence_transformer is not None:
        return _sentence_transformer

    with _st_lock:
        if _sentence_transformer is not None:
            return _sentence_transformer

        # 目標裝置字串化
        dev_obj = gpu_manager.get_device()
        target = "cuda" if getattr(dev_obj, "type", "cpu") == "cuda" and torch.cuda.is_available() else "cpu"
        logger.info(f"載入模型中，目標設備：{target}")

        try:
            # 1) 先在 CPU 完整實體化，徹底避開 meta tensor
            cpu_model = _create_cpu_model()

            # 2) 如需 CUDA，再搬裝置（此時已無 meta）
            model = _move_model_to_target_device(cpu_model, target)
            model.eval()

            _sentence_transformer = model
            logger.info(f"模型載入成功，裝置：{_sentence_transformer.device}")
            return _sentence_transformer

        except Exception as e:
            logger.error(f"載入 SentenceTransformer 失敗，回退 CPU：{e}")
            _sentence_transformer = _fallback_to_cpu()
            return _sentence_transformer



def _compute_text_embedding_batch(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """批次計算文本嵌入。

    功能：
        - 批次處理多個文本
        - 提升計算效率
        - 處理批次大小限制

    Args:
        texts (List[str]): 文本列表
        model (SentenceTransformer): 嵌入模型

    Returns:
        np.ndarray: 文本嵌入矩陣

    Note:
        - 會自動處理批次大小限制
        - 會處理空文本的情況
    """
    if not texts:
        return np.array([])
    
    # 過濾空文本
    valid_texts = [text for text in texts if text and text.strip()]
    if not valid_texts:
        return np.array([])
    
    try:
        # 批次計算嵌入
        embeddings = model.encode(valid_texts, convert_to_tensor=False, show_progress_bar=False)
        return embeddings
    except Exception as e:
        logger.error(f"批次計算文本嵌入失敗：{e}")
        return np.array([])


def _check_embedding_validity(embedding: torch.Tensor) -> bool:
    """檢查嵌入向量的有效性"""
    if embedding is None:
        return False
    if hasattr(embedding, 'is_meta') and embedding.is_meta:
        return False
    if not torch.isfinite(embedding).all():
        return False
    return True


def compute_text_embedding(text: str, model: Optional[SentenceTransformer] = None) -> Optional[torch.Tensor]:
    """計算文本向量，避免 meta tensor 與 NaN 問題，並可自動重載模型。"""
    global _sentence_transformer

    try:
        if model is None:
            model = get_sentence_transformer()

        with torch.no_grad():
            embedding = model.encode(text, convert_to_tensor=True)

            # 檢查是否為 meta tensor
            if not _check_embedding_validity(embedding):
                logger.warning("檢測到 meta tensor，重新載入模型編碼")
                _sentence_transformer = None
                model = get_sentence_transformer()
                embedding = model.encode(text, convert_to_tensor=True)

            # 檢查 tensor 合法性
            if not _check_embedding_validity(embedding):
                logger.error("嵌入向量含 NaN 或 Inf，回傳 None")
                return None

            return embedding

    except Exception as e:
        logger.error(f"計算文本嵌入向量時出錯: {str(e)}")
        return None


def compute_text_embeddings_batch(texts: List[str], model: Optional[SentenceTransformer] = None) -> List[Optional[np.ndarray]]:
    """批次計算多個文本的語義嵌入。

    功能：
        - 批次處理多個文本
        - 提升計算效率
        - 處理批次大小限制

    Args:
        texts (List[str]): 文本列表
        model (Optional[SentenceTransformer]): 嵌入模型，預設使用快取模型

    Returns:
        List[Optional[np.ndarray]]: 文本嵌入向量列表

    Note:
        - 會自動載入模型（如果未提供）
        - 會過濾無效文本
        - 支援批次處理
    """
    if not texts:
        return []
    
    try:
        if model is None:
            model = get_sentence_transformer()
        
        # 過濾有效文本
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                is_meaningful, _ = is_meaningful_text(text)
                if is_meaningful:
                    valid_texts.append(text)
                    valid_indices.append(i)
        
        if not valid_texts:
            return [None] * len(texts)
        
        # 批次計算嵌入
        embeddings = _compute_text_embedding_batch(valid_texts, model)
        if len(embeddings) == 0:
            return [None] * len(texts)
        
        # 重建完整結果
        result = [None] * len(texts)
        for i, embedding in enumerate(embeddings):
            if i < len(valid_indices):
                result[valid_indices[i]] = embedding
        
        logger.debug(f"批次計算文本嵌入成功，有效文本：{len(valid_texts)}/{len(texts)}")
        return result
    except Exception as e:
        logger.error(f"批次計算文本嵌入失敗：{e}")
        return [None] * len(texts)



def compute_text_similarities_batch(texts1: List[str], texts2: List[str], 
                                   model: Optional[SentenceTransformer] = None) -> List[Optional[float]]:
    """批次計算多個文本對的語義相似度。

    功能：
        - 批次處理多個文本對
        - 提升計算效率
        - 處理批次大小限制

    Args:
        texts1 (List[str]): 第一個文本列表
        texts2 (List[str]): 第二個文本列表
        model (Optional[SentenceTransformer]): 嵌入模型，預設使用快取模型

    Returns:
        List[Optional[float]]: 相似度分數列表

    Note:
        - 會自動載入模型（如果未提供）
        - 會過濾無效文本
        - 支援批次處理
    """
    if not texts1 or not texts2 or len(texts1) != len(texts2):
        return []
    
    try:
        if model is None:
            model = get_sentence_transformer()
        
        # 計算嵌入
        embeddings1 = compute_text_embeddings_batch(texts1, model)
        embeddings2 = compute_text_embeddings_batch(texts2, model)
        
        # 計算相似度
        similarities = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            if emb1 is not None and emb2 is not None:
                similarity = util.cos_sim(emb1, emb2).item()
                similarities.append(float(similarity))
            else:
                similarities.append(None)
        
        logger.debug(f"批次計算文本相似度成功，有效對數：{len([s for s in similarities if s is not None])}/{len(similarities)}")
        return similarities
    except Exception as e:
        logger.error(f"批次計算文本相似度失敗：{e}")
        return [None] * len(texts1)
