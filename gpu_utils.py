import torch
import cv2
import numpy as np
from logger import logger, log_performance
import os
from dependencies import check_cuda

class GPUManager:
    """GPU 管理器，用於統一管理 GPU 相關功能"""
    
    _instance = None
    _initialized = False
    _cuda_checked = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not GPUManager._initialized:
            self._init_gpu()
            GPUManager._initialized = True
    
    def _init_gpu(self):
        """初始化 GPU 相關設定"""
        # 使用 dependencies.py 中的檢查結果
        self.pytorch_cuda_available = self._check_cuda()
        if self.pytorch_cuda_available:
            self.pytorch_device = torch.device("cuda")
            self.pytorch_device_count = torch.cuda.device_count()
            self.pytorch_device_name = torch.cuda.get_device_name(0)
            
            # 設定 CUDA 效能最佳化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # 設定 CUDA 記憶體分配器
            torch.cuda.set_per_process_memory_fraction(0.8)  # 限制 GPU 記憶體使用
            
            if not GPUManager._cuda_checked:
                logger.info(f"PyTorch CUDA 可用: 設備數量={self.pytorch_device_count}, 設備名稱={self.pytorch_device_name}")
                GPUManager._cuda_checked = True
        else:
            self.pytorch_device = torch.device("cpu")
            if not GPUManager._cuda_checked:
                logger.warning("PyTorch CUDA 不可用，使用 CPU")
                GPUManager._cuda_checked = True
    
    def _check_cuda(self):
        """內部檢查 CUDA 可用性"""
        return torch.cuda.is_available()
    
    def get_pytorch_device(self):
        """獲取 PyTorch 設備"""
        return self.pytorch_device
    
    def is_pytorch_cuda_available(self):
        """檢查 PyTorch CUDA 是否可用"""
        return self.pytorch_cuda_available
    
    def to_gpu(self, model):
        """將 PyTorch 模型移至 GPU"""
        if self.pytorch_cuda_available:
            model = model.to(self.pytorch_device)
            return model
        return model
    
    def to_cpu(self, tensor):
        """將 PyTorch 張量移至 CPU"""
        if self.pytorch_cuda_available and tensor.is_cuda:
            return tensor.cpu()
        return tensor
    
    def clear_gpu_memory(self):
        """清理 GPU 記憶體"""
        if self.pytorch_cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def optimize_for_inference(self, model):
        """最佳化模型用於推理"""
        if self.pytorch_cuda_available:
            model = model.to(self.pytorch_device)
            model.eval()
            with torch.no_grad():
                return model
        return model
    
    def create_cuda_stream(self):
        """建立 CUDA 流"""
        if self.pytorch_cuda_available:
            return torch.cuda.Stream()
        return None

# 建立全域 GPU 管理器實例
gpu_manager = GPUManager() 