import torch
from utils.logger import logger
import os

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialized = False
        
    def initialize(self):
        if self.initialized:
            return
            
        if torch.cuda.is_available():
            # 清理 GPU 緩存
            torch.cuda.empty_cache()
            
            # 設置當前設備
            torch.cuda.set_device(self.device)
            
            # 打印 GPU 信息
            logger.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"當前 GPU 內存使用量: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            logger.info(f"當前 GPU 緩存使用量: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        else:
            logger.warning("未檢測到可用的 GPU，將使用 CPU 進行處理")
            
        self.initialized = True
    
    def get_device(self):
        return self.device
    
    def clear_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("已清理 GPU 內存")
            
    def log_gpu_memory(self):
        if torch.cuda.is_available():
            logger.info(f"當前 GPU 內存使用量: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            logger.info(f"當前 GPU 緩存使用量: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# 建立全域 GPU 管理器實例
gpu_manager = GPUManager() 