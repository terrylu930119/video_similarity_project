import torch
from typing import Optional
from utils.logger import logger
class GPUManager:
    """GPU 管理器，用於統一管理 GPU 相關功能"""

    _instance: Optional["GPUManager"] = None
    _initialized: bool = False
    _cuda_checked: bool = False

    def __new__(cls) -> "GPUManager":
        if cls._instance is None:
            cls._instance = super(GPUManager, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device: torch.device = torch.device("cuda:0")
        else:
            self.device: torch.device = torch.device("cpu")
        self.initialized: bool = False

    def initialize(self) -> None:
        if self.initialized:
            return

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_device(self.device.index)
            logger.info(f"使用 GPU: {torch.cuda.get_device_name(self.device.index)}")
            logger.info(f"當前 GPU 內存使用量: {torch.cuda.memory_allocated(self.device.index) / 1024**2:.2f} MB")
            logger.info(f"當前 GPU 緩存使用量: {torch.cuda.memory_reserved(self.device.index) / 1024**2:.2f} MB")
        else:
            logger.warning("未檢測到可用的 GPU，將使用 CPU 進行處理")

        self.initialized = True

    def get_device(self) -> torch.device:
        return self.device

    def clear_gpu_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def log_gpu_memory(self) -> None:
        if torch.cuda.is_available():
            logger.info(f"當前 GPU 內存使用量: {torch.cuda.memory_allocated(self.device.index) / 1024**2:.2f} MB")
            logger.info(f"當前 GPU 緩存使用量: {torch.cuda.memory_reserved(self.device.index) / 1024**2:.2f} MB")


# 建立全域 GPU 管理器實例
gpu_manager: GPUManager = GPUManager()
