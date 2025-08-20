# utils/gpu_utils.py
import torch
from typing import Optional
from utils.logger import logger


class GPUManager:
    """GPU 管理器"""

    def __init__(self) -> None:
        self.initialized: bool = False
        self._device: torch.device | None = None

    def initialize(self) -> None:
        if self.initialized:
            return

        # 確保 CUDA 可用性檢查
        if torch.cuda.is_available():
            try:
                # 測試 CUDA 是否真正可用
                test_tensor = torch.tensor([1.0]).cuda()
                _ = test_tensor + 1
                del test_tensor

                self._device = torch.device("cuda:0")
                torch.cuda.empty_cache()
                torch.cuda.set_device(self._device.index)

                logger.info(
                    f"使用 GPU: {torch.cuda.get_device_name(self._device.index)}"
                )
            except Exception as e:
                logger.warning(f"CUDA 初始化失敗，使用 CPU: {e}")
                self._device = torch.device("cpu")
        else:
            logger.warning("未檢測到可用的 GPU，將使用 CPU 進行處理")
            self._device = torch.device("cpu")

        self.initialized = True

    def get_device(self) -> torch.device:
        if not self.initialized:
            self.initialize()
        return self._device

    def clear_gpu_memory(self) -> None:
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()
            # 強制同步以確保清理完成
            torch.cuda.synchronize()


# 建立全域 GPU 管理器實例
gpu_manager: GPUManager = GPUManager()
