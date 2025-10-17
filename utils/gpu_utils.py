# utils/gpu_utils.py
"""
檔案用途：GPU 資源管理與記憶體清理工具

此模組提供 GPU 資源管理功能，包括：
- CUDA 可用性檢測與初始化
- GPU 記憶體管理與清理
- 設備選擇與配置
- 記憶體洩漏防護

主要類別：
- GPUManager: GPU 資源管理器，提供統一的 GPU 操作介面
"""

import torch
from typing import Optional
from utils.logger import logger


class GPUManager:
    """GPU 管理器，負責 CUDA 設備初始化與記憶體管理。

    此類別提供統一的 GPU 資源管理介面，包括：
    - 自動檢測 CUDA 可用性
    - 安全的設備初始化
    - 記憶體清理與優化
    - 錯誤處理與回退機制

    Attributes:
        initialized (bool): 是否已完成初始化
        _device (torch.device | None): 當前使用的設備
    """

    def __init__(self) -> None:
        self.initialized: bool = False
        self._device: torch.device | None = None

    def initialize(self) -> None:
        """初始化 GPU 設備，檢測 CUDA 可用性並設定適當的設備。

        功能：
            - 檢查 CUDA 是否可用
            - 執行實際的 GPU 運算測試
            - 設定設備並清理記憶體
            - 失敗時自動回退到 CPU

        Raises:
            Exception: 當設備初始化過程中發生未預期錯誤時
        """
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
        """取得當前使用的設備。

        Returns:
            torch.device: 當前設定的設備（GPU 或 CPU）

        Note:
            - 如果尚未初始化，會自動執行初始化
            - 回傳的設備物件可直接用於 PyTorch 運算
        """
        if not self.initialized:
            self.initialize()
        return self._device

    def clear_gpu_memory(self) -> None:
        """清理 GPU 記憶體，釋放未使用的快取記憶體。

        功能：
            - 清空 CUDA 快取記憶體
            - 強制同步以確保清理完成
            - 僅在 GPU 設備上執行

        Note:
            - 此方法僅在 CUDA 設備上有效
            - 建議在大型模型運算後呼叫以釋放記憶體
        """
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()
            # 強制同步以確保清理完成
            torch.cuda.synchronize()


# 建立全域 GPU 管理器實例
gpu_manager: GPUManager = GPUManager()
