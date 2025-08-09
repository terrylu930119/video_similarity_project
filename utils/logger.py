import os
import sys
import time
import logging
import platform
from typing import Optional, Type
from logging.handlers import TimedRotatingFileHandler  # 新增

# 創建日誌目錄
log_dir: str = "logs"
os.makedirs(log_dir, exist_ok=True)

# 通用格式（給檔案用，保留時間戳）
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# console 是否用「純訊息」(交由上層統一套時間戳)
PLAIN_CONSOLE = os.getenv("PLAIN_CONSOLE_LOG", "0") in ("1", "true", "True")

# 使用 TimedRotatingFileHandler（保留時間戳）
file_handler = TimedRotatingFileHandler(
    filename=os.path.join(log_dir, "app.log"),
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8",
    utc=False
)
# 設定檔名後綴，產出 app.log.YYYY-MM-DD 格式
file_handler.suffix = "%Y-%m-%d"
file_handler.setFormatter(file_formatter)

# 控制台處理器：依環境變數切換格式
console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
if PLAIN_CONSOLE:
    console_handler.setFormatter(logging.Formatter("%(message)s"))  # ← 純訊息
else:
    console_handler.setFormatter(file_formatter)  # ← 獨立跑 CLI 時保留時間戳

# 創建日誌記錄器
logger: logging.Logger = logging.getLogger("video_similarity")
logger.setLevel(logging.INFO)

# 清除現有處理器並添加新的
logger.handlers = []
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 避免往 root 傳播導致重印
logger.propagate = False


# 添加性能監控功能
class PerformanceLogger:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.start_time: Optional[float] = None

    def __enter__(self) -> "PerformanceLogger":
        self.start_time = time.time()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object]
    ) -> None:
        if self.start_time:
            elapsed: float = time.time() - self.start_time
            logger.info(f"{self.name} 耗時: {elapsed:.2f}秒")


def log_performance(name: str) -> PerformanceLogger:
    """用於記錄函數執行時間的裝飾器"""
    return PerformanceLogger(name)


def log_gpu_info() -> None:
    """記錄 GPU 相關資訊"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count: int = torch.cuda.device_count()
            device_name: str = torch.cuda.get_device_name(0)
            logger.info(f"GPU 資訊: 設備數量={device_count}, 設備名稱={device_name}")
        else:
            logger.info("GPU 不可用，使用 CPU 模式")
    except ImportError:
        logger.warning("PyTorch 未安裝，無法獲取 GPU 資訊")


def log_system_info() -> None:
    """記錄系統相關資訊"""
    logger.info(f"作業系統: {platform.system()} {platform.release()}")
    logger.info(f"Python 版本: {sys.version}")
    logger.info(f"工作目錄: {os.getcwd()}")
