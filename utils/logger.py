# utils/logger.py
"""
檔案用途：統一日誌記錄系統

此模組提供完整的日誌記錄功能，包括：
- 檔案與控制台日誌輸出（按日期分檔）
- NDJSON 事件串流（供前端 SSE 使用）
- 性能監控與系統資訊記錄

主要功能：
- logger: 標準日誌記錄器
- emit: NDJSON 事件輸出
- PerformanceLogger: 性能監控
- 各種系統資訊記錄函式
"""

import os
import sys
import time
import json
import logging
import platform
from datetime import datetime
from typing import Optional, Type

# === 環境變數開關 ===
# 是否讓 console 使用純訊息（不帶時間戳）；通常由上層統一加時間戳時使用
PLAIN_CONSOLE = os.getenv("PLAIN_CONSOLE_LOG", "0") in ("1", "true", "True")
# 若開啟，console log 走 stderr，stdout 專供 emit() 的 NDJSON 事件
JSON_EVENTS_TO_STDOUT = os.getenv("JSON_EVENTS_TO_STDOUT", "0") in ("1", "true", "True")

# === 檔案與格式設定 ===
log_dir: str = "logs"
os.makedirs(log_dir, exist_ok=True)

# 日誌格式設定
detailed_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
plain_formatter = logging.Formatter("%(message)s")

# 使用日期作為檔案名稱，避免輪轉時的檔案鎖定問題
log_filename = f"app_{datetime.now().strftime('%Y-%m-%d')}.log"
file_handler = logging.FileHandler(
    filename=os.path.join(log_dir, log_filename),
    encoding="utf-8"
)
file_handler.setFormatter(detailed_formatter)

# Console handler：當需要讓 stdout 保持乾淨（給 NDJSON 事件）時，console 改用 stderr
_console_stream = sys.stderr if JSON_EVENTS_TO_STDOUT else sys.stdout
console_handler: logging.StreamHandler = logging.StreamHandler(_console_stream)
console_handler.setFormatter(plain_formatter if PLAIN_CONSOLE else detailed_formatter)

# === 建立 logger ===
logger: logging.Logger = logging.getLogger("video_similarity")
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False  # 避免往 root 傳播造成重複輸出

# === NDJSON 事件輸出（供前端 / SSE 消費） ===


def emit(event_type: str, **kw) -> None:
    """輸出一行 JSON（NDJSON），寫到 stdout。

    功能：
        - 將事件資料序列化為 JSON 格式
        - 輸出到標準輸出（供前端 SSE 消費）
        - 處理序列化錯誤，確保輸出不中斷

    Args:
        event_type (str): 事件種類（如 "progress" / "log" / "error" / "done" / "hello"）
        **kw: 其他事件資料欄位

    Note:
        - 使用 NDJSON 格式，每行一個 JSON 物件
        - 不可序列化的值會轉為字串
        - 會立即刷新輸出緩衝區
    """
    obj = {"type": event_type, **kw}
    try:
        print(json.dumps(obj, ensure_ascii=False), flush=True)  # stdout
    except Exception as e:
        # fallback：將不可序列化值轉為字串，維持管線不中斷
        obj["__err"] = f"emit-serde:{e}"
        obj["payload"] = {k: str(v) for k, v in kw.items()}
        print(json.dumps(obj, ensure_ascii=False), flush=True)

# === 性能與系統資訊輔助 ===


class PerformanceLogger:
    """性能監控記錄器，用於測量代碼執行時間。

    此類別提供上下文管理器功能，自動記錄代碼區塊的執行時間。

    Attributes:
        name (str): 性能監控的名稱標識
        start_time (Optional[float]): 開始時間戳
    """

    def __init__(self, name: str) -> None:
        """初始化性能記錄器。

        Args:
            name (str): 性能監控的名稱標識
        """
        self.name: str = name
        self.start_time: Optional[float] = None

    def __enter__(self) -> "PerformanceLogger":
        """進入上下文管理器，記錄開始時間。

        Returns:
            PerformanceLogger: 自身實例
        """
        self.start_time = time.time()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object]
    ) -> None:
        """退出上下文管理器，計算並記錄執行時間。

        Args:
            exc_type: 例外類型
            exc_val: 例外值
            exc_tb: 例外追蹤
        """
        if self.start_time:
            elapsed: float = time.time() - self.start_time
            logger.info(f"{self.name} 耗時: {elapsed:.2f}秒")


def log_performance(name: str) -> PerformanceLogger:
    """回傳可用於 with 區塊的性能記錄器。

    Args:
        name (str): 性能監控的名稱標識

    Returns:
        PerformanceLogger: 性能記錄器實例

    Example:
        ```python
        with log_performance("資料處理"):
            # 執行需要監控的代碼
            process_data()
        ```
    """
    return PerformanceLogger(name)


def log_gpu_info() -> None:
    """記錄 GPU 相關資訊。

    功能：
        - 檢查 PyTorch 是否可用
        - 檢測 CUDA 設備數量與名稱
        - 記錄 GPU 狀態到日誌

    Note:
        - 如果 PyTorch 未安裝，會記錄警告訊息
        - 如果 GPU 不可用，會記錄使用 CPU 模式
    """
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
    """記錄系統相關資訊。

    功能：
        - 記錄作業系統版本
        - 記錄 Python 版本
        - 記錄當前工作目錄

    Note:
        - 用於除錯與環境確認
        - 在應用程式啟動時建議呼叫
    """
    logger.info(f"作業系統: {platform.system()} {platform.release()}")
    logger.info(f"Python 版本: {sys.version}")
    logger.info(f"工作目錄: {os.getcwd()}")


__all__ = [
    "logger",
    "emit",
    "PerformanceLogger",
    "log_performance",
    "log_gpu_info",
    "log_system_info",
]
