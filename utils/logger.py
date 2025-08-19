# utils/logger.py
import os
import sys
import time
import json
import logging
import platform
from typing import Optional, Type
from logging.handlers import TimedRotatingFileHandler

# === 環境變數開關 ===
# 是否讓 console 使用純訊息（不帶時間戳）；通常由上層統一加時間戳時使用
PLAIN_CONSOLE = os.getenv("PLAIN_CONSOLE_LOG", "0") in ("1", "true", "True")
# 若開啟，console log 走 stderr，stdout 專供 emit() 的 NDJSON 事件
JSON_EVENTS_TO_STDOUT = os.getenv("JSON_EVENTS_TO_STDOUT", "0") in ("1", "true", "True")

# === 檔案與格式設定 ===
log_dir: str = "logs"
os.makedirs(log_dir, exist_ok=True)

# 檔案格式（保留時間戳）
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# 檔案輪轉（每天 1 份，保留 7 天）
file_handler = TimedRotatingFileHandler(
    filename=os.path.join(log_dir, "app.log"),
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8",
    utc=False
)
# 產出 app.log.YYYY-MM-DD
file_handler.suffix = "%Y-%m-%d"
file_handler.setFormatter(file_formatter)

# Console handler：當需要讓 stdout 保持乾淨（給 NDJSON 事件）時，console 改用 stderr
_console_stream = sys.stderr if JSON_EVENTS_TO_STDOUT else sys.stdout
console_handler: logging.StreamHandler = logging.StreamHandler(_console_stream)
if PLAIN_CONSOLE:
    console_handler.setFormatter(logging.Formatter("%(message)s"))  # 純訊息
else:
    console_handler.setFormatter(file_formatter)  # 獨立 CLI 時也保留時間戳

# === 建立 logger ===
logger: logging.Logger = logging.getLogger("video_similarity")
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False  # 避免往 root 傳播造成重複輸出

# === NDJSON 事件輸出（供前端 / SSE 消費） ===


def emit(event_type: str, **kw) -> None:
    """
    輸出一行 JSON（NDJSON），寫到 stdout。
    - event_type: 事件種類（如 "progress" / "log" / "error" / "done" / "hello"）
    - 其餘欄位原樣附帶；不可序列化時退回 str，避免整體失敗
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
    """回傳可用於 with 區塊的性能記錄器。"""
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


__all__ = [
    "logger",
    "emit",
    "PerformanceLogger",
    "log_performance",
    "log_gpu_info",
    "log_system_info",
]
