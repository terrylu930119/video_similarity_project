import os
import sys
import time
import logging

# 創建日誌目錄
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "app.log")

# 配置日誌格式
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# 配置文件處理器
file_handler = logging.FileHandler(log_path, encoding='utf-8')
file_handler.setFormatter(formatter)

# 配置控制台處理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

# 創建日誌記錄器
logger = logging.getLogger("video_similarity")
logger.setLevel(logging.INFO)

# 清除現有的處理器
logger.handlers = []

# 添加處理器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 添加性能監控功能
class PerformanceLogger:
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = time.time() - self.start_time
            logger.info(f"{self.name} 耗時: {elapsed:.2f}秒")

def log_performance(name: str) -> PerformanceLogger:
    """用於記錄函數執行時間的裝飾器"""
    return PerformanceLogger(name)

def log_gpu_info():
    """記錄 GPU 相關資訊"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU 資訊: 設備數量={device_count}, 設備名稱={device_name}")
        else:
            logger.info("GPU 不可用，使用 CPU 模式")
    except ImportError:
        logger.warning("PyTorch 未安裝，無法獲取 GPU 資訊")

def log_system_info():
    """記錄系統相關資訊"""
    import platform
    logger.info(f"作業系統: {platform.system()} {platform.release()}")
    logger.info(f"Python 版本: {sys.version}")
    logger.info(f"工作目錄: {os.getcwd()}")