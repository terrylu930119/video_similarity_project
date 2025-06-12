import os
import cv2
import torch
import subprocess
from utils.logger import logger

# =============== 標記變數 ===============
_checked_ffmpeg = False
_checked_yt_dlp = False
_checked_cuda = False

# =============== 安裝需求 ===============
def install_requirements():
    os.system("pip install -r requirements.txt")

# =============== 套件可用性檢查 ===============
def check_ffmpeg():
    global _checked_ffmpeg
    if _checked_ffmpeg:
        return True

    try:
        subprocess.run(["ffmpeg", "-version"], check=True)
        logger.info("ffmpeg 可用")
        _checked_ffmpeg = True
        return True
    except subprocess.CalledProcessError:
        logger.error("請安裝 ffmpeg：https://ffmpeg.org/download.html")
        return False

def check_yt_dlp():
    global _checked_yt_dlp
    if _checked_yt_dlp:
        return True

    try:
        subprocess.run(["yt-dlp", "--version"], check=True)
        logger.info("yt-dlp 可用")
        _checked_yt_dlp = True
        return True
    except subprocess.CalledProcessError:
        logger.error("請安裝 yt-dlp：pip install yt-dlp")
        return False

def check_cuda():
    global _checked_cuda
    if _checked_cuda:
        return True

    pytorch_cuda_available = torch.cuda.is_available()
    if pytorch_cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"PyTorch CUDA 可用: 設備數量={device_count}, 設備名稱={device_name}")
        _checked_cuda = True
        return True
    else:
        logger.warning("PyTorch CUDA 不可用，將使用 CPU 進行處理")
        _checked_cuda = True
        return False

def check_gpu_dependencies():
    """檢查 GPU 相關相依套件狀態"""
    logger.info(f"PyTorch 版本: {torch.__version__}")
    logger.info(f"OpenCV 版本: {cv2.__version__}")

    if check_cuda():
        logger.info("GPU 加速可用，將使用 GPU 進行處理")
        return True
    else:
        logger.warning("GPU 加速不可用，將使用 CPU 進行處理")
        logger.warning("如需使用 GPU，請安裝 CUDA 和 cuDNN")
        logger.warning("CUDA：https://developer.nvidia.com/cuda-downloads")
        logger.warning("cuDNN：https://developer.nvidia.com/cudnn")
        return False

if __name__ == "__main__":
    install_requirements()
    check_ffmpeg()
    check_yt_dlp()
    check_gpu_dependencies()
