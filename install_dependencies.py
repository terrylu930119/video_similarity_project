#!/usr/bin/env python3
"""
Video Similarity Project - 自動依賴安裝腳本
自動檢查並安裝專案所需的所有依賴套件和外部工具
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import urllib.request
import zipfile
import tarfile

# =============== 顏色輸出 ===============


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_colored(text: str, color: str = Colors.ENDC) -> None:
    """彩色輸出"""
    print(f"{color}{text}{Colors.ENDC}")


def print_header(text: str) -> None:
    """標題輸出"""
    print_colored(f"\n{'='*60}", Colors.HEADER)
    print_colored(f"  {text}", Colors.HEADER)
    print_colored(f"{'='*60}", Colors.HEADER)


def print_success(text: str) -> None:
    """成功訊息"""
    print_colored(f"✅ {text}", Colors.OKGREEN)


def print_warning(text: str) -> None:
    """警告訊息"""
    print_colored(f"⚠️  {text}", Colors.WARNING)


def print_error(text: str) -> None:
    """錯誤訊息"""
    print_colored(f"❌ {text}", Colors.FAIL)


def print_info(text: str) -> None:
    """資訊訊息"""
    print_colored(f"ℹ️  {text}", Colors.OKBLUE)

# =============== 系統檢查 ===============


def get_system_info() -> Tuple[str, str, str]:
    """獲取系統資訊"""
    system = platform.system()
    machine = platform.machine()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return system, machine, python_version


def check_python_version() -> bool:
    """檢查 Python 版本"""
    print_header("檢查 Python 版本")

    system, machine, python_version = get_system_info()
    print_info(f"作業系統: {system} {machine}")
    print_info(f"Python 版本: {python_version}")

    if sys.version_info < (3, 8):
        print_error(f"Python 版本過舊 ({python_version})，需要 Python 3.8 或以上")
        return False

    print_success(f"Python 版本符合要求 ({python_version})")
    return True


def check_pip(venv_python: Path) -> bool:
    """檢查 pip 是否可用"""
    try:
        subprocess.run([str(venv_python), "-m", "pip", "--version"],
                       check=True, capture_output=True)
        print_success("pip 可用")
        return True
    except subprocess.CalledProcessError:
        print_error("pip 不可用，請先安裝 pip")
        return False

# =============== 虛擬環境管理 ===============


def create_virtual_environment(python_executable: str) -> bool:
    """建立虛擬環境"""
    print_header("建立虛擬環境")
    venv_path = Path(".venv")
    if venv_path.exists():
        print_info("虛擬環境已存在")
        return True
    try:
        print_info("建立虛擬環境...")
        subprocess.run([python_executable, "-m", "venv", ".venv"], check=True)
        print_success("虛擬環境建立成功")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"建立虛擬環境失敗: {e}")
        return False


def get_venv_python() -> Optional[Path]:
    """獲取虛擬環境中的 Python 路徑"""
    system = platform.system()
    if system == "Windows":
        python_path = Path(".venv/Scripts/python.exe")
    else:
        python_path = Path(".venv/bin/python")
    return python_path if python_path.exists() else None


# =============== 外部工具安裝 ===============


def install_ffmpeg() -> bool:
    """安裝 FFmpeg"""
    print_header("安裝 FFmpeg")

    # 檢查是否已安裝
    if shutil.which("ffmpeg"):
        print_success("FFmpeg 已安裝")
        return True

    system = platform.system()

    if system == "Windows":
        return install_ffmpeg_windows()
    elif system == "Darwin":  # macOS
        return install_ffmpeg_macos()
    elif system == "Linux":
        return install_ffmpeg_linux()
    else:
        print_error(f"不支援的作業系統: {system}")
        return False


def install_ffmpeg_windows() -> bool:
    """Windows 安裝 FFmpeg"""
    try:
        print_info("使用 chocolatey 安裝 FFmpeg...")
        # 檢查是否有 chocolatey
        if not shutil.which("choco"):
            print_warning("未找到 chocolatey，請手動安裝 FFmpeg")
            print_info("下載地址: https://ffmpeg.org/download.html")
            return False

        subprocess.run(["choco", "install", "ffmpeg", "-y"], check=True)
        print_success("FFmpeg 安裝成功")
        return True
    except subprocess.CalledProcessError:
        print_warning("chocolatey 安裝失敗，請手動安裝 FFmpeg")
        print_info("下載地址: https://ffmpeg.org/download.html")
        return False


def install_ffmpeg_macos() -> bool:
    """macOS 安裝 FFmpeg"""
    try:
        print_info("使用 Homebrew 安裝 FFmpeg...")
        subprocess.run(["brew", "install", "ffmpeg"], check=True)
        print_success("FFmpeg 安裝成功")
        return True
    except subprocess.CalledProcessError:
        print_warning("Homebrew 安裝失敗，請手動安裝 FFmpeg")
        print_info("下載地址: https://ffmpeg.org/download.html")
        return False


def install_ffmpeg_linux() -> bool:
    """Linux 安裝 FFmpeg"""
    try:
        print_info("使用 apt 安裝 FFmpeg...")
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y", "ffmpeg"], check=True)
        print_success("FFmpeg 安裝成功")
        return True
    except subprocess.CalledProcessError:
        print_warning("apt 安裝失敗，請手動安裝 FFmpeg")
        print_info("下載地址: https://ffmpeg.org/download.html")
        return False


def install_yt_dlp(venv_python: Path) -> bool:
    """安裝 yt-dlp"""
    print_header("安裝 yt-dlp")

    if shutil.which("yt-dlp"):
        print_success("yt-dlp 已安裝")
        return True

    try:
        print_info("安裝 yt-dlp...")
        subprocess.run([str(venv_python), "-m", "pip", "install", "yt-dlp"], check=True)
        print_success("yt-dlp 安裝成功")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"yt-dlp 安裝失敗: {e}")
        return False


def install_node_dependencies() -> bool:
    """安裝 Node.js 依賴（Vue 前端）"""
    print_header("安裝 Vue 前端依賴")

    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print_warning("frontend 目錄不存在，跳過前端依賴安裝")
        return True

    # 動態偵測 npm 路徑（在 Windows 下會是 npm.cmd）
    npm_path = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm_path:
        print_error("找不到 npm，請先安裝 Node.js 並加入系統 PATH")
        print_info("下載地址: https://nodejs.org/")
        return False

    # 顯示 node/npm 版本
    subprocess.run([npm_path, "--version"], check=True)
    subprocess.run(["node", "--version"], check=True)

    try:
        print_info(f"使用 npm 路徑: {npm_path}")
        print_info("安裝 Vue 前端依賴中...")
        subprocess.run([npm_path, "install"], cwd=str(frontend_dir), check=True)
        print_success("Vue 前端依賴安裝成功")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Vue 前端依賴安裝失敗: {e}")
        return False


# =============== Python 套件安裝 ===============


def upgrade_pip(venv_python: Path) -> bool:
    print_header("升級 pip")
    try:
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print_success("pip 升級成功")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"pip 升級失敗: {e}")
        return False


def install_requirements(venv_python: Path) -> bool:
    """安裝 requirements.txt 中的套件"""
    print_header("安裝 Python 套件")

    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print_error("找不到 requirements.txt 檔案")
        return False

    try:
        print_info("安裝依賴套件...")
        subprocess.run([str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print_success("Python 套件安裝成功")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Python 套件安裝失敗: {e}")
        return False

# =============== 環境檢查 ===============


def check_installed_packages(venv_python: Path) -> bool:
    """檢查已安裝的套件"""
    print_header("檢查已安裝套件")

    required_packages = [
        "torch", "torchvision", "torchaudio",
        "cv2", "librosa", "numpy",
        "faster-whisper", "sentence-transformers",
        "yt-dlp", "fastapi", "uvicorn"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            result = subprocess.run([
                str(venv_python), "-c", f"import {package.replace('-', '_')}"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print_success(f"{package} ✓")
            else:
                print_error(f"{package} ✗")
                missing_packages.append(package)
        except Exception:
            print_error(f"{package} ✗")
            missing_packages.append(package)

    if missing_packages:
        print_warning(f"缺少套件: {', '.join(missing_packages)}")
        return False

    print_success("所有必要套件已安裝")
    return True


def check_pytorch_cuda(venv_python: Path) -> bool:
    """檢查 PyTorch CUDA 支援"""
    print_header("檢查 PyTorch CUDA 支援")

    try:
        result = subprocess.run([
            str(venv_python), "-c", 
            "import torch; "
            "print(f'PyTorch 版本: {torch.__version__}'); "
            "print(f'CUDA 版本: {torch.version.cuda}'); "
            "print(f'CUDA 可用: {torch.cuda.is_available()}'); "
            "print(f'GPU 數量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); "
            "print(f'GPU 名稱: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print_success("PyTorch 檢查成功")
            print(result.stdout)
            return True
        else:
            print_error(f"PyTorch 檢查失敗: {result.stderr}")
            return False
            
    except Exception as e:
        print_error(f"PyTorch 檢查失敗: {e}")
        return False


def check_nvidia_driver() -> bool:
    """檢查 NVIDIA 驅動安裝"""
    print_header("檢查 NVIDIA 驅動安裝")

    # 檢查 NVIDIA 驅動
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print_success("NVIDIA 驅動已安裝")
            print_info(result.stdout.split('\n')[0])  # 顯示 GPU 資訊
        else:
            print_error("NVIDIA 驅動未安裝或不可用")
            return False
    except FileNotFoundError:
        print_error("nvidia-smi 不可用，請安裝 NVIDIA 驅動")
        return False

    return True


def check_external_tools() -> bool:
    """檢查外部工具"""
    print_header("檢查外部工具")

    tools_status = []

    # 檢查 FFmpeg
    if shutil.which("ffmpeg"):
        print_success("FFmpeg ✓")
        tools_status.append(True)
    else:
        print_error("FFmpeg ✗")
        tools_status.append(False)

    # 檢查 yt-dlp
    if shutil.which("yt-dlp"):
        print_success("yt-dlp ✓")
        tools_status.append(True)
    else:
        print_error("yt-dlp ✗")
        tools_status.append(False)

    if all(tools_status):
        print_success("所有外部工具已安裝")
        return True
    else:
        print_warning("部分外部工具未安裝")
        return False

# =============== 主函數 ===============


def main() -> None:
    print_header("Video Similarity Project - 自動依賴安裝")

    if not check_python_version():
        sys.exit(1)

    # 建立虛擬環境（這邊要用系統的 Python）
    if not create_virtual_environment(sys.executable):
        sys.exit(1)

    venv_python = get_venv_python()
    if not venv_python:
        print_error("找不到虛擬環境的 Python")
        sys.exit(1)

    print_info(f"使用虛擬環境: {venv_python}")

    if not check_pip(venv_python):
        sys.exit(1)

    if not upgrade_pip(venv_python):
        sys.exit(1)

    if not install_requirements(venv_python):
        sys.exit(1)

    install_ffmpeg()
    install_yt_dlp(venv_python)
    install_node_dependencies()

    print_header("安裝結果檢查")
    check_installed_packages(venv_python)
    check_external_tools()
    check_nvidia_driver()
    check_pytorch_cuda(venv_python)

    print_header("安裝完成")
    print_success("依賴安裝完成！")


if __name__ == "__main__":
    main()
