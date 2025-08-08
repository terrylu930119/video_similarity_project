import subprocess
import webbrowser
import sys
import os
import time
from pathlib import Path

# ========= 設定 =========
PROJECT_DIR = Path(__file__).parent.resolve()
BACKEND_ENTRY = "backend.main:app"
FRONTEND_DIR = PROJECT_DIR / "frontend"
VENV_PYTHON = PROJECT_DIR / ".venv/Scripts/python.exe"
NPM_PATH = "C:/Program Files/nodejs/npm.cmd"
VITE_URL = "http://localhost:5173"

# ========= 檢查 =========
if not VENV_PYTHON.exists():
    print(f"找不到虛擬環境 Python：{VENV_PYTHON}")
    sys.exit(1)

if not (PROJECT_DIR / "backend/main.py").exists():
    print("找不到 backend/main.py")
    sys.exit(1)

if not FRONTEND_DIR.exists():
    print("找不到 frontend 資料夾")
    sys.exit(1)

# ========= 命令建構 =========
backend_cmd = f'"{VENV_PYTHON}" -m uvicorn {BACKEND_ENTRY} --reload --host 127.0.0.1 --port 8000'
frontend_cmd = f'"{NPM_PATH}" run dev'

# ========= 在新終端機視窗中啟動 =========
print("[🚀] 開啟後端終端機...")
subprocess.Popen(f'start "後端伺服器" cmd /k {backend_cmd}', shell=True)

print("[🌱] 開啟前端終端機...")
subprocess.Popen(f'start "前端介面" cmd /k {frontend_cmd}', cwd=str(FRONTEND_DIR), shell=True)

# ========= 開瀏覽器 =========
print("[🌐] 等待前端啟動中...")
time.sleep(3)
webbrowser.open(VITE_URL)

print("[✅] 所有服務已啟動完畢，可在終端機視窗中查看 log！")
