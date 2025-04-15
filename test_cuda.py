import torch
import sys

def test_cuda():
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 是否可用:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("CUDA 版本:", torch.version.cuda)
        print("當前 CUDA 設備:", torch.cuda.current_device())
        print("CUDA 設備數量:", torch.cuda.device_count())
        print("CUDA 設備名稱:", torch.cuda.get_device_name(0))
        
        # 測試 CUDA 張量運算
        x = torch.rand(5, 3)
        print("\nCPU 張量:")
        print(x)
        print("CPU 張量設備:", x.device)
        
        x = x.cuda()
        print("\nGPU 張量:")
        print(x)
        print("GPU 張量設備:", x.device)
        
        # 測試簡單運算
        y = x + x
        print("\nGPU 運算結果:")
        print(y)
    else:
        print("\n警告: CUDA 不可用，請檢查:")
        print("1. 是否安裝了 NVIDIA 顯卡驅動")
        print("2. 是否安裝了 CUDA Toolkit")
        print("3. 是否安裝了支援 CUDA 的 PyTorch 版本")

if __name__ == "__main__":
    test_cuda() 