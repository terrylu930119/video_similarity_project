import os
import argparse
import sys
import traceback
import signal
import shutil
from utils.downloader import download_video
from core.audio_processor import extract_audio
from core.text_processor import transcribe_audio
from core.similarity import calculate_overall_similarity, display_similarity_results
from utils.video_utils import extract_frames, get_video_info
from utils.logger import logger
from utils.gpu_utils import gpu_manager
import torch

# 全局變量用於追踪清理狀態
_cleanup_in_progress = False

def signal_handler(signum, frame):
    """處理中斷信號"""
    global _cleanup_in_progress
    if _cleanup_in_progress:
        logger.info("清理正在進行中，請稍候...")
        return
        
    logger.info("接收到中斷信號，開始清理...")
    _cleanup_in_progress = True
    try:
        cleanup_files("downloads")
    finally:
        _cleanup_in_progress = False
    sys.exit(1)

def setup_signal_handlers():
    """設置信號處理器"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def check_file_exists(file_path: str) -> bool:
    """檢查檔案是否存在"""
    if not os.path.exists(file_path):
        logger.error(f"檔案不存在: {file_path}")
        return False
    return True

def check_files_exist(file_paths: list) -> bool:
    """檢查多個檔案是否存在"""
    for file_path in file_paths:
        if not check_file_exists(file_path):
            return False
    return True

def process_video(video_path: str, output_dir: str, time_interval: float, video_url: str = None, use_silence_detection: bool = True, use_source_separation: bool = True) -> tuple:
    """處理單個影片，返回音訊路徑、轉錄文本、幀列表和視頻時長"""
    if not check_file_exists(video_path):
        raise FileNotFoundError(f"影片檔案不存在: {video_path}")
    
    # 獲取視頻時長
    total_frames, width, height, fps = get_video_info(video_path)
    if total_frames == 0 or fps == 0:
        raise ValueError(f"無法獲取視頻信息: {video_path}")
    video_duration = total_frames / fps
    
    # 提取音訊
    audio_path = extract_audio(video_path)
    if not check_file_exists(audio_path):
        raise FileNotFoundError(f"音訊檔案不存在: {audio_path}")
    
    # 轉錄音訊（如果有提供影片 URL，會先嘗試提取字幕）
    # 注意：transcribe_audio 函數會自動進行人聲分離
    transcript = transcribe_audio(
        audio_path, 
        video_url, 
        output_dir, 
        use_silence_detection=use_silence_detection,
        use_source_separation=use_source_separation
    )
    if not transcript:
        logger.warning(f"無法獲取轉錄文本，將使用空文本: {video_path}")
        transcript = ""
    
    # 提取幀
    try:
        frames_dir = os.path.join(output_dir, "frames", os.path.basename(video_path).split('.')[0])
        os.makedirs(frames_dir, exist_ok=True)
        frames = extract_frames(video_path, frames_dir, time_interval)
        
        if not frames:
            raise FileNotFoundError(f"提取幀失敗: {video_path}")
        
        # 檢查幀檔案是否存在並轉換為絕對路徑
        valid_frames = []
        for frame in frames:
            if os.path.exists(frame):
                valid_frames.append(os.path.abspath(frame))
            else:
                logger.warning(f"幀檔案不存在: {frame}")
        
        if not valid_frames:
            raise FileNotFoundError(f"沒有有效的幀檔案")
            
        return audio_path, transcript, valid_frames, video_duration
        
    except Exception as e:
        logger.error(f"處理視頻時出錯 {video_path}: {str(e)}")
        raise

def cleanup_files(output_dir: str):
    """清理下載的文件和臨時文件"""
    try:
        # 檢查目錄是否存在
        if not os.path.exists(output_dir):
            return
            
        logger.info(f"開始清理目錄: {output_dir}")
        
        # 刪除整個目錄及其內容
        shutil.rmtree(output_dir)
        logger.info(f"已清理目錄: {output_dir}")
        
    except Exception as e:
        logger.error(f"清理文件時出錯: {str(e)}")

def main():
    try:
        # 設置信號處理器
        setup_signal_handlers()
        
        # 檢查 GPU 可用性
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            logger.info("GPU 加速可用，將使用 GPU 進行處理")
        else:
            logger.warning("GPU 加速不可用，將使用 CPU 進行處理")
        
        # 設定參數
        reference_link = "https://www.youtube.com/watch?v=ebcpK31YY3g"
        comparison_links = [ "https://www.youtube.com/watch?v=-aYBLF9OAaI"
        ]
        time_interval = 1.0  # 每2秒提取一幀
        resolution = "720p"
        output_dir = "downloads"
        
        try:
            # 創建輸出目錄
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
            
            # 用於緩存已處理的視頻
            processed_videos = {}
            
            # 處理參考視頻
            logger.info(f"下載參考視頻: {reference_link}")
            # 使用通用的下載函數，支援各種網站
            ref_video_path = download_video(reference_link, output_dir, resolution)
            ref_audio_path, ref_transcript, ref_frames, ref_duration = process_video(
                ref_video_path, 
                output_dir, 
                time_interval,
                reference_link  # 傳遞影片 URL
            )
            
            # 緩存參考視頻的處理結果
            processed_videos[reference_link] = {
                "video_path": ref_video_path,
                "audio_path": ref_audio_path,
                "transcript": ref_transcript,
                "frames": ref_frames,
                "duration": ref_duration
            }
            
            # 存儲所有比對結果
            comparison_results = []
            
            # 處理比對視頻
            for link in comparison_links:
                try:
                    logger.info(f"處理比對視頻: {link}")
                    
                    # 檢查是否已經處理過這個視頻
                    if link in processed_videos:
                        logger.info(f"使用緩存的處理結果: {link}")
                        comp_video_path = processed_videos[link]["video_path"]
                        comp_audio_path = processed_videos[link]["audio_path"]
                        comp_transcript = processed_videos[link]["transcript"]
                        comp_frames = processed_videos[link]["frames"]
                        comp_duration = processed_videos[link]["duration"]
                    else:
                        # 下載並處理新視頻，使用通用的下載函數
                        comp_video_path = download_video(link, output_dir, resolution)
                        comp_audio_path, comp_transcript, comp_frames, comp_duration = process_video(
                            comp_video_path, 
                            output_dir, 
                            time_interval,
                            link  # 傳遞影片 URL
                        )
                        
                        # 緩存處理結果
                        processed_videos[link] = {
                            "video_path": comp_video_path,
                            "audio_path": comp_audio_path,
                            "transcript": comp_transcript,
                            "frames": comp_frames,
                            "duration": comp_duration
                        }
                    
                    # 檢查所有必要的文件是否存在
                    if not all(check_file_exists(path) for path in [ref_audio_path, comp_audio_path] + ref_frames + comp_frames):
                        raise FileNotFoundError("部分必要文件不存在")
                    
                    # 使用較長的視頻時長作為比較基準
                    video_duration = max(ref_duration, comp_duration)
                    
                    # 計算相似度
                    similarity_result = calculate_overall_similarity(
                        ref_audio_path,
                        comp_audio_path,
                        ref_frames,
                        comp_frames,
                        ref_transcript,
                        comp_transcript,
                        video_duration
                    )
                    
                    # 添加視頻連結到結果中
                    similarity_result["link"] = link
                    comparison_results.append(similarity_result)
                    
                except Exception as e:
                    logger.error(f"處理比對視頻 {link} 時出錯: {str(e)}")
                    print(f"錯誤: 無法處理比對視頻 {link}: {str(e)}")
                    continue
            
            # 顯示所有比對結果
            display_similarity_results(reference_link, comparison_results)
            
            # 詢問使用者是否要刪除檔案
            while True:
                choice = input("\n是否要刪除生成的檔案？(y/n): ").lower()
                if choice in ['y', 'n']:
                    break
                print("請輸入 y 或 n")
            
            if choice == 'y':
                logger.info("使用者選擇刪除檔案")
                cleanup_files(output_dir)
            else:
                logger.info("使用者選擇保留檔案")
                
        except KeyboardInterrupt:
            logger.info("程式被使用者中斷")
            # 確保在中斷時也清理文件
            cleanup_files(output_dir)
            sys.exit(1)
        except Exception as e:
            logger.error(f"程式執行出錯: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"錯誤: {str(e)}")
            print("詳細錯誤信息已記錄到日誌文件中")
            
            # 確保在出錯時也清理文件
            cleanup_files(output_dir)
            sys.exit(1)
    except Exception as e:
        logger.error(f"程式發生未預期的錯誤: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
