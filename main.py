import os
import sys
import torch
import signal
import shutil
import traceback
import argparse
from utils.logger import logger
from utils.downloader import download_video
from core.audio_processor import extract_audio
from core.text_processor import transcribe_audio
from concurrent.futures import ThreadPoolExecutor
from utils.video_utils import extract_frames, get_video_info
from core.similarity import calculate_overall_similarity, display_similarity_results

# =============== 全局變量與清理控制 ===============
_cleanup_in_progress = False

# =============== 類別定義：影片處理器 ===============
class VideoProcessor:
    def __init__(self, output_dir: str, time_interval: float):
        self.output_dir = output_dir
        self.time_interval = time_interval
        self.processed = {}

    def download_and_process(self, link: str) -> dict:
        if link in self.processed:
            logger.info(f"使用緩存的處理結果: {link}")
            return self.processed[link]

        logger.info(f"下載與處理影片: {link}")
        video_path = download_video(link, self.output_dir)
        data = self._process_video(video_path, link)
        self.processed[link] = data
        return data

    def _process_video(self, video_path: str, video_url: str) -> dict:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"影片不存在: {video_path}")

        total_frames, _, _, fps = get_video_info(video_path)
        if total_frames == 0 or fps == 0:
            raise ValueError(f"無法獲取影片資訊: {video_path}")
        duration = total_frames / fps

        audio_path = extract_audio(video_path)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音訊不存在: {audio_path}")

        transcript = transcribe_audio(audio_path, video_url, self.output_dir) or ""

        frames_dir = os.path.join(self.output_dir, "frames", os.path.basename(video_path).split('.')[0])
        os.makedirs(frames_dir, exist_ok=True)
        frames = extract_frames(video_path, frames_dir, self.time_interval)

        valid_frames = [os.path.abspath(f) for f in frames if os.path.exists(f)]
        if not valid_frames:
            raise FileNotFoundError("沒有有效的幀檔案")

        return {
            "video_path": video_path,
            "audio_path": audio_path,
            "transcript": transcript,
            "frames": valid_frames,
            "duration": duration
        }

# =============== 清理工具 ===============
def cleanup_files(path: str):
    if os.path.exists(path):
        try:
            logger.info(f"清理: {path}")
            shutil.rmtree(path)
        except PermissionError as e:
            logger.warning(f"⚠️ 無法刪除部分檔案（可能正在使用中）: {e}")
        except Exception as e:
            logger.error(f"清理失敗: {str(e)}")

# =============== 訊號處理器 ===============
def signal_handler(signum, frame):
    global _cleanup_in_progress
    if _cleanup_in_progress:
        return
    logger.info("接收到中斷，開始清理...")
    _cleanup_in_progress = True
    cleanup_files("downloads")
    cleanup_files("feature_cache")
    _cleanup_in_progress = False
    sys.exit(1)

def setup_signal_handlers():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# =============== 主流程 ===============
def main():
    parser = argparse.ArgumentParser(description="影片比對系統")
    parser.add_argument("--ref", required=True, help="參考影片連結")
    parser.add_argument("--comp", nargs='+', required=True, help="比對影片連結")
    parser.add_argument("--output", default="downloads", help="輸出資料夾")
    parser.add_argument("--cache", default="feature_cache", help="快取資料夾")  
    parser.add_argument("--interval", type=float, default=2.0, help="幀提取時間間隔")
    parser.add_argument("--keep", action="store_true", help="是否保留中間檔案")
    
    if len(sys.argv) == 1:
        # 預設測試參數（無 CLI 時）
        args = parser.parse_args([
            '--ref', '',
            '--comp', '',
            '--interval', '2.0',
            '--output', 'downloads',
            '--cache', 'feature_cache',
            '--keep'
        ])
    else:
        args = parser.parse_args()

    try:
        setup_signal_handlers()

        logger.info("GPU 可用" if torch.cuda.is_available() else "使用 CPU")
        os.makedirs(os.path.join(args.output, "frames"), exist_ok=True)
        processor = VideoProcessor(args.output, args.interval)

        # 處理參考影片
        ref_data = processor.download_and_process(args.ref)

        # 處理比對影片（併發加速）
        comparison_results = []
        def process_and_compare(link):
            try:
                comp_data = processor.download_and_process(link)
                if not all(os.path.exists(p) for p in [ref_data["audio_path"], comp_data["audio_path"]] + ref_data["frames"] + comp_data["frames"]):
                    raise FileNotFoundError("部分文件不存在")
                duration = max(ref_data["duration"], comp_data["duration"])
                result = calculate_overall_similarity(
                    ref_data["audio_path"], comp_data["audio_path"],
                    ref_data["frames"], comp_data["frames"],
                    ref_data["transcript"], comp_data["transcript"],
                    duration
                )
                result["link"] = link
                comparison_results.append(result)
            except Exception as e:
                logger.error(f"比對失敗 {link}: {str(e)}")

        with ThreadPoolExecutor() as executor:   #max_workers=6
            executor.map(process_and_compare, args.comp)

        display_similarity_results(args.ref, comparison_results)

        if not args.keep:
            logger.info("自動清理中...")
            cleanup_files(args.output)
            cleanup_files(args.cache)

    except Exception as e:
        logger.error(f"主程式錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        cleanup_files(args.output)
        cleanup_files(args.cache)
        sys.exit(1)

if __name__ == "__main__":
    main()
