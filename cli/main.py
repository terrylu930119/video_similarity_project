import os
import sys
import json
import signal
import shutil
import argparse
import traceback
import multiprocessing
from pathlib import Path
from utils.logger import logger
from utils.downloader import download_video
from core.audio_processor import extract_audio
from typing import Union, Dict, List, Optional
from core.text_processor import transcribe_audio
from concurrent.futures import ThreadPoolExecutor
from utils.video_utils import extract_frames, get_video_info
from core.similarity import calculate_overall_similarity, display_similarity_results

BASE_DIR = Path(__file__).resolve().parents[1]  # 專案根目錄

DOWNLOADS_DIR = BASE_DIR / "downloads"
CACHE_DIR = BASE_DIR / "feature_cache"

os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ================ 全域變量與清理控制 ================
_cleanup_in_progress = False

# ================ 類別定義：影片處理器 ================


class VideoProcessor:
    """
    影片處理器：負責下載影片、提取音訊與幀，並將處理結果緩存以利重複使用。
    Attributes:
        output_dir (str): 下載與輸出檔案的根目錄。
        time_interval (Union[float, str]): 抽幀時間間隔，或設定為 "auto" 由系統自動決定。
        processed (Dict[str, dict]): 已處理影片的緩存結果。
    """

    def __init__(self, output_dir: str, time_interval: Union[float, str]) -> None:
        """初始化影片處理器實例。"""
        self.output_dir: str = output_dir
        self.time_interval: Union[float, str] = time_interval
        self.processed: Dict[str, dict] = {}

    def _decide_interval(self, duration: float) -> float:
        """
        根據影片長度自動決定抽幀時間間隔。
        Args:
            duration (float): 影片時長（秒）。
        Returns:
            float: 推薦的抽幀間隔（秒）。
        """
        if duration <= 120:
            return 0.5           # 短片：0.5 秒/幀
        elif duration <= 600:
            return 1.0           # 中短片：1.0 秒/幀
        elif duration <= 1200:
            return 2.0          # 中長片：2.0 秒/幀
        else:
            return 3.0          # 長片：3.0 秒/幀

    def download_and_process(self, link: str, preferred_lang: Optional[str] = None) -> dict:
        """
        下載並處理指定影片，若已處理則直接回傳緩存結果。
        Args:
            link (str): 影片 URL。
        Returns:
            dict: 處理結果，包含 video_path、audio_path、transcript、frames、duration。
        """
        if link in self.processed:
            logger.info(f"使用緩存的處理結果: {link}")
            return self.processed[link]

        logger.info(f"下載與處理影片: {link}")
        video_path = download_video(link, self.output_dir)
        data = self._process_video(video_path, link, preferred_lang)
        self.processed[link] = data

        return data

    def _process_video(self, video_path: str, video_url: str, preferred_lang: Optional[str] = None) -> dict:
        """
        處理本地影片檔案：
          1. 檢查影片檔案存在
          2. 擷取影片資訊（總幀、FPS）
          3. 提取音訊並轉錄
          4. 抽取影格並過濾有效檔案
        Args:
            video_path (str): 本地影片檔案路徑。
            video_url (str): 影片原始 URL（用於轉錄服務上下文）。
        Raises:
            FileNotFoundError: 影片或音訊檔案不存在。
            ValueError: 無法讀取影片資訊或幀。
        Returns:
            dict: 處理後資料，包括 paths、transcript 及抽幀清單等。
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"影片不存在: {video_path}")

        total_frames, _, _, fps = get_video_info(video_path)
        if total_frames == 0 or fps == 0:
            raise ValueError(f"無法獲取影片資訊: {video_path}")
        duration: float = total_frames / fps

        audio_path: str = extract_audio(video_path)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音訊不存在: {audio_path}")

        transcript, lang = "", None
        try:
            transcript, lang = transcribe_audio(audio_path, video_url, self.output_dir, preferred_lang)
        except Exception as e:
            logger.error(f"轉錄失敗: {e}")

        frames_dir: str = os.path.join(
            self.output_dir,
            "frames",
            os.path.basename(video_path).split('.')[0]
        )
        os.makedirs(frames_dir, exist_ok=True)

        interval: float = (
            self._decide_interval(duration)
            if self.time_interval == "auto"
            else float(self.time_interval)
        )
        logger.info(f"使用幀間隔：{interval:.2f} 秒")

        frames: List[str] = extract_frames(video_path, frames_dir, interval)
        valid_frames: List[str] = [os.path.abspath(f) for f in frames if os.path.exists(f)]
        if not valid_frames:
            raise FileNotFoundError("沒有有效的幀檔案")

        return {
            "video_path": video_path,
            "audio_path": audio_path,
            "transcript": transcript,
            "frames": valid_frames,
            "duration": duration,
            "lang": lang,
        }

# ================ 清理工具 ================


def cleanup_files(path: str) -> None:
    """
    刪除指定目錄及其所有子檔案。
    Args:
        path (str): 目標檔案或資料夾路徑。
    """
    if os.path.exists(path):
        try:
            logger.info(f"清理: {path}")
            shutil.rmtree(path)
        except PermissionError as e:
            logger.warning(f"無法刪除部分檔案（可能正在使用中）: {e}")
        except Exception as e:
            logger.error(f"清理失敗: {str(e)}")

# ================ 訊號處理器 ================


def signal_handler(signum: int, frame) -> None:
    """
    處理中斷訊號，先進行檔案清理再結束程式。
    避免重複清理。
    """
    global _cleanup_in_progress
    if _cleanup_in_progress:
        return
    logger.info("接收到中斷，開始清理...")
    _cleanup_in_progress = True
    cleanup_files("downloads")
    cleanup_files("feature_cache")
    _cleanup_in_progress = False
    sys.exit(1)


def setup_signal_handlers() -> None:
    """註冊 SIGINT 與 SIGTERM 的處理器。"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# ================ 主流程 ================


def main():
    """
    主流程：
      1. 解析 CLI 參數（參考影片、比對影片列表、輸出資料夾、快取、抽幀間隔、是否保留檔案）。
      2. 設置訊號處理、建立目錄、初始化 VideoProcessor。
      3. 下載並處理參考影片。
      4. 使用 ThreadPoolExecutor 併發下載、處理比對影片並計算相似度。
      5. 顯示比對結果，並依需求自動清理暫存檔案。
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="影片比對系統")
    parser.add_argument("--ref", required=True, help="參考影片連結")
    parser.add_argument("--comp", nargs='+', required=True, help="比對影片連結")
    parser.add_argument("--output", default=str(DOWNLOADS_DIR), help="輸出資料夾")
    parser.add_argument("--cache", default=str(CACHE_DIR), help="快取資料夾")
    parser.add_argument("--interval", default="auto", help="幀提取時間間隔（秒），可設為 auto 表示自動決定")
    parser.add_argument("--keep", action="store_true", help="是否保留中間檔案")

    # 如果指令沒有提供參數，則使用預設值
    if len(sys.argv) == 1:
        args = parser.parse_args([
            '--ref', '',
            '--comp', '',
            '--interval', 'auto',
            '--output', str(DOWNLOADS_DIR),
            '--cache', str(CACHE_DIR),
            '--keep'
        ])
    else:
        args = parser.parse_args()

    try:
        setup_signal_handlers()
        os.makedirs(os.path.join(args.output, "frames"), exist_ok=True)

        processor: VideoProcessor = VideoProcessor(args.output, args.interval)
        ref_data: dict = processor.download_and_process(args.ref)           # 處理參考影片
        ref_lang = ref_data["lang"]
        comparison_results: List[dict] = []                                 # 併發處理比對影片並計算相似度

        def process_and_compare(link: str) -> None:
            """內部函式：下載、處理並計算與參考影片的相似度"""
            try:
                comp_data: dict = processor.download_and_process(link, preferred_lang=ref_lang)
                required_paths: List[str] = (
                    [ref_data["audio_path"], comp_data["audio_path"]]
                    + ref_data["frames"] + comp_data["frames"]
                )
                if not all(os.path.exists(p) for p in required_paths):
                    raise FileNotFoundError("部分文件不存在")

                result: dict = calculate_overall_similarity(
                    ref_data["audio_path"], comp_data["audio_path"],
                    ref_data["frames"], comp_data["frames"],
                    ref_data["transcript"], comp_data["transcript"],
                    max(ref_data["duration"], comp_data["duration"])
                )
                result["link"] = link
                comparison_results.append(result)

            except Exception as e:
                logger.error(f"比對失敗 {link}: {str(e)}")

        max_workers: int = max(2, min(4, multiprocessing.cpu_count() - 2))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(process_and_compare, args.comp)

        display_similarity_results(args.ref, comparison_results)
        print(json.dumps(comparison_results, ensure_ascii=False))

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
