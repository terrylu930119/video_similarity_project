# cli/main.py
"""
影片比對系統 CLI 主程式

此模組提供影片比對系統的命令列介面，包括：
- 影片下載與處理
- 多執行緒比對處理
- 進度追蹤與事件發送
- 檔案清理與信號處理
"""

import os
import sys
import json
import signal
import shutil
import argparse
import traceback
import threading
import multiprocessing
from pathlib import Path
from utils.logger import logger, emit
from utils.downloader import download_video
from core.audio_processor import extract_audio
from typing import Union, Dict, List, Optional, Tuple
from core.text_processor import transcribe_audio
from concurrent.futures import ThreadPoolExecutor
from utils.video_utils import extract_frames, get_video_info
from core.similarity import calculate_overall_similarity, display_similarity_results

# ======================== 全域變數 ========================
BASE_DIR = Path(__file__).resolve().parents[1]
DOWNLOADS_DIR = BASE_DIR / "downloads"
CACHE_DIR = BASE_DIR / "feature_cache"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

_cleanup_in_progress = False

# ======================== 工具函式 ========================


def sha10(s: str) -> str:
    """計算字串的 SHA1 雜湊值前 10 位"""
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


# ======================== 影片處理器類別 ========================
class VideoProcessor:
    """影片下載與處理器，負責影片的下載、音訊提取、文本轉錄和幀提取"""

    def __init__(self, output_dir: str, time_interval: Union[float, str]) -> None:
        """
        初始化影片處理器

        Args:
            output_dir: 輸出目錄
            time_interval: 幀提取時間間隔
        """
        self.output_dir: str = output_dir
        self.time_interval: Union[float, str] = time_interval
        self.processed: Dict[str, dict] = {}

    def _decide_interval(self, duration: float) -> float:
        """
        根據影片長度自動決定幀提取間隔

        Args:
            duration: 影片長度（秒）

        Returns:
            float: 建議的幀提取間隔
        """
        if duration <= 120:
            return 0.5
        elif duration <= 600:
            return 1.0
        elif duration <= 1200:
            return 2.0
        else:
            return 3.0

    def download_and_process(self, link: str, task_id: str, is_ref: bool = False,
                             preferred_lang: Optional[str] = None) -> dict:
        """
        下載並處理影片

        Args:
            link: 影片連結
            task_id: 任務 ID
            is_ref: 是否為參考影片
            preferred_lang: 偏好語言

        Returns:
            dict: 處理結果
        """

        # 開始準備影片
        logger.info(f"開始準備影片: {link}")
        emit("progress", task_id=task_id, phase="download", percent=1, msg="開始準備影片")

        # 檢查是否已經處理過
        if link in self.processed:
            logger.info(f"使用緩存的處理結果: {link}")
            emit("progress", task_id=task_id, phase="download", percent=10, msg="影片已存在（快取）")
            return self.processed[link]

        # 檢查是否存在本地檔案中，如果不存在就下載
        video_path = download_video(link, self.output_dir, task_id=task_id)
        emit("progress", task_id=task_id, phase="download", percent=10, msg="影片就緒")

        data = self._process_video(video_path, link, task_id, preferred_lang)
        self.processed[link] = data
        return data

    def _extract_audio_and_transcript(self, video_path: str, video_url: str, task_id: str,
                                      preferred_lang: Optional[str] = None) -> Tuple[str, str, Optional[str]]:
        """
        提取音訊並轉錄文本

        Args:
            video_path: 影片路徑
            video_url: 影片 URL
            task_id: 任務 ID
            preferred_lang: 偏好語言

        Returns:
            Tuple[str, str, Optional[str]]: (音訊路徑, 轉錄文本, 語言)
        """
        emit("progress", task_id=task_id, phase="audio", percent=12, msg="開始提取音訊")
        audio_path: str = extract_audio(video_path)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音訊不存在: {audio_path}")
        emit("progress", task_id=task_id, phase="audio", percent=18, msg="音訊就緒")

        transcript, lang = "", None
        try:
            emit("progress", task_id=task_id, phase="文本處理", percent=20, msg="開始處裡文本內容")
            transcript, lang = transcribe_audio(audio_path, video_url, self.output_dir, preferred_lang, task_id=task_id)
        except Exception as e:
            logger.error(f"轉錄失敗: {e}")
            emit("log", task_id=task_id, msg=f"轉錄失敗: {e}")

        return audio_path, transcript, lang

    def _extract_frames(self, video_path: str, task_id: str, duration: float) -> List[str]:
        """
        提取影片幀

        Args:
            video_path: 影片路徑
            task_id: 任務 ID
            duration: 影片長度

        Returns:
            List[str]: 幀檔案路徑列表
        """
        frames_dir: str = os.path.join(self.output_dir, "frames", os.path.basename(video_path).split('.')[0])
        os.makedirs(frames_dir, exist_ok=True)

        interval: float = (self._decide_interval(duration) if self.time_interval == "auto" else float(self.time_interval))
        logger.info(f"使用幀間隔：{interval:.2f} 秒")
        emit("log", task_id=task_id, msg=f"使用幀間隔：{interval:.2f} 秒")

        frames: List[str] = extract_frames(video_path, frames_dir, interval)
        valid_frames: List[str] = [os.path.abspath(f) for f in frames if os.path.exists(f)]
        if not valid_frames:
            raise FileNotFoundError("沒有有效的幀檔案")

        emit("progress", task_id=task_id, phase="extract", percent=60, msg=f"抽幀完成")
        return valid_frames

    def _process_video(self, video_path: str, video_url: str, task_id: str,
                       preferred_lang: Optional[str] = None) -> dict:
        """
        處理影片：提取音訊、轉錄文本、提取幀

        Args:
            video_path: 影片路徑
            video_url: 影片 URL
            task_id: 任務 ID
            preferred_lang: 偏好語言

        Returns:
            dict: 處理結果
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"影片不存在: {video_path}")

        # 獲取影片資訊
        total_frames, _, _, fps = get_video_info(video_path)
        if total_frames == 0 or fps == 0:
            raise ValueError(f"無法獲取影片資訊: {video_path}")
        duration: float = total_frames / fps

        # 提取音訊和轉錄文本
        audio_path, transcript, lang = self._extract_audio_and_transcript(video_path, video_url, task_id, preferred_lang)

        # 提取幀
        valid_frames = self._extract_frames(video_path, task_id, duration)

        return {
            "video_path": video_path,
            "audio_path": audio_path,
            "transcript": transcript,
            "frames": valid_frames,
            "duration": duration,
            "lang": lang,
        }


# ======================== 檔案清理與信號處理 ========================
def cleanup_files(path: str) -> None:
    """
    清理指定路徑的檔案

    Args:
        path: 要清理的路徑
    """
    if os.path.exists(path):
        try:
            logger.info(f"清理: {path}")
            shutil.rmtree(path)
        except PermissionError as e:
            logger.warning(f"無法刪除部分檔案（可能正在使用中）: {e}")
        except Exception as e:
            logger.error(f"清理失敗: {str(e)}")


def signal_handler(signum: int, frame) -> None:
    """
    信號處理器，處理中斷信號

    Args:
        signum: 信號編號
        frame: 當前幀
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
    """設定信號處理器"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# ======================== 進度追蹤與比對處理 ========================
class ProgressTracker:
    """進度追蹤器，管理整體比對進度"""

    def __init__(self, ref_task_id: str, total_targets: int):
        """
        初始化進度追蹤器

        Args:
            ref_task_id: 參考影片任務 ID
            total_targets: 總目標數量
        """
        self.ref_task_id = ref_task_id
        self.total_targets = max(1, total_targets)
        self.done_lock = threading.Lock()
        self.done_count = {"n": 0}

    def bump_ref_progress(self, note: str = "") -> None:
        """
        更新參考影片進度

        Args:
            note: 進度說明
        """
        with self.done_lock:
            n = self.done_count["n"]
            # 讓整體比對進度在 60% → 100% 之間線性推進
            pct = 60 + int((n / self.total_targets) * 40)
            msg = f"比對進度：{n}/{self.total_targets}"
            if note:
                msg += f"({note})"
            emit("progress", task_id=self.ref_task_id, phase="compare", percent=min(pct, 99), msg=msg)

    def increment_done_count(self, note: str = "") -> None:
        """
        增加完成計數

        Args:
            note: 完成說明
        """
        with self.done_lock:
            self.done_count["n"] += 1
            n = self.done_count["n"]
        self.bump_ref_progress(note=note)

        # 全部完成 → 把 ref 補到 100%
        if n >= self.total_targets:
            emit("progress", task_id=self.ref_task_id, phase="compare", percent=100, msg="全部比對完成")


def _process_single_comparison(link: str, task_id: str, ref_data: dict,
                               processor: VideoProcessor, progress_tracker: ProgressTracker) -> dict:
    """
    處理單個比對任務

    Args:
        link: 目標影片連結
        task_id: 任務 ID
        ref_data: 參考影片資料
        processor: 影片處理器
        progress_tracker: 進度追蹤器

    Returns:
        dict: 比對結果
    """
    try:
        # 在 ref 卡打一行目前開始比對哪支
        emit("log", task_id=progress_tracker.ref_task_id, msg=f"開始與目標影片比對：{link}")
        emit("progress", task_id=task_id, phase="queued", percent=1, msg="佇列中")

        # 下載並處理目標影片
        comp_data: dict = processor.download_and_process(link, task_id=task_id, preferred_lang=ref_data["lang"])

        # 檢查必要檔案
        required_paths: List[str] = [ref_data["audio_path"], comp_data["audio_path"]] + ref_data["frames"] + comp_data["frames"]
        if not all(os.path.exists(p) for p in required_paths):
            raise FileNotFoundError("部分文件不存在")

        # 計算相似度
        result: dict = calculate_overall_similarity(
            ref_data["audio_path"], comp_data["audio_path"],
            ref_data["frames"], comp_data["frames"],
            ref_data["transcript"], comp_data["transcript"],
            max(ref_data["duration"], comp_data["duration"]),
            emit_cb=emit, task_id=task_id, link=link
        )
        result["link"] = link

        # 發送完成事件
        emit("done", task_id=task_id, link=link,
             audio=result.get("audio_similarity", 0),
             visual=result.get("image_similarity", 0),
             text=result.get("text_similarity", 0),
             score=result.get("overall_similarity", 0),
             text_meaningful=result.get("text_meaningful"),
             text_status=result.get("text_status"))
        emit("progress", task_id=task_id, phase="compare", percent=100, msg="完成")

        # 更新進度
        progress_tracker.increment_done_count(note=f"{link} 完成")
        return result

    except Exception as e:
        emit("error", task_id=task_id, msg=str(e), link=link)
        # 失敗也算進度，避免卡住
        progress_tracker.increment_done_count(note=f"{link} 失敗")
        raise


def _setup_default_args() -> argparse.Namespace:
    """
    設定預設參數（當沒有提供命令列參數時）

    Returns:
        argparse.Namespace: 預設參數
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="影片比對系統")
    parser.add_argument("--ref", required=True, help="參考影片連結")
    parser.add_argument("--comp", nargs='+', required=True, help="比對影片連結")
    parser.add_argument("--output", default=str(DOWNLOADS_DIR), help="輸出資料夾")
    parser.add_argument("--cache", default=str(CACHE_DIR), help="快取資料夾")
    parser.add_argument("--interval", default="auto", help="幀提取時間間隔（秒），可設為 auto 表示自動決定")
    parser.add_argument("--keep", action="store_true", help="是否保留中間檔案")

    return parser.parse_args([
        '--ref', '',
        '--comp', '',
        '--interval', 'auto',
        '--output', str(DOWNLOADS_DIR),
        '--cache', str(CACHE_DIR),
        '--keep'
    ])


def _initialize_environment(args: argparse.Namespace) -> None:
    """
    初始化環境

    Args:
        args: 命令列參數
    """
    setup_signal_handlers()
    os.makedirs(os.path.join(args.output, "frames"), exist_ok=True)


def _create_task_ids(ref_url: str, comp_urls: List[str]) -> Tuple[str, List[Dict[str, str]]]:
    """
    建立任務 ID

    Args:
        ref_url: 參考影片 URL
        comp_urls: 比對影片 URL 列表

    Returns:
        Tuple[str, List[Dict[str, str]]]: (參考任務 ID, 目標任務列表)
    """
    ref_task_id = f"ref-{sha10(ref_url)}"
    targets: List[Dict[str, str]] = [{"url": link, "task_id": sha10(link)} for link in comp_urls]
    return ref_task_id, targets


def _announce_tasks(ref_task_id: str, ref_url: str, targets: List[Dict[str, str]]) -> None:
    """
    公告任務（方便前端立刻建卡）

    Args:
        ref_task_id: 參考任務 ID
        ref_url: 參考影片 URL
        targets: 目標任務列表
    """
    emit("hello", ref={"task_id": ref_task_id, "url": ref_url}, targets=targets)


def _process_reference_video(args: argparse.Namespace, ref_task_id: str,
                             processor: VideoProcessor) -> dict:
    """
    處理參考影片

    Args:
        args: 命令列參數
        ref_task_id: 參考任務 ID
        processor: 影片處理器

    Returns:
        dict: 參考影片資料
    """
    emit("progress", task_id=ref_task_id, phase="queued", percent=1, msg="佇列中")
    ref_data: dict = processor.download_and_process(args.ref, task_id=ref_task_id, is_ref=True)

    # 不要直接 100%，先標示「準備開始比對」
    emit("progress", task_id=ref_task_id, phase="compare", percent=60,
         msg=f"參考影片處理完成，開始比對(0/{len(args.comp)})")

    return ref_data


def _execute_comparisons(targets: List[Dict[str, str]], ref_data: dict,
                         processor: VideoProcessor, progress_tracker: ProgressTracker) -> List[dict]:
    """
    執行所有比對任務

    Args:
        targets: 目標任務列表
        ref_data: 參考影片資料
        processor: 影片處理器
        progress_tracker: 進度追蹤器

    Returns:
        List[dict]: 比對結果列表
    """
    comparison_results: List[dict] = []
    max_workers: int = max(2, min(4, multiprocessing.cpu_count() - 2))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有比對任務
        future_to_target = {
            executor.submit(_process_single_comparison, tgt["url"], tgt["task_id"],
                            ref_data, processor, progress_tracker): tgt
            for tgt in targets
        }

        # 收集結果
        for future in future_to_target:
            try:
                result = future.result()
                comparison_results.append(result)
            except Exception as e:
                logger.error(f"比對任務失敗: {e}")

    return comparison_results


def _finalize_and_cleanup(args: argparse.Namespace, comparison_results: List[dict],
                          ref_url: str) -> None:
    """
    完成處理並清理

    Args:
        args: 命令列參數
        comparison_results: 比對結果列表
        ref_url: 參考影片 URL
    """
    # 顯示結果
    display_similarity_results(ref_url, comparison_results)
    print(json.dumps(comparison_results, ensure_ascii=False))

    # 清理檔案
    if not args.keep:
        logger.info("自動清理中...")
        cleanup_files(args.output)
        cleanup_files(args.cache)


# ======================== 主程式 ========================
def main():
    """主程式入口點"""
    try:
        # 解析命令列參數
        if len(sys.argv) == 1:
            args = _setup_default_args()
        else:
            parser: argparse.ArgumentParser = argparse.ArgumentParser(description="影片比對系統")
            parser.add_argument("--ref", required=True, help="參考影片連結")
            parser.add_argument("--comp", nargs='+', required=True, help="比對影片連結")
            parser.add_argument("--output", default=str(DOWNLOADS_DIR), help="輸出資料夾")
            parser.add_argument("--cache", default=str(CACHE_DIR), help="快取資料夾")
            parser.add_argument("--interval", default="auto", help="幀提取時間間隔（秒），可設為 auto 表示自動決定")
            parser.add_argument("--keep", action="store_true", help="是否保留中間檔案")
            args = parser.parse_args()

        # 初始化環境
        _initialize_environment(args)

        # 建立任務 ID
        ref_task_id, targets = _create_task_ids(args.ref, args.comp)

        # 公告任務
        _announce_tasks(ref_task_id, args.ref, targets)

        # 建立影片處理器
        processor: VideoProcessor = VideoProcessor(args.output, args.interval)

        # 處理參考影片
        ref_data: dict = _process_reference_video(args, ref_task_id, processor)
        ref_lang = ref_data["lang"]

        # 建立進度追蹤器
        progress_tracker = ProgressTracker(ref_task_id, len(targets))

        # 執行比對
        comparison_results = _execute_comparisons(targets, ref_data, processor, progress_tracker)

        # 完成處理並清理
        _finalize_and_cleanup(args, comparison_results, args.ref)

    except Exception as e:
        logger.error(f"主程式錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        cleanup_files(args.output if 'args' in locals() else str(DOWNLOADS_DIR))
        cleanup_files(args.cache if 'args' in locals() else str(CACHE_DIR))
        sys.exit(1)


if __name__ == "__main__":
    main()
