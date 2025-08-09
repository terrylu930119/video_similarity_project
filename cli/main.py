
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
from utils.logger import logger
from utils.downloader import download_video
from core.audio_processor import extract_audio
from typing import Union, Dict, List, Optional, Callable
from core.text_processor import transcribe_audio
from concurrent.futures import ThreadPoolExecutor
from utils.video_utils import extract_frames, get_video_info
from core.similarity import calculate_overall_similarity, display_similarity_results

BASE_DIR = Path(__file__).resolve().parents[1]
DOWNLOADS_DIR = BASE_DIR / "downloads"
CACHE_DIR = BASE_DIR / "feature_cache"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

_cleanup_in_progress = False

# ---------- Helpers ----------


def sha10(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def emit(event_type: str, **kw):
    """NDJSON event (ALSO flush immediately), safe fallback if non-serializable."""
    obj = {"type": event_type, **kw}
    try:
        print(json.dumps(obj, ensure_ascii=False), flush=True)
    except Exception as e:
        # last resort: stringify
        obj["__err"] = f"emit-serde:{e}"
        obj["payload"] = {k: str(v) for k, v in kw.items()}
        print(json.dumps(obj, ensure_ascii=False), flush=True)

# ---------- Video Processor ----------


class VideoProcessor:
    def __init__(self, output_dir: str, time_interval: Union[float, str]) -> None:
        self.output_dir: str = output_dir
        self.time_interval: Union[float, str] = time_interval
        self.processed: Dict[str, dict] = {}

    def _decide_interval(self, duration: float) -> float:
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
        if link in self.processed:
            logger.info(f"使用緩存的處理結果: {link}")
            emit("progress", task_id=task_id, phase="download", percent=10, msg="影片已存在（快取）")
            return self.processed[link]

        emit("progress", task_id=task_id, phase="download", percent=0, msg="開始下載/載入影片")
        logger.info(f"下載與處理影片: {link}")
        video_path = download_video(link, self.output_dir)
        emit("progress", task_id=task_id, phase="download", percent=10, msg="影片就緒")

        data = self._process_video(video_path, link, task_id, preferred_lang)
        self.processed[link] = data
        emit("progress", task_id=task_id, phase="extract", percent=60, msg="抽幀完成")
        return data

    def _process_video(self, video_path: str, video_url: str, task_id: str,
                       preferred_lang: Optional[str] = None) -> dict:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"影片不存在: {video_path}")

        total_frames, _, _, fps = get_video_info(video_path)
        if total_frames == 0 or fps == 0:
            raise ValueError(f"無法獲取影片資訊: {video_path}")
        duration: float = total_frames / fps

        emit("progress", task_id=task_id, phase="audio", percent=60, msg="開始提取音訊")
        audio_path: str = extract_audio(video_path)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音訊不存在: {audio_path}")
        emit("progress", task_id=task_id, phase="audio", percent=62, msg="音訊就緒")

        transcript, lang = "", None
        try:
            emit("progress", task_id=task_id, phase="transcribe", percent=20, msg="開始轉錄")
            transcript, lang = transcribe_audio(audio_path, video_url, self.output_dir, preferred_lang)
            emit("progress", task_id=task_id, phase="transcribe", percent=45, msg="轉錄完成")
        except Exception as e:
            logger.error(f"轉錄失敗: {e}")
            emit("log", task_id=task_id, msg=f"轉錄失敗: {e}")

        frames_dir: str = os.path.join(self.output_dir, "frames", os.path.basename(video_path).split('.')[0])
        os.makedirs(frames_dir, exist_ok=True)

        interval: float = (self._decide_interval(duration) if self.time_interval ==
                           "auto" else float(self.time_interval))
        logger.info(f"使用幀間隔：{interval:.2f} 秒")
        emit("log", task_id=task_id, msg=f"使用幀間隔：{interval:.2f} 秒")

        frames: List[str] = extract_frames(video_path, frames_dir, interval)
        valid_frames: List[str] = [os.path.abspath(f) for f in frames if os.path.exists(f)]
        if not valid_frames:
            raise FileNotFoundError("沒有有效的幀檔案")

        # 抽幀進度大致算到 60%
        emit("progress", task_id=task_id, phase="extract", percent=60, msg=f"使用現有的 {len(valid_frames)} 個幀文件")

        return {
            "video_path": video_path,
            "audio_path": audio_path,
            "transcript": transcript,
            "frames": valid_frames,
            "duration": duration,
            "lang": lang,
        }

# ---------- Cleanup & Signals ----------


def cleanup_files(path: str) -> None:
    if os.path.exists(path):
        try:
            logger.info(f"清理: {path}")
            shutil.rmtree(path)
        except PermissionError as e:
            logger.warning(f"無法刪除部分檔案（可能正在使用中）: {e}")
        except Exception as e:
            logger.error(f"清理失敗: {str(e)}")


def signal_handler(signum: int, frame) -> None:
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
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# ---------- Main ----------


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="影片比對系統")
    parser.add_argument("--ref", required=True, help="參考影片連結")
    parser.add_argument("--comp", nargs='+', required=True, help="比對影片連結")
    parser.add_argument("--output", default=str(DOWNLOADS_DIR), help="輸出資料夾")
    parser.add_argument("--cache", default=str(CACHE_DIR), help="快取資料夾")
    parser.add_argument("--interval", default="auto", help="幀提取時間間隔（秒），可設為 auto 表示自動決定")
    parser.add_argument("--keep", action="store_true", help="是否保留中間檔案")
    args = parser.parse_args()

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

        # 定義 ref/targets 任務 id
        ref_task_id = f"ref-{sha10(args.ref)}"
        targets: List[Dict[str, str]] = [{"url": link, "task_id": sha10(
            link)} for link in args.comp]

        # 先把任務公告出去（方便前端立刻建卡）
        emit("hello", ref={"task_id": ref_task_id, "url": args.ref}, targets=targets)

        processor: VideoProcessor = VideoProcessor(args.output, args.interval)

        # 下載並處理參考影片
        emit("progress", task_id=ref_task_id, phase="queued", percent=1, msg="佇列中")
        ref_data: dict = processor.download_and_process(args.ref, task_id=ref_task_id, is_ref=True)
        ref_lang = ref_data["lang"]

        # 不要直接 100%，先標示「準備開始比對」
        emit("progress", task_id=ref_task_id, phase="compare", percent=60, msg=f"參考影片處理完成，開始比對(0/{len(targets)})")

        done_lock = threading.Lock()
        done_count = {"n": 0}  # 可變容器，方便閉包內修改
        total_targets = max(1, len(targets))  # 避免除以 0

        def bump_ref_progress(note: str = ""):
            with done_lock:
                n = done_count["n"]
                # 讓整體比對進度在 60% → 100% 之間線性推進
                pct = 60 + int((n / total_targets) * 40)
                msg = f"比對進度：{n}/{total_targets}"
                if note:
                    msg += f"({note})"
                emit("progress", task_id=ref_task_id, phase="compare", percent=min(pct, 99), msg=msg)

        comparison_results: List[dict] = []

        def process_and_compare(link: str, task_id: str) -> None:
            try:
                # 在 ref 卡打一行目前開始比對哪支
                emit("log", task_id=ref_task_id, msg=f"開始與目標影片比對：{link}")
                emit("progress", task_id=task_id, phase="queued", percent=1, msg="佇列中")

                comp_data: dict = processor.download_and_process(link, task_id=task_id, preferred_lang=ref_lang)
                required_paths: List[str] = [ref_data["audio_path"],
                                             comp_data["audio_path"]] + ref_data["frames"] + comp_data["frames"]
                if not all(os.path.exists(p) for p in required_paths):
                    raise FileNotFoundError("部分文件不存在")

                result: dict = calculate_overall_similarity(
                    ref_data["audio_path"], comp_data["audio_path"],
                    ref_data["frames"], comp_data["frames"],
                    ref_data["transcript"], comp_data["transcript"],
                    max(ref_data["duration"], comp_data["duration"]),
                    emit_cb=emit, task_id=task_id, link=link
                )
                result["link"] = link
                comparison_results.append(result)

                emit("done", task_id=task_id, link=link,
                     audio=result.get("audio_similarity", 0),
                     visual=result.get("image_similarity", 0),
                     text=result.get("text_similarity", 0),
                     score=result.get("overall_similarity", 0))
                emit("progress", task_id=task_id, phase="compare", percent=100, msg="完成")

                # 更新 ref 整體進度
                with done_lock:
                    done_count["n"] += 1
                    n = done_count["n"]
                bump_ref_progress(note=f"{link} 完成")

                # 全部完成 → 把 ref 補到 100%
                if n >= total_targets:
                    emit("progress", task_id=ref_task_id, phase="compare", percent=100, msg="全部比對完成")
            except Exception as e:
                emit("error", task_id=task_id, msg=str(e), link=link)
                # 失敗也算進度，避免卡住
                with done_lock:
                    done_count["n"] += 1
                bump_ref_progress(note=f"{link} 失敗")

        max_workers: int = max(2, min(4, multiprocessing.cpu_count() - 2))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for tgt in targets:
                executor.submit(process_and_compare, tgt["url"], tgt["task_id"])

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
