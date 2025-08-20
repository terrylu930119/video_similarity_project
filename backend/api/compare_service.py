# backend/api/compare_service.py
import os
import sys
import time
import json
import signal
import logging
import asyncio
import hashlib
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from contextlib import asynccontextmanager
from sse_starlette.sse import EventSourceResponse

# =============== 日誌設定 ===============
LOG = logging.getLogger("compare.sse")
if not LOG.handlers:
    LOG.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(sh)

# 由環境變數控制是否 echo 到伺服器 log（預設開）
SERVER_ECHO = os.getenv("SERVER_ECHO", "1") == "1"


# =============== 資料結構定義 ===============
class ProgressEventData(NamedTuple):
    """進度事件資料結構"""
    task_id: str
    url: str
    ref_url: str
    phase: str
    percent: int
    msg: str


class TaskEntryData(NamedTuple):
    """任務條目資料結構"""
    task_id: str
    url: str
    ref_url: str


# =============== 工具函式 ===============
def _ts() -> int:
    """取得當前時間戳記（毫秒）"""
    return int(time.time() * 1000)


def make_task_id(url: str) -> str:
    """根據 URL 產生任務 ID"""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


def _extract_youtube_id(url: str, host: str, path: str, query: dict) -> Optional[str]:
    """提取 YouTube 影片 ID"""
    if "youtu.be" in host:
        return path.strip("/").split("/")[-1]
    if "youtube.com" in host:
        if query.get("v"):
            return query["v"][0]
        tail = path.strip("/").split("/")[-1]
        return tail or url
    return None


def _extract_bilibili_id(url: str) -> Optional[str]:
    """提取 Bilibili 影片 ID"""
    import re
    match = re.search(r"(BV[a-zA-Z0-9]+)", url)
    return match.group(1) if match else None


def video_id(u: str) -> str:
    """
    從 URL 中提取影片 ID

    支援的網站：
    - YouTube (youtube.com, youtu.be)
    - Bilibili (bilibili.com, b23.tv)
    - 其他網站使用 host:path 格式
    """
    try:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(u)
        host = (parsed.hostname or "").replace("www.", "")
        path = parsed.path
        query = parse_qs(parsed.query)

        # 嘗試提取 YouTube ID
        youtube_id = _extract_youtube_id(u, host, path, query)
        if youtube_id:
            return youtube_id

        # 嘗試提取 Bilibili ID
        if ("bilibili.com" in host) or ("b23.tv" in host):
            bilibili_id = _extract_bilibili_id(u)
            if bilibili_id:
                return bilibili_id

        # 其他網站使用 host:path 格式
        return (host + ":" + (path.strip("/") or u))[:64]

    except Exception:
        return u[:64]


def kill_tree(p: subprocess.Popen) -> None:
    """強制終止程序及其子程序"""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], check=False)
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        pass


# =============== 程序上下文類別 ===============
class _ProcContext:
    """程序執行上下文，儲存參考影片相關資訊"""

    def __init__(self, ref_url: str, ref_task_id: str):
        self.ref_url = ref_url
        self.ref_task_id = ref_task_id


# =============== 主要服務類別 ===============
class CompareService:
    """影片相似度比對服務主類別"""

    def __init__(self) -> None:
        """初始化比對服務"""
        # 事件佇列和程序管理
        self.EVENT_QUEUE: asyncio.Queue[dict] = asyncio.Queue()
        self.RUNNING_PROC: Optional[subprocess.Popen] = None
        self.CURRENT_TASK_IDS: List[Dict[str, str]] = []  # 只在 hello 後填入
        self._lock = threading.Lock()

        # 目錄設定
        base = Path(__file__).resolve().parents[2]
        self.BASE_DIR = base
        self.DOWNLOADS_DIR = base / "downloads"
        self.CACHE_DIR = base / "feature_cache"

        # 任務相關狀態
        self._ref_task_id: Optional[str] = None
        self._ref_url: Optional[str] = None
        self._id_to_url: Dict[str, str] = {}

        # 日誌轉發設定
        self._forward_raw_ref_logs = os.getenv("RAW_REF_LOGS", "0") == "1"
        self._forward_raw_target_logs = os.getenv("RAW_TARGET_LOGS", "0") == "1"

    @asynccontextmanager
    async def lifespan(self, app):
        """應用程式生命週期管理"""
        try:
            yield
        finally:
            try:
                with self._lock:
                    p = self.RUNNING_PROC
                if p and p.poll() is None:
                    kill_tree(p)
            except Exception:
                pass

    async def sse_events(self):
        """SSE 事件串流"""
        async def gen():
            # 不再回放 queued，避免重複；完全交給 hello 建立
            while True:
                e = await self.EVENT_QUEUE.get()
                yield {"event": "message", "data": json.dumps(e)}
        return EventSourceResponse(gen())

    # =============== 主要業務邏輯 ===============
    def start_compare(self, *, ref: str, comp: List[str], interval: str = "auto", keep: bool = False):
        """
        啟動影片比對任務

        Args:
            ref: 參考影片 URL
            comp: 要比對的影片 URL 列表
            interval: 幀提取間隔
            keep: 是否保留中間檔案
        """
        with self._lock:
            if self.RUNNING_PROC and self.RUNNING_PROC.poll() is None:
                raise RuntimeError("已有比對任務在進行中")

            # 初始化任務狀態
            self._init_task_state(ref)

            # 建立命令列
            cmd = self._build_command(ref, comp, interval, keep)

            # 啟動子程序
            self._start_subprocess(cmd)

            # 啟動讀取器執行緒
            self._start_reader_threads(ref)

            # 回傳目前尚無 task_ids（等 hello 後前端自會收到）
            return {"task_ids": [], "cmd": cmd}

    def status(self, *, ref: str, comp: List[str]) -> List[dict]:
        """
        查詢任務狀態

        Args:
            ref: 參考影片 URL
            comp: 要比對的影片 URL 列表

        Returns:
            各影片的處理狀態列表
        """
        urls = [ref] + list(comp or [])
        return [self._probe_video_status(u) for u in urls]

    def cancel(self, *, task_ids: List[str]) -> dict:
        """
        取消指定任務

        Args:
            task_ids: 要取消的任務 ID 列表

        Returns:
            取消結果
        """
        # 發送取消事件
        self._send_cancel_events(task_ids)

        # 終止程序
        killed = self._terminate_process()

        return {"ok": True, "killed": killed}

    # =============== 私有方法 ===============
    def _init_task_state(self, ref: str) -> None:
        """初始化任務狀態"""
        self.CURRENT_TASK_IDS = []
        self._ref_task_id = f"ref-{make_task_id(ref)}"
        self._ref_url = ref
        self._id_to_url.clear()

    def _build_command(self, ref: str, comp: List[str], interval: str, keep: bool) -> List[str]:
        """建立命令列參數"""
        cmd = [
            sys.executable, "-m", "cli.main",
            "--ref", ref,
            "--interval", interval,
            "--output", str(self.DOWNLOADS_DIR),
            "--cache", str(self.CACHE_DIR),
            "--comp", *comp,
        ]
        if keep:
            cmd.append("--keep")
        return cmd

    def _start_subprocess(self, cmd: List[str]) -> None:
        """啟動子程序"""
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["PLAIN_CONSOLE_LOG"] = "1"

        if os.name != "nt":
            self.RUNNING_PROC = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False,
                preexec_fn=os.setsid, env=env
            )
        else:
            self.RUNNING_PROC = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, env=env
            )

    def _start_reader_threads(self, ref: str) -> None:
        """啟動讀取器執行緒"""
        buf: List[str] = []
        ctx = _ProcContext(ref_url=ref, ref_task_id=self._ref_task_id)

        t_out = self._start_reader_thread(self.RUNNING_PROC.stdout, buf, ctx)
        t_err = self._start_reader_thread(self.RUNNING_PROC.stderr, buf, ctx)

        threading.Thread(
            target=self._wait_and_emit_done,
            args=(ref, buf, (t_out, t_err)),
            daemon=True
        ).start()

    def _probe_video_status(self, url: str) -> dict:
        """探測單一影片的處理狀態"""
        vid = video_id(url)
        mp4 = self.DOWNLOADS_DIR / f"{vid}.mp4"
        transcript_txt = self.DOWNLOADS_DIR / f"{vid}_transcript.txt"
        frames_dir = self.DOWNLOADS_DIR / "frames" / vid

        # 檢查各階段檔案
        has_video = mp4.exists() and mp4.stat().st_size > 0
        has_transcript = transcript_txt.exists()
        has_frames = frames_dir.exists() and any(p.suffix == ".jpg" for p in frames_dir.iterdir())

        # 判斷處理階段和進度
        if has_frames:
            phase, pct = "extract", 60
        elif has_transcript:
            phase, pct = "transcribe", 45
        elif has_video:
            phase, pct = "download", 10
        else:
            phase, pct = "queued", 1

        return {
            "url": url,
            "phase": phase,
            "percent": pct,
            "cached_flags": {
                "video": has_video,
                "transcript": has_transcript,
                "frames": has_frames
            }
        }

    def _send_cancel_events(self, task_ids: List[str]) -> None:
        """發送取消事件"""
        want = set(task_ids or [])
        for item in self.CURRENT_TASK_IDS:
            if (not want) or (item["task_id"] in want):
                self._enqueue({
                    "type": "canceled",
                    "task_id": item["task_id"],
                    "url": item["url"],
                    "ref_url": item["ref_url"]
                })

    def _terminate_process(self) -> bool:
        """終止執行中的程序"""
        killed = False
        with self._lock:
            if self.RUNNING_PROC and self.RUNNING_PROC.poll() is None:
                kill_tree(self.RUNNING_PROC)
                killed = True
        return killed

    def _enqueue(self, e: dict) -> None:
        """將事件加入佇列"""
        if "ts" not in e:
            e["ts"] = _ts()
        try:
            self.EVENT_QUEUE.put_nowait(e)
        except Exception:
            pass

    def _decode_line(self, b: bytes) -> str:
        """解碼位元組為字串，嘗試多種編碼"""
        if not b:
            return ""
        for enc in ("utf-8", "utf-8-sig", "cp950", "big5", "latin1"):
            try:
                return b.decode(enc, errors="replace")
            except Exception:
                continue
        return b.decode("utf-8", errors="replace")

    def _normalize_event_numbers(self, obj: dict) -> None:
        """正規化事件中的數值"""
        if obj.get("type") == "progress":
            pct = obj.get("percent")
            if isinstance(pct, (int, float)):
                obj["percent"] = max(0, min(100, float(pct)))
            return

        if obj.get("type") == "done":
            for k in ("score", "audio", "visual", "text"):
                v = obj.get(k)
                if isinstance(v, (int, float)):
                    x = float(v)
                    if x <= 1.0:
                        x *= 100.0
                    obj[k] = round(x, 2)

    def _handle_json_event(self, obj: dict, ctx: "_ProcContext") -> None:
        """處理 JSON 格式的事件"""
        etype = obj.get("type")

        if etype == "hello":
            self._handle_hello_event(obj, ctx)
            return

        # 補充缺失的欄位
        self._fill_missing_fields(obj)

        # 正規化數值
        self._normalize_event_numbers(obj)

        # 加入佇列
        self._enqueue(obj)

    def _create_task_entry(self, task_id: str, url: str, ref_url: str) -> dict:
        """建立任務條目"""
        return {
            "task_id": task_id,
            "url": url,
            "ref_url": ref_url
        }

    def _create_progress_event(self, data: ProgressEventData) -> dict:
        """建立進度事件"""
        return {
            "type": "progress",
            "task_id": data.task_id,
            "url": data.url,
            "ref_url": data.ref_url,
            "phase": data.phase,
            "percent": data.percent,
            "msg": data.msg
        }

    def _process_reference_video(self, ref_data: dict, ctx: "_ProcContext") -> None:
        """處理參考影片"""
        if not ref_data:
            return

        self._ref_task_id = ref_data.get("task_id") or ctx.ref_task_id
        self._ref_url = ref_data.get("url") or ctx.ref_url

        if self._ref_task_id and self._ref_url:
            self._id_to_url[self._ref_task_id] = self._ref_url
            self.CURRENT_TASK_IDS.append(
                self._create_task_entry(self._ref_task_id, self._ref_url, self._ref_url)
            )

            progress_data = ProgressEventData(
                task_id=self._ref_task_id,
                url=self._ref_url,
                ref_url=self._ref_url,
                phase="queued",
                percent=1,
                msg="佇列中"
            )
            self._enqueue(self._create_progress_event(progress_data))

    def _process_target_videos(self, targets: List[dict]) -> None:
        """處理目標影片列表"""
        for target in targets:
            task_id = target.get("task_id") or make_task_id(target.get("url", ""))
            url = target.get("url")

            if task_id and url:
                self._id_to_url[task_id] = url
                self.CURRENT_TASK_IDS.append(
                    self._create_task_entry(task_id, url, self._ref_url)
                )

                progress_data = ProgressEventData(
                    task_id=task_id,
                    url=url,
                    ref_url=self._ref_url,
                    phase="queued",
                    percent=1,
                    msg="佇列中"
                )
                self._enqueue(self._create_progress_event(progress_data))

    def _handle_hello_event(self, obj: dict, ctx: "_ProcContext") -> None:
        """處理 hello 事件，建立所有任務"""
        ref_data = obj.get("ref") or {}
        targets = obj.get("targets") or []
        self.CURRENT_TASK_IDS = []

        # 處理參考影片
        self._process_reference_video(ref_data, ctx)

        # 處理目標影片
        self._process_target_videos(targets)

    def _fill_missing_fields(self, obj: dict) -> None:
        """補充事件中缺失的欄位"""
        tid = obj.get("task_id")
        if tid and "url" not in obj:
            u = self._id_to_url.get(tid)
            if u:
                obj["url"] = u

        if "ref_url" not in obj and self._ref_url:
            obj["ref_url"] = self._ref_url

    def _start_reader_thread(self, stream, buf: List[str], ctx: "_ProcContext"):
        """啟動讀取器執行緒"""
        def run():
            """執行緒主要邏輯"""
            while True:
                if self._should_stop_reading(stream):
                    break

                try:
                    line = stream.readline()
                    if not line:
                        if self._should_stop_reading(stream):
                            break
                        time.sleep(0.01)
                        continue

                    decoded = self._decode_line(line).rstrip("\r\n")
                    if not decoded:
                        continue

                    buf.append(decoded)

                    if SERVER_ECHO:
                        LOG.info(decoded)

                    if self._try_parse_json(decoded, ctx):
                        continue

                    self._handle_non_json_log(decoded, ctx)

                except Exception as ex:
                    if SERVER_ECHO:
                        LOG.exception("reader error: %s", ex)
                    break

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t

    def _should_stop_reading(self, stream) -> bool:
        """檢查是否應該停止讀取"""
        return (self.RUNNING_PROC and
                self.RUNNING_PROC.poll() is not None and
                stream.closed)

    def _try_parse_json(self, decoded: str, ctx: "_ProcContext") -> bool:
        """嘗試解析 JSON 事件"""
        try:
            obj = json.loads(decoded)
            if isinstance(obj, dict) and obj.get("type"):
                self._handle_json_event(obj, ctx)
                return True
        except Exception:
            pass
        return False

    def _handle_non_json_log(self, decoded: str, ctx: "_ProcContext") -> None:
        """處理非 JSON 格式的日誌"""
        # 用「目前是否有作用中的 target」來判斷這行屬於誰
        is_target = bool(getattr(ctx, "active_target_id", None))

        # 決定要不要轉發
        should_forward = (
            self._forward_raw_target_logs if is_target else self._forward_raw_ref_logs
        )

        if should_forward:
            # 發到對的 task_id（target 時用 active_target_id，否則 ref）
            tid = getattr(ctx, "active_target_id", None) or ctx.ref_task_id
            self._enqueue({
                "type": "log",
                "task_id": tid,
                "url": self._id_to_url.get(tid) or self._id_to_url.get(ctx.ref_task_id) or ctx.ref_url,
                "ref_url": ctx.ref_url,
                "msg": decoded
            })

    def _wait_and_emit_done(self, ref: str, buf: List[str], threads: Tuple[threading.Thread, ...]):
        """等待程序完成並發送完成事件"""
        with self._lock:
            proc = self.RUNNING_PROC
        if not proc:
            return

        # 等待程序結束
        code = proc.wait()

        # 等待讀取器執行緒結束
        self._join_reader_threads(threads)

        # 處理 CLI 輸出結果
        self._process_cli_results(ref, buf)

        # 清理狀態
        with self._lock:
            self.RUNNING_PROC = None

        # 發送參考影片完成事件
        if code == 0 and self._ref_task_id:
            self._enqueue({
                "type": "progress",
                "task_id": self._ref_task_id,
                "url": self._id_to_url.get(self._ref_task_id, ref),
                "ref_url": ref,
                "phase": "compare",
                "percent": 100,
                "msg": "完成"
            })

    def _join_reader_threads(self, threads: Tuple[threading.Thread, ...]) -> None:
        """等待讀取器執行緒結束"""
        for t in threads:
            try:
                t.join(timeout=0.5)
            except Exception:
                pass

    def _process_cli_results(self, ref: str, buf: List[str]) -> None:
        """處理 CLI 輸出的結果"""
        try:
            joined = "\n".join(buf)
            results = self._extract_results_from_stdout(joined)
            if results and isinstance(results, list):
                for item in results:
                    self._emit_result_event(item, ref)
        except Exception:
            pass

    def _emit_result_event(self, item: dict, ref: str) -> None:
        """發送結果事件"""
        link = item.get("link")
        tid = self._get_task_id_for_url(link)

        if not tid and link:
            tid = make_task_id(link)
            self._id_to_url[tid] = link

        evt = {
            "type": "done",
            "task_id": tid,
            "url": link,
            "ref_url": ref,
            "score": float(item.get("overall_similarity", 0.0)),
            "visual": float(item.get("image_similarity", 0.0)),
            "audio": float(item.get("audio_similarity", 0.0)),
            "text": float(item.get("text_similarity", 0.0)),
            "hot": item.get("hotspots") or []
        }

        self._normalize_event_numbers(evt)
        self._enqueue(evt)

        # 發送進度事件
        self._enqueue({
            "type": "progress",
            "task_id": tid,
            "url": link,
            "ref_url": ref,
            "phase": "compare",
            "percent": 100,
            "msg": "完成"
        })

    def _get_task_id_for_url(self, url: str) -> Optional[str]:
        """根據 URL 取得任務 ID"""
        for k, v in self._id_to_url.items():
            if v == url:
                return k
        return None

    @staticmethod
    def _extract_results_from_stdout(text: str):
        """從標準輸出中提取結果 JSON"""
        import re
        candidates = re.findall(r"(\[\s*\{.*?\}\s*\])", text, flags=re.S)
        for s in reversed(candidates):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return obj
            except Exception:
                continue
        return None


# =============== 全域服務實例 ===============
service = CompareService()
