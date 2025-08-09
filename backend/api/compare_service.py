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
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from sse_starlette.sse import EventSourceResponse

LOG = logging.getLogger("compare.sse")
if not LOG.handlers:
    LOG.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(sh)

# 由環境變數控制是否 echo 到伺服器 log（預設開）
SERVER_ECHO = os.getenv("SERVER_ECHO", "1") == "1"


def _ts() -> int:
    return int(time.time() * 1000)


def make_task_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


def video_id(u: str) -> str:
    try:
        from urllib.parse import urlparse, parse_qs
        o = urlparse(u)
        host = (o.hostname or "").replace("www.", "")
        if "youtu.be" in host:
            return o.path.strip("/").split("/")[-1]
        if "youtube.com" in host:
            q = parse_qs(o.query)
            if q.get("v"):
                return q["v"][0]
            tail = o.path.strip("/").split("/")[-1]
            return tail or u
        if ("bilibili.com" in host) or ("b23.tv" in host):
            import re as _re
            m = _re.search(r"(BV[a-zA-Z0-9]+)", u)
            if m:
                return m.group(1)
        return (host + ":" + (o.path.strip("/") or u))[:64]
    except Exception:
        return u[:64]


def kill_tree(p: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], check=False)
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        pass


class CompareService:
    def __init__(self) -> None:
        self.EVENT_QUEUE: asyncio.Queue[dict] = asyncio.Queue()
        self.RUNNING_PROC: Optional[subprocess.Popen] = None
        self.CURRENT_TASK_IDS: List[Dict[str, str]] = []  # 只在 hello 後填入
        self._lock = threading.Lock()

        base = Path(__file__).resolve().parents[2]
        self.BASE_DIR = base
        self.DOWNLOADS_DIR = base / "downloads"
        self.CACHE_DIR = base / "feature_cache"

        # 由 hello 決定
        self._ref_task_id: Optional[str] = None
        self._ref_url: Optional[str] = None
        self._id_to_url: Dict[str, str] = {}

        self._forward_raw_ref_logs = os.getenv("RAW_REF_LOGS", "0") == "1"
        self._forward_raw_target_logs = os.getenv("RAW_TARGET_LOGS", "0") == "1"

    @asynccontextmanager
    async def lifespan(self, app):
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
        async def gen():
            # 不再回放 queued，避免重複；完全交給 hello 建立
            while True:
                e = await self.EVENT_QUEUE.get()
                yield {"event": "message", "data": json.dumps(e)}
        return EventSourceResponse(gen())

    def start_compare(self, *, ref: str, comp: List[str], interval: str = "auto", keep: bool = False):
        with self._lock:
            if self.RUNNING_PROC and self.RUNNING_PROC.poll() is None:
                raise RuntimeError("已有比對任務在進行中")

            # 只啟動子程序；任務卡片等待 hello 再建立
            self.CURRENT_TASK_IDS = []
            self._ref_task_id = f"ref-{make_task_id(ref)}"
            self._ref_url = ref
            self._id_to_url.clear()

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

            buf: List[str] = []
            ctx = _ProcContext(ref_url=ref, ref_task_id=self._ref_task_id)

            t_out = self._start_reader_thread(self.RUNNING_PROC.stdout, buf, ctx)
            t_err = self._start_reader_thread(self.RUNNING_PROC.stderr, buf, ctx)
            threading.Thread(
                target=self._wait_and_emit_done,
                args=(ref, buf, (t_out, t_err)),
                daemon=True
            ).start()

            # 回傳目前尚無 task_ids（等 hello 後前端自會收到）
            return {"task_ids": [], "cmd": cmd}

    def status(self, *, ref: str, comp: List[str]) -> List[dict]:
        def probe_one(url: str) -> dict:
            vid = video_id(url)
            mp4 = self.DOWNLOADS_DIR / f"{vid}.mp4"
            transcript_txt = self.DOWNLOADS_DIR / f"{vid}_transcript.txt"
            frames_dir = self.DOWNLOADS_DIR / "frames" / vid

            has_video = mp4.exists() and mp4.stat().st_size > 0
            has_transcript = transcript_txt.exists()
            has_frames = frames_dir.exists() and any(p.suffix == ".jpg" for p in frames_dir.iterdir())

            if has_frames:
                phase, pct = "extract", 60
            elif has_transcript:
                phase, pct = "transcribe", 45
            elif has_video:
                phase, pct = "download", 10
            else:
                phase, pct = "queued", 1

            return {"url": url, "phase": phase, "percent": pct,
                    "cached_flags": {"video": has_video, "transcript": has_transcript, "frames": has_frames}}

        urls = [ref] + list(comp or [])
        return [probe_one(u) for u in urls]

    def cancel(self, *, task_ids: List[str]) -> dict:
        want = set(task_ids or [])
        for item in self.CURRENT_TASK_IDS:
            if (not want) or (item["task_id"] in want):
                self._enqueue({"type": "canceled", "task_id": item["task_id"], "url": item["url"],
                               "ref_url": item["ref_url"]})
        killed = False
        with self._lock:
            if self.RUNNING_PROC and self.RUNNING_PROC.poll() is None:
                kill_tree(self.RUNNING_PROC)
                killed = True
        return {"ok": True, "killed": killed}

    # ───────── internals ─────────
    def _enqueue(self, e: dict) -> None:
        if "ts" not in e:
            e["ts"] = _ts()
        try:
            self.EVENT_QUEUE.put_nowait(e)
        except Exception:
            pass

    def _decode_line(self, b: bytes) -> str:
        if not b:
            return ""
        for enc in ("utf-8", "utf-8-sig", "cp950", "big5", "latin1"):
            try:
                return b.decode(enc, errors="replace")
            except Exception:
                continue
        return b.decode("utf-8", errors="replace")

    def _normalize_event_numbers(self, obj: dict) -> None:
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
        etype = obj.get("type")

        if etype == "hello":
            # 以 hello 為單一真相，建立所有任務並丟一次 queued
            ref = obj.get("ref") or {}
            targets = obj.get("targets") or []
            self.CURRENT_TASK_IDS = []

            if ref:
                self._ref_task_id = ref.get("task_id") or ctx.ref_task_id
                self._ref_url = ref.get("url") or ctx.ref_url
                if self._ref_task_id and self._ref_url:
                    self._id_to_url[self._ref_task_id] = self._ref_url
                    self.CURRENT_TASK_IDS.append(
                        {"task_id": self._ref_task_id, "url": self._ref_url, "ref_url": self._ref_url})
                    self._enqueue({"type": "progress", "task_id": self._ref_task_id, "url": self._ref_url,
                                   "ref_url": self._ref_url, "phase": "queued", "percent": 1, "msg": "佇列中"})

            for t in targets:
                tid = t.get("task_id") or make_task_id(t.get("url", ""))
                url = t.get("url")
                if tid and url:
                    self._id_to_url[tid] = url
                    self.CURRENT_TASK_IDS.append({"task_id": tid, "url": url, "ref_url": self._ref_url})
                    self._enqueue({"type": "progress", "task_id": tid, "url": url,
                                   "ref_url": self._ref_url, "phase": "queued", "percent": 1, "msg": "佇列中"})
            return

        tid = obj.get("task_id")
        if tid and "url" not in obj:
            u = self._id_to_url.get(tid)
            if u:
                obj["url"] = u
        if "ref_url" not in obj and self._ref_url:
            obj["ref_url"] = self._ref_url

        self._normalize_event_numbers(obj)
        self._enqueue(obj)

    def _start_reader_thread(self, stream, buf: List[str], ctx: "_ProcContext"):
        def run():
            while True:
                if self.RUNNING_PROC and self.RUNNING_PROC.poll() is not None and stream.closed:
                    break
                try:
                    line = stream.readline()
                    if not line:
                        if self.RUNNING_PROC and self.RUNNING_PROC.poll() is not None:
                            break
                        time.sleep(0.01)
                        continue
                    decoded = self._decode_line(line).rstrip("\r\n")
                    if not decoded:
                        continue
                    buf.append(decoded)
                    if SERVER_ECHO:
                        LOG.info(decoded)
                    # JSON 優先
                    try:
                        obj = json.loads(decoded)
                        if isinstance(obj, dict) and obj.get("type"):
                            self._handle_json_event(obj, ctx)
                            continue
                    except Exception:
                        pass
                    # 非 JSON → 視設定決定是否轉發到前端
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
                    # 預設不 forward，直接略過
                except Exception as ex:
                    if SERVER_ECHO:
                        LOG.exception("reader error: %s", ex)
                    break

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t

    def _wait_and_emit_done(self, ref: str, buf: List[str], threads: tuple):
        with self._lock:
            proc = self.RUNNING_PROC
        if not proc:
            return

        code = proc.wait()
        for t in threads:
            try:
                t.join(timeout=0.5)
            except Exception:
                pass

        # 備援：若 CLI 最後輸出整包 JSON 陣列，拆成 done 事件
        try:
            joined = "\n".join(buf)
            results = self._extract_results_from_stdout(joined)
            if results and isinstance(results, list):
                for item in results:
                    link = item.get("link")
                    tid = None
                    if link:
                        for k, v in self._id_to_url.items():
                            if v == link:
                                tid = k
                                break
                        if not tid:
                            tid = make_task_id(link)
                            self._id_to_url[tid] = link

                    evt = {
                        "type": "done", "task_id": tid, "url": link, "ref_url": ref,
                        "score": float(item.get("overall_similarity", 0.0)),
                        "visual": float(item.get("image_similarity", 0.0)),
                        "audio": float(item.get("audio_similarity", 0.0)),
                        "text": float(item.get("text_similarity", 0.0)),
                        "hot": item.get("hotspots") or []
                    }
                    self._normalize_event_numbers(evt)
                    self._enqueue(evt)
                    self._enqueue({
                        "type": "progress", "task_id": tid, "url": link, "ref_url": ref,
                        "phase": "compare", "percent": 100, "msg": "完成"
                    })
        except Exception:
            pass

        with self._lock:
            self.RUNNING_PROC = None

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

    @staticmethod
    def _extract_results_from_stdout(text: str):
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


class _ProcContext:
    def __init__(self, ref_url: str, ref_task_id: str):
        self.ref_url = ref_url
        self.ref_task_id = ref_task_id


service = CompareService()
