# backend/api/compare_api.py
from __future__ import annotations

import asyncio
import hashlib
import json
import locale
import os
import re
import signal
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ───────── Logging：鏡像到後端 console ─────────
import logging
LOG = logging.getLogger("compare.sse")
if not LOG.handlers:
    LOG.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(sh)

SERVER_ECHO = True  # 關閉鏡像可改 False

# ───────── Router & globals ─────────


@asynccontextmanager
async def router_lifespan(app):
    try:
        yield
    finally:
        try:
            if RUNNING_PROC and RUNNING_PROC.poll() is None:
                RUNNING_PROC.terminate()
        except Exception:
            pass

router = APIRouter(lifespan=router_lifespan)

EVENT_QUEUE: asyncio.Queue[dict] = asyncio.Queue()
RUNNING_PROC: Optional[subprocess.Popen] = None
CURRENT_TASK_IDS: List[Dict[str, str]] = []  # [{task_id,url,ref_url}...]

# ───────── utilities ─────────


def _enqueue(e: dict) -> None:
    try:
        EVENT_QUEUE.put_nowait(e)
    except Exception:
        pass


def _ts() -> int:
    return int(time.time() * 1000)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def make_task_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


def video_id(u: str) -> str:
    """Try to extract a stable id from common video sites."""
    try:
        from urllib.parse import urlparse, parse_qs
        o = urlparse(u)
        host = (o.hostname or "").replace("www.", "")
        if "youtu.be" in host:
            return o.path.strip("/").split("/")[-1]
        if "youtube.com" in host:
            q = parse_qs(o.query)
            if q.get("v"):  # normal watch?v=ID
                return q["v"][0]
            # shorts,/live, etc.
            tail = o.path.strip("/").split("/")[-1]
            if tail:
                return tail
            return u
        if ("bilibili.com" in host) or ("b23.tv" in host):
            m = re.search(r"(BV[a-zA-Z0-9]+)", u)
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

# ───────── models ─────────


class CompareReq(BaseModel):
    ref: str
    comp: List[str]
    interval: str = "auto"
    keep: bool = False


class StatusReq(BaseModel):
    ref: str
    comp: List[str]


class StatusItem(BaseModel):
    url: str
    phase: str
    percent: int
    cached_flags: Dict[str, bool]


class CancelReq(BaseModel):
    task_ids: List[str]


class CancelResp(BaseModel):
    ok: bool
    killed: bool = False

# ───────── SSE：/api/events ─────────


@router.get("/events")
async def events():
    async def gen():
        # 初次連線把目前任務丟一次 queued 事件
        if CURRENT_TASK_IDS:
            now = _ts()
            for it in CURRENT_TASK_IDS:
                yield {"event": "message", "data": json.dumps({
                    "type": "progress", "task_id": it["task_id"], "url": it["url"], "ref_url": it["ref_url"],
                    "phase": "queued", "percent": 1, "msg": "佇列中", "ts": now
                })}
        while True:
            e = await EVENT_QUEUE.get()
            yield {"event": "message", "data": json.dumps(e)}
    return EventSourceResponse(gen())

# ───────── 主流程：啟動比對（POST） ─────────


@router.post("/compare")
async def compare(req: CompareReq):
    ref = req.ref
    comp = req.comp
    interval = req.interval
    keep = req.keep

    global RUNNING_PROC, CURRENT_TASK_IDS
    if RUNNING_PROC and RUNNING_PROC.poll() is None:
        raise HTTPException(status_code=409, detail="已有比對任務在進行中")

    # 初始化任務 id 並廣播 queued
    CURRENT_TASK_IDS = []
    now = _ts()
    for link in comp:
        tid = make_task_id(link)
        CURRENT_TASK_IDS.append({"task_id": tid, "url": link, "ref_url": ref})
        _enqueue({"type": "progress", "task_id": tid, "url": link, "ref_url": ref,
                  "phase": "queued", "percent": 1, "msg": "佇列中", "ts": now})

    # 組命令（一次帶入所有比對影片）
    BASE_DIR = Path(__file__).resolve().parents[2]
    DOWNLOADS_DIR = BASE_DIR / "downloads"
    CACHE_DIR = BASE_DIR / "feature_cache"
    cmd = [
        sys.executable, "-m", "cli.main",
        "--ref", ref,
        "--interval", interval,
        "--output", str(DOWNLOADS_DIR),
        "--cache", str(CACHE_DIR),
        "--comp", *comp,
    ]
    if keep:
        cmd.append("--keep")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    if os.name != "nt":
        RUNNING_PROC = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False,
            preexec_fn=os.setsid, env=env
        )
    else:
        RUNNING_PROC = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, env=env
        )

    # ── 解析流 → 推 SSE 事件 ──
    class ProcContext:
        def __init__(self, ref_url: str):
            self.ref_url = ref_url
            self.current_url: Optional[str] = None
            self.current_task_id: Optional[str] = None
            # 以輸出檔名/影片ID映射回任務，避免多任務併發時跑錯卡
            self.file_task: Dict[str, str] = {}   # filename/vid -> task_id

    def decode_line(b: bytes) -> str:
        if not b:
            return ""
        for enc in ("utf-8", "utf-8-sig", locale.getpreferredencoding(False), "cp950", "big5", "latin1"):
            try:
                return b.decode(enc, errors="replace")
            except Exception:
                continue
        return b.decode("utf-8", errors="replace")

    # 進度區間配置（你指定的比例）
    RANGE = {
        "download": (0, 10),
        "transcribe": (10, 45),
        "extract": (45, 60),
        "audio": (60, 80),
        "image": (80, 90),
        "text": (90, 100),
    }

    # 通用 regex
    PCT_ANY = re.compile(r"(\d+(?:\.\d+)?)%")  # 行內任何百分比，取最後一個
    # 例：轉錄進度:  13%|█▎        | 2/16 ...
    ASR_PCT = re.compile(r"(?:轉錄進度[:：]?\s*)?(\d{1,3})%\|")
    # 例：downloads\XXX\YYY.mp4 / .m4a / .webm / .wav / .info.json
    FILE_TOK = re.compile(r"downloads[\\/]([^\s\"']+\.(?:mp4|m4a|webm|wav|info\.json))", re.I)
    # 例：watch_XXXXXXXXXXX / BVxxxx / 其他
    VID_IN_PATH = re.compile(r"(?:watch_([0-9a-f]+)|\b(BV[0-9A-Za-z]+)\b|([A-Za-z0-9_-]{11}))")

    FRAMES_DONE = re.compile(r"(?:成功提取|提取)\s+(\d+)\s*個?幀")
    FRAMES_SAMPLED = re.compile(r"採樣幀數[:：]?\s*視頻1=(\d+)[^0-9]+視頻2=(\d+)")
    # 通用錨點（站無關）
    RE_DL_WEBPAGE = re.compile(r"\[[^\]]+\].*Downloading webpage", re.I)
    RE_DL_FORMATS = re.compile(r"\[[^\]]+\].*Downloading (?:video )?formats", re.I)
    RE_DL_ONE_FORMAT = re.compile(r"\[info\].*Downloading\s+\d+\s+format", re.I)
    RE_DL_DEST = re.compile(r"\[download\]\s*Destination:", re.I)
    RE_DL_MERGE = re.compile(r"\[Merger\]\s*Merging formats into", re.I)
    RE_DL_DONE_LINE = re.compile(r"(下載完成|has already been downloaded|100%.*?in\s)", re.I)

    RE_ASR_START = re.compile(r"(開始 Whisper 轉錄|開始轉錄|Transcrib\w+ start|Start(?:ing)? transcrib\w+)", re.I)
    RE_ASR_MODEL_OK = re.compile(r"(Whisper 模型載入完成|Whisper model loaded)", re.I)
    RE_ASR_SAVED = re.compile(r"(轉錄已儲存|Saved transcript|wrote transcript)", re.I)

    RE_FRAMES_START = re.compile(r"(開始抽幀|提取幀|extracting frames|using existing\s+\d+\s+frames)", re.I)
    RE_FRAMES_EXIST = re.compile(r"(使用現有的\s+\d+\s+個幀文件|using existing\s+\d+\s+frames)", re.I)

    RE_AUDIO_PHASE = re.compile(r"(開始計算音頻相似度|Audio similarity|音頻相似度[:：])", re.I)
    RE_IMAGE_PHASE = re.compile(r"(開始計算圖像相似度|Image similarity|圖像相似度[:：])", re.I)
    RE_TEXT_PHASE = re.compile(r"(開始計算文本相似度|Text similarity|文本相似度[:：])", re.I)

    def phase_event(ctx: ProcContext, phase: str, percent: float, msg: str) -> dict:
        return {
            "type": "progress",
            "task_id": ctx.current_task_id,
            "url": ctx.current_url,
            "ref_url": ctx.ref_url,
            "phase": phase,
            "percent": round(percent, 2),
            "msg": msg.strip(),
            "ts": _ts()
        }

    def map_download_pct(x: float) -> float:
        a, b = RANGE["download"]
        return _lerp(a, b, max(0.0, min(1.0, x / 100.0)))

    def set_task_by_url_or_file(decoded: str, ctx: ProcContext):
        """盡可能把當前行歸到正確任務：
           1) 行內 URL
           2) 行內輸出檔名/影片 ID
           3) fallback：維持前一個
        """
        # 1) URL
        murl = re.search(r"(https?://[^\s'\"<>]+)", decoded)
        if murl:
            url = murl.group(1)
            ctx.current_url = url
            ctx.current_task_id = make_task_id(url)
            # 也順便建立 URL → 可能的 vid 提示
            vid = video_id(url)
            if vid:
                ctx.file_task[vid] = ctx.current_task_id
            return

        # 2) 檔名/影片ID提示
        mfile = FILE_TOK.search(decoded)
        if mfile:
            token = mfile.group(1)
            # 嘗試從檔名抓出 video token
            mid = VID_IN_PATH.search(token)
            if mid:
                for g in mid.groups():
                    if not g:
                        continue
                    tid = ctx.file_task.get(g)
                    if tid:
                        # 以 vid 反查 URL 對應的 task
                        ctx.current_task_id = tid
                        # 找 URL（若前面已紀錄）
                        for it in CURRENT_TASK_IDS:
                            if tid == it["task_id"]:
                                ctx.current_url = it["url"]
                                break
                        return
                # 沒命中既有的，先把此 token 暫綁目前 task
                if ctx.current_task_id:
                    for g in mid.groups():
                        if g:
                            ctx.file_task[g] = ctx.current_task_id
                            break
            else:
                # 直接用完整檔名作為 key
                if ctx.current_task_id:
                    ctx.file_task[token] = ctx.current_task_id
                else:
                    # 沒有當前 task，就嘗試把檔名中的 BV / 11 字 ID 映射到任務
                    mid2 = VID_IN_PATH.search(token)
                    if mid2:
                        for g in mid2.groups():
                            if not g:
                                continue
                            tid = ctx.file_task.get(g)
                            if tid:
                                ctx.current_task_id = tid
                                for it in CURRENT_TASK_IDS:
                                    if tid == it["task_id"]:
                                        ctx.current_url = it["url"]
                                        break
                                return

    def line_to_progress_event(decoded: str, ctx: ProcContext) -> Optional[dict]:
        text = decoded

        # 先決定這行該屬於哪個任務
        set_task_by_url_or_file(decoded, ctx)

        # 百分比（通吃各站）：取行內最後一個 %
        m_all = PCT_ANY.findall(text)
        if m_all:
            pct = float(m_all[-1])
            return phase_event(ctx, "download", map_download_pct(pct), text)

        # 轉錄進度條（如：轉錄進度: 13%|…）
        m = ASR_PCT.search(text)
        if m:
            pct = min(100.0, float(m.group(1)))
            a, b = RANGE["transcribe"]
            return phase_event(ctx, "transcribe", _lerp(a, b, pct / 100.0), text)

        # 抽幀（以「成功提取 N 幀 / 採樣幀數」估算）
        m = FRAMES_DONE.search(text)
        if m:
            done = float(m.group(1))
            target = 300.0
            m2 = FRAMES_SAMPLED.search(text)
            if m2:
                target = (float(m2.group(1)) + float(m2.group(2))) / 2.0
                target = max(target, done)
            t = max(0.0, min(1.0, done / max(1.0, target)))
            a, b = RANGE["extract"]
            return phase_event(ctx, "extract", _lerp(a, b, t), text)

        # 下載階段通用錨點（不看站名）
        if RE_DL_WEBPAGE.search(text):
            return phase_event(ctx, "download", max(RANGE["download"][0], 2), text)
        if RE_DL_FORMATS.search(text):
            return phase_event(ctx, "download", max(RANGE["download"][0], 4), text)
        if RE_DL_ONE_FORMAT.search(text):
            return phase_event(ctx, "download", max(RANGE["download"][0], 6), text)
        if RE_DL_DEST.search(text):
            return phase_event(ctx, "download", max(RANGE["download"][0], 7), text)
        if RE_DL_MERGE.search(text):
            return phase_event(ctx, "download", RANGE["download"][1] - 0.2, text)
        if RE_DL_DONE_LINE.search(text):
            return phase_event(ctx, "download", RANGE["download"][1], text)

        # 段落起點（沒有百分比也推進到段落起點）
        if re.search(r"(開始下載|下載影片|影片已存在且大小正常|File is already downloaded)", text, re.I):
            return phase_event(ctx, "download", RANGE["download"][0], text)
        if RE_ASR_START.search(text):
            return phase_event(ctx, "transcribe", max(RANGE["transcribe"][0], 12), text)
        if RE_ASR_MODEL_OK.search(text):
            a, b = RANGE["transcribe"]
            return phase_event(ctx, "transcribe", max(_lerp(a, b, 0.2), 20), text)
        if RE_ASR_SAVED.search(text):
            return phase_event(ctx, "transcribe", RANGE["transcribe"][1], text)

        if RE_FRAMES_START.search(text):
            return phase_event(ctx, "extract", RANGE["extract"][0], text)
        if RE_FRAMES_EXIST.search(text):
            return phase_event(ctx, "extract", RANGE["extract"][1], text)

        # 三段比對（靠關鍵字把條推到該段終點，讓 UI 流暢）
        if RE_AUDIO_PHASE.search(text):
            return phase_event(ctx, "audio", RANGE["audio"][1], text)
        if RE_IMAGE_PHASE.search(text):
            return phase_event(ctx, "image", RANGE["image"][1], text)
        if RE_TEXT_PHASE.search(text):
            return phase_event(ctx, "text", RANGE["text"][1] - 1, text)  # 先到 99

        return None

    def start_reader_threads(proc: subprocess.Popen, out_buf: List[str], err_buf: List[str], ctx: ProcContext):
        def feed(decoded_line: str):
            # yt-dlp 常用 \r 覆寫進度，統一視為換行分割
            parts = decoded_line.replace("\r", "\n").split("\n")
            for part in parts:
                if not part.strip():
                    continue
                if SERVER_ECHO:
                    LOG.info(part.strip())
                ev = line_to_progress_event(part, ctx)
                if ev:
                    _enqueue(ev)
                else:
                    _enqueue({"type": "log", "task_id": ctx.current_task_id, "url": ctx.current_url,
                              "ref_url": ctx.ref_url, "msg": part.strip(), "ts": _ts()})

        def read_stream(stream, buf, is_err: bool):
            while True:
                if proc.poll() is not None and stream.closed:
                    break
                try:
                    line = stream.readline()
                    if not line:
                        if proc.poll() is not None:
                            break
                        time.sleep(0.01)
                        continue
                    decoded = decode_line(line)
                    buf.append(decoded)
                    feed(decoded)
                except Exception as ex:
                    if SERVER_ECHO:
                        LOG.exception("read_stream error: %s", ex)
                    break

        threading.Thread(target=read_stream, args=(proc.stdout, out_buf, False), daemon=True).start()
        threading.Thread(target=read_stream, args=(proc.stderr, err_buf, True), daemon=True).start()

    out_buf: List[str] = []
    err_buf: List[str] = []
    ctx = ProcContext(ref)
    if CURRENT_TASK_IDS:
        first = CURRENT_TASK_IDS[0]["url"]
        ctx.current_url = first
        ctx.current_task_id = make_task_id(first)
    start_reader_threads(RUNNING_PROC, out_buf, err_buf, ctx)

    # 等待完成 → 送 done 與 100%
    def _wait_and_emit_done():
        global RUNNING_PROC
        code = RUNNING_PROC.wait()
        joined = "\n".join(out_buf)
        results = _extract_results_from_stdout(joined)

        id_to_task = {video_id(it["url"]): it["task_id"] for it in CURRENT_TASK_IDS}
        url_to_task = {it["url"]: it["task_id"] for it in CURRENT_TASK_IDS}

        if results and isinstance(results, list):
            for item in results:
                link = item.get("link")
                tid = None
                if link:
                    vid = video_id(link)
                    tid = id_to_task.get(vid) or url_to_task.get(link) or make_task_id(link)
                _enqueue({
                    "type": "done", "task_id": tid, "url": link, "ref_url": ref,
                    "score": round(float(item.get("overall_similarity", 0.0)) * 100, 2),
                    "visual": float(item.get("image_similarity", 0.0)),
                    "audio": float(item.get("audio_similarity", 0.0)),
                    "text": float(item.get("text_similarity", 0.0)),
                    "hot": item.get("hotspots") or [], "ts": _ts(),
                })
                _enqueue({  # 最終補滿
                    "type": "progress", "task_id": tid, "url": link, "ref_url": ref,
                    "phase": "compare", "percent": 100, "msg": "完成", "ts": _ts()
                })
        else:
            _enqueue({"type": "log", "task_id": None, "url": None, "ref_url": ref,
                      "msg": f"比對程序結束（exit={code})", "ts": _ts()})

        RUNNING_PROC = None
        # 只有成功且有結果才收尾參考卡
        if code == 0 and results and isinstance(results, list) and len(results) > 0:
            _enqueue({
                "type": "progress",
                "task_id": f"ref-{video_id(ref)}",
                "url": ref, "ref_url": ref,
                "phase": "compare", "percent": 100, "msg": "完成", "ts": _ts()
            })

    threading.Thread(target=_wait_and_emit_done, daemon=True).start()

    return {"task_ids": CURRENT_TASK_IDS, "cmd": cmd}

# ───────── 解析 stdout 最終 JSON ─────────


def _extract_results_from_stdout(text: str):
    try:
        m = re.search(r"(\[\s*\{.*\}\s*\])\s*$", text, re.S)
        if m:
            return json.loads(m.group(1))
    except Exception:
        pass
    return None

# ───────── 狀態查詢（僅影響 UI） ─────────


@router.post("/status")
async def status(req: StatusReq) -> List[StatusItem]:
    BASE_DIR = Path(__file__).resolve().parents[2]
    DOWNLOADS_DIR = BASE_DIR / "downloads"
    CACHE_DIR = BASE_DIR / "feature_cache"
    FRAMES_DIR = DOWNLOADS_DIR / "frames"

    def probe_one(url: str) -> StatusItem:
        vid = video_id(url)
        mp4 = DOWNLOADS_DIR / f"{vid}.mp4"
        transcript = CACHE_DIR / f"{vid}_transcript.json"
        frame_dir = FRAMES_DIR / vid

        has_video = mp4.exists() and mp4.stat().st_size > 0
        has_transcript = transcript.exists()
        has_frames = frame_dir.exists() and any(
            p.name.startswith(f"{vid}_frame_") and p.suffix == ".jpg" for p in frame_dir.iterdir()
        )

        flags = {"video": has_video, "transcript": has_transcript, "frames": has_frames}
        if has_frames:
            phase, pct = "extract", 60
        elif has_transcript:
            phase, pct = "transcribe", 45
        elif has_video:
            phase, pct = "download", 10
        else:
            phase, pct = "queued", 1
        return StatusItem(url=url, phase=phase, percent=pct, cached_flags=flags)

    urls = [req.ref] + list(req.comp or [])
    return [probe_one(u) for u in urls]

# ───────── 取消 ─────────


@router.post("/cancel", response_model=CancelResp)
async def cancel(req: CancelReq):
    global RUNNING_PROC, CURRENT_TASK_IDS
    want = set(req.task_ids or [])
    for item in CURRENT_TASK_IDS:
        if (not want) or (item["task_id"] in want):
            _enqueue({"type": "canceled", "task_id": item["task_id"], "url": item["url"],
                      "ref_url": item["ref_url"], "ts": _ts()})
    if RUNNING_PROC and RUNNING_PROC.poll() is None:
        kill_tree(RUNNING_PROC)
        return CancelResp(ok=True, killed=True)
    return CancelResp(ok=True, killed=False)
