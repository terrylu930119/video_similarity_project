# backend/api/compare_api.py
"""相似度比對 API：封裝比對服務的 HTTP 端點。

檔案用途：
- 暴露與比對流程相關的 REST 端點（事件、啟動比對、查詢狀態、取消任務）。
- 透過 Pydantic 模型進行請求/回應結構的基本驗證與文件化。

設計原則：
- 保持端點薄層（thin controller），主要邏輯委派給 `compare_service`。
- 僅在必要時將 service 例外轉譯為 HTTP 狀態碼。
"""
from __future__ import annotations

from pydantic import BaseModel
from typing import Dict, List
from .compare_service import service
from contextlib import asynccontextmanager
from fastapi import APIRouter, HTTPException


# ───────── Pydantic 模型（供文件/驗證/回應用） ─────────

class CompareReq(BaseModel):
    """啟動比對請求模型。

    Args:
        ref (str): 參考影片 URL。
        comp (List[str]): 需要被比對的一組影片 URL。
        interval (str): 取樣區間策略，預設 "auto"。
        keep (bool): 是否保留中間產物。
    """
    ref: str
    comp: List[str]
    interval: str = "auto"
    keep: bool = False


class StatusReq(BaseModel):
    """查詢任務狀態請求模型。"""
    ref: str
    comp: List[str]


class StatusItem(BaseModel):
    """單一任務狀態回應項目。"""
    url: str
    phase: str
    percent: int
    cached_flags: Dict[str, bool]


class CancelReq(BaseModel):
    """取消任務請求模型。"""
    task_ids: List[str]


class CancelResp(BaseModel):
    """取消任務回應模型。"""
    ok: bool
    killed: bool = False


# ───────── Router with lifespan 委派給 service ─────────

@asynccontextmanager
async def router_lifespan(app):
    async with service.lifespan(app):
        yield

router = APIRouter(lifespan=router_lifespan)


# ───────── 路由 ─────────

@router.get("/events")
async def events():
    """伺服器推送事件（SSE）端點。

    Returns:
        Any: 交由 service 產出的事件串流（取決於 service 實作）。
    """
    return await service.sse_events()


@router.post("")
async def compare(req: CompareReq):
    """啟動比對流程。

    Raises:
        HTTPException: 當服務回報資源衝突（例如相同任務正在執行）。
    """
    try:
        return service.start_compare(ref=req.ref, comp=req.comp, interval=req.interval, keep=req.keep)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/status")
async def status(req: StatusReq) -> List[StatusItem]:
    """查詢一組 URL 對應的比對狀態。"""
    items = service.status(ref=req.ref, comp=req.comp)
    return [StatusItem(**it) for it in items]


@router.post("/cancel", response_model=CancelResp)
async def cancel(req: CancelReq):
    """取消指定的比對任務。"""
    result = service.cancel(task_ids=req.task_ids)
    return CancelResp(**result)
