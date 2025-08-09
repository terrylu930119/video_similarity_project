# backend/api/compare_api.py
from __future__ import annotations

from pydantic import BaseModel
from typing import Dict, List
from .compare_service import service
from contextlib import asynccontextmanager
from fastapi import APIRouter, HTTPException


# ───────── Pydantic 模型（供文件/驗證/回應用） ─────────

class CompareReq(BaseModel):
    ref: str
    comp: List[str]
    interval: str = "auto"
    keep: bool = False
    allow_self: bool = False


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


# ───────── Router with lifespan 委派給 service ─────────

@asynccontextmanager
async def router_lifespan(app):
    async with service.lifespan(app):
        yield

router = APIRouter(lifespan=router_lifespan)


# ───────── 路由 ─────────

@router.get("/events")
async def events():
    return await service.sse_events()


@router.post("/compare")
async def compare(req: CompareReq):
    try:
        return service.start_compare(ref=req.ref, comp=req.comp, interval=req.interval, keep=req.keep)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/status")
async def status(req: StatusReq) -> List[StatusItem]:
    items = service.status(ref=req.ref, comp=req.comp)
    return [StatusItem(**it) for it in items]


@router.post("/cancel", response_model=CancelResp)
async def cancel(req: CancelReq):
    result = service.cancel(task_ids=req.task_ids)
    return CancelResp(**result)
