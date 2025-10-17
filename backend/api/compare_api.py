# 檔案用途：影片相似度比對 API 路由模組（FastAPI 路由定義與 Pydantic 模型）
"""
影片相似度比對 API 模組

此模組提供影片相似度比對的 RESTful API 介面，包含：
- 比對任務啟動、狀態查詢、取消功能
- Server-Sent Events (SSE) 即時進度推送
- Pydantic 資料驗證模型定義

主要功能：
- /compare: 啟動影片比對任務
- /status: 查詢任務處理狀態  
- /cancel: 取消進行中的任務
- /events: SSE 事件串流
"""

from __future__ import annotations

from pydantic import BaseModel
from typing import Dict, List
from .compare_service import service
from contextlib import asynccontextmanager
from fastapi import APIRouter, HTTPException


# ───────── Pydantic 模型（供文件/驗證/回應用） ─────────

class CompareReq(BaseModel):
    """影片比對請求模型"""
    ref: str  # 參考影片 URL
    comp: List[str]  # 要比對的影片 URL 列表
    interval: str = "auto"  # 幀提取間隔（auto/數字）
    keep: bool = False  # 是否保留中間檔案
    allow_self: bool = False  # 是否允許與自己比對


class StatusReq(BaseModel):
    """狀態查詢請求模型"""
    ref: str  # 參考影片 URL
    comp: List[str]  # 要比對的影片 URL 列表


class StatusItem(BaseModel):
    """單一影片狀態項目模型"""
    url: str  # 影片 URL
    phase: str  # 處理階段（queued/download/transcribe/extract/compare）
    percent: int  # 進度百分比（0-100）
    cached_flags: Dict[str, bool]  # 快取檔案存在標記


class CancelReq(BaseModel):
    """取消任務請求模型"""
    task_ids: List[str]  # 要取消的任務 ID 列表


class CancelResp(BaseModel):
    """取消任務回應模型"""
    ok: bool  # 操作是否成功
    killed: bool = False  # 是否成功終止程序


# ───────── Router with lifespan 委派給 service ─────────

@asynccontextmanager
async def router_lifespan(app):
    """
    路由器生命週期管理器
    
    將應用程式生命週期管理委派給服務層，確保程序正確啟動和清理。
    
    Args:
        app: FastAPI 應用程式實例
        
    Yields:
        None: 在生命週期期間保持運行
    """
    async with service.lifespan(app):
        yield

router = APIRouter(lifespan=router_lifespan)


# ───────── 路由 ─────────

@router.get("/events")
async def events():
    """
    取得 Server-Sent Events 串流
    
    提供即時的任務進度更新，包括：
    - 任務狀態變化
    - 進度百分比更新
    - 錯誤訊息
    - 完成結果
    
    Returns:
        EventSourceResponse: SSE 事件串流回應
    """
    return await service.sse_events()


@router.post("/compare")
async def compare(req: CompareReq):
    """
    啟動影片相似度比對任務
    
    根據提供的參考影片和比對影片列表，啟動非同步比對任務。
    任務將在背景執行，進度可透過 /events 端點監聽。
    
    Args:
        req: 比對請求參數，包含參考影片和比對影片列表
        
    Returns:
        dict: 包含任務 ID 列表和命令列參數
        
    Raises:
        HTTPException: 當已有任務在執行時（409 狀態碼）
    """
    try:
        return service.start_compare(ref=req.ref, comp=req.comp, interval=req.interval, keep=req.keep)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/status")
async def status(req: StatusReq) -> List[StatusItem]:
    """
    查詢影片處理狀態
    
    檢查指定影片的處理進度，包括下載、轉錄、特徵提取等階段。
    
    Args:
        req: 狀態查詢請求，包含參考影片和比對影片列表
        
    Returns:
        List[StatusItem]: 各影片的處理狀態列表
    """
    items = service.status(ref=req.ref, comp=req.comp)
    return [StatusItem(**it) for it in items]


@router.post("/cancel", response_model=CancelResp)
async def cancel(req: CancelReq):
    """
    取消進行中的比對任務
    
    終止指定的比對任務，並清理相關資源。
    
    Args:
        req: 取消請求，包含要取消的任務 ID 列表
        
    Returns:
        CancelResp: 取消操作結果
    """
    result = service.cancel(task_ids=req.task_ids)
    return CancelResp(**result)
