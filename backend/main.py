# backend/main.py
from fastapi import FastAPI
from backend.api import compare_api
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 設定 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 掛上路由
app.include_router(compare_api.router, prefix="/api")
