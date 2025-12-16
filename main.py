# -*- coding: utf-8 -*-
"""
Main Application
主应用程序，整合所有controller路由
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from controller.segment_controller import segment_router
from controller.search_controller import search_router
from controller.model_controller import model_router
from controller.user_controller import user_router
from controller.confirm_controller import confirm_router
from controller.video_controller import video_router, refresh_video_labels_cache
from service.model_manager import get_model_manager

# 创建FastAPI应用实例
app = FastAPI(title="Medical Image Segmentation API", version="1.0.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 预加载模型
@app.on_event("startup")
async def startup_event():
    """应用启动时预加载模型"""
    try:
        model_manager = get_model_manager()
        # 预加载推理模型和FAISS服务（一次性加载，后续统一复用缓存）
        model_manager.get_inference_model(model_path="ckpt/multi_seg_best.weight")
        model_manager.get_faiss_service()
        print("模型预加载完成")
        
        # 刷新视频标签缓存
        refresh_video_labels_cache()
    except Exception as e:
        print(f"模型预加载失败: {e}")

# 包含路由
app.include_router(segment_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(model_router, prefix="/api")
app.include_router(user_router, prefix="/api")
app.include_router(confirm_router, prefix="/api")
app.include_router(video_router, prefix="/api")

@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "图像分割与检索API服务",
        "version": "1.0.0",
        "endpoints": {
            "segment": "/segment/image - 图像分割",
            "search": "/search/faiss - FAISS检索",
            "model": "/model/classes - 获取类别信息, /model/reset - 重置模型"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "message": "服务运行正常"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8080,
        reload=True,
        log_level="info",
        workers=4
    )