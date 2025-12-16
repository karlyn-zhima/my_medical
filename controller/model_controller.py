# -*- coding: utf-8 -*-
"""
Model Controller
模型管理控制器，处理模型相关的API请求
"""

import os
import json
from fastapi import APIRouter, Form
from typing import List, Optional
from pydantic import BaseModel

from service.model_manager import get_model_manager
from constants.error_codes import ErrorCode
from constants.error_messages import get_error_message
# 创建路由器
model_router = APIRouter(prefix="/model", tags=["模型管理"])

class ApiResponse(BaseModel):
    success: bool
    error_code: Optional[int] = None
    message: Optional[str] = None
    results: Optional[List[dict]] = None

class ClassInfo(BaseModel):
    """类别信息模型"""
    id: int
    name: str
    description: Optional[str] = None

@model_router.get("/classes", response_model=ApiResponse)
async def get_classes():
    """
    获取支持分类的类别信息，初始化存储在json中，后续修改回修改json文件，若是使用频繁，后续加入到缓存中
    
    Returns:
        ApiResponse: 包含类别信息的响应
    """
    try:
        # 尝试从配置文件读取类别信息
        config_path = "config/config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                classes = config.get('classes', [])
        else:
            # 默认类别信息
            classes = [
                {"id": 1, "name": "skin", "description": "皮肤区域"},
                {"id": 2, "name": "breast tumors", "description": "乳腺肿瘤"},
                {"id": 3, "name": "background", "description": "背景区域"}
            ]
        
        return ApiResponse(success=True, results=classes)
    
    except Exception as e:
        return ApiResponse(
            success=False,
            error_code=int(ErrorCode.SYSTEM_ERROR),
            message=get_error_message(ErrorCode.SYSTEM_ERROR)
        )

@model_router.post("/model/reset", response_model=ApiResponse)
async def reset_model():
    """重置模型"""
    try:
        # 使用模型管理器重置模型
        model_manager = get_model_manager()
        model_manager.reset_model()
        
        return ApiResponse(
            code=ErrorCode.SUCCESS,
            message=get_error_message(ErrorCode.SUCCESS),
            data={"status": "Model reset successfully"}
        )
    except Exception as e:
        return ApiResponse(
            code=ErrorCode.INTERNAL_ERROR,
            message=get_error_message(ErrorCode.INTERNAL_ERROR),
            data={"error": str(e)}
        )