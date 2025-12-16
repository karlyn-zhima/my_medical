# -*- coding: utf-8 -*-
"""
Search Controller
搜索控制器，处理FAISS检索相关的API请求
"""

import os
import hashlib
from fastapi import APIRouter, File, UploadFile, Form
from typing import List, Optional
from pydantic import BaseModel

from service.model_manager import get_model_manager
from constants.error_codes import ErrorCode
from constants.error_messages import get_error_message

# 创建路由器
search_router = APIRouter(prefix="/search", tags=["图像检索"])

class ApiResponse(BaseModel):
    success: bool
    error_code: Optional[int] = None
    message: Optional[str] = None
    results: Optional[List[dict]] = None

class AddToIndexRequest(BaseModel):
    skip_duplicates: Optional[bool] = True
    label_dir: Optional[str] = None
    mask_dir: Optional[str] = None

# 全局FAISS服务实例
faiss_service = None

def get_faiss_service():
    """获取FAISS服务实例"""
    model_manager = get_model_manager()
    return model_manager.get_faiss_service()

def set_faiss_service(service):
    """设置FAISS服务实例"""
    # 使用模型管理器，不需要手动设置FAISS服务
    # FAISS服务会在模型重置后自动重新初始化
    pass

def set_faiss_service(service):
    """设置FAISS服务实例"""
    # 使用模型管理器，不需要手动设置FAISS服务
    # FAISS服务会在模型重置后自动重新初始化
    pass

@search_router.post("/faiss", response_model=ApiResponse)
async def search_faiss(
    image: UploadFile = File(...)
):
    """
    接收图像作为参数，返回 Top-k 检索结果
    
    Args:
        image: 上传的图像文件
        
    Returns:
        ApiResponse: 包含检索结果的响应
    """
    # 读取图片数据并计算哈希值
    image_data = await image.read()
    image_hash = hashlib.md5(image_data).hexdigest()
    temp_image_path = f"temp_{image_hash}.png"
    
    try:
        with open(temp_image_path, "wb") as buffer:
            buffer.write(image_data)
        
        # 获取FAISS服务并运行检索
        service = get_faiss_service()
        results = service.search_faiss(temp_image_path, 1)
        
        return ApiResponse(success=True, results=results)
    
    except Exception as e:
        return ApiResponse(
            success=False, 
            error_code=int(ErrorCode.SYSTEM_ERROR),
            message=get_error_message(ErrorCode.SYSTEM_ERROR)
        )
    
    finally:
        # 清理暂存文件
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

@search_router.post("/add", response_model=ApiResponse)
async def add_to_rag(
    images: List[UploadFile] = File(...),
    skip_duplicates: bool = Form(True),
    label_dir: Optional[str] = Form(None),
    mask_dir: Optional[str] = Form(None)
):
    """
    添加新的图像记录到RAG索引中
    
    Args:
        images: 上传的图像文件列表
        skip_duplicates: 是否跳过已存在的重复图像
        label_dir: 标签目录路径（可选）
        mask_dir: 掩码目录路径（可选）
        
    Returns:
        ApiResponse: 包含添加结果的响应
    """
    temp_image_paths = []
    try:
        # 保存所有上传的图片到临时文件
        for image in images:
            # 读取图片数据并计算哈希值
            image_data = await image.read()
            image_hash = hashlib.md5(image_data).hexdigest()
            temp_image_path = f"temp_{image_hash}_{image.filename}"
            
            with open(temp_image_path, "wb") as buffer:
                buffer.write(image_data)
            temp_image_paths.append(temp_image_path)
        
        # 获取FAISS服务并添加到索引
        service = get_faiss_service()
        result = service.add_to_index(
            image_paths=temp_image_paths,
            skip_duplicates=skip_duplicates,
            label_dir=label_dir,
            mask_dir=mask_dir
        )
        
        if result["success"]:
            return ApiResponse(
                success=True, 
                message=result["message"],
                results=[{
                    "added_count": result["added_count"],
                    "total_count": result["total_count"]
                }]
            )
        else:
            return ApiResponse(
                success=False,
                error_code=int(ErrorCode.SYSTEM_ERROR),
                message=result["message"]
            )
    
    except Exception as e:
        return ApiResponse(
            success=False, 
            error_code=int(ErrorCode.SYSTEM_ERROR),
            message=f"添加记录时发生错误: {str(e)}"
        )
    
    finally:
        # 清理所有暂存文件
        for temp_path in temp_image_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)