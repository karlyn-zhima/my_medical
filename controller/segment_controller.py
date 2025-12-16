# -*- coding: utf-8 -*-
"""
Segment Controller
图像分割控制器，处理图像分割相关的API请求
"""

import os
import uuid
import cv2
import io
import hashlib
from decimal import Decimal
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
import asyncio
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from constants.error_codes import ErrorCode
from constants.error_messages import get_error_message
from service.ai_analysis import AIAnalysisService, AIAnalysisConfig
from service.model_manager import get_model_manager
from service.user import UserService
from domain.segment_record import SegmentRecord
from domain.analysis_record import AnalysisRecord, SimilarImage, LayerInfo
from domain.repository.segment_record_repository import SegmentRecordRepository
from domain.repository.analysis_record_repository import AnalysisRecordRepository
from infrastructure.mysql import getMedicalDBWithTableName
from infrastructure.auth.fastapi_auth import get_current_user, get_user_id_from_payload

# 创建路由器
segment_router = APIRouter(prefix="/segment", tags=["图像分割"])

# 响应模型
class AnalysisResponse(BaseModel):
    success: bool
    record_id: Optional[int] = None
    detected_layers: List[dict] = []
    ai_analysis_summary: Optional[str] = None
    error_message: Optional[str] = None

class LayerSelectionRequest(BaseModel):
    analysis_record_id: int
    selected_layers: List[str]

class SegmentResponse(BaseModel):
    success: bool
    record_id: Optional[int] = None
    segment_result_path: Optional[str] = None
    error_message: Optional[str] = None


# 全局服务实例
inference_model = None
faiss_service = None
ai_analysis_service = None
segment_repository = None
analysis_repository = None
user_service = None

def get_inference_model():
    """获取推理模型实例"""
    model_manager = get_model_manager()
    return model_manager.get_inference_model()

def get_faiss_service():
    """获取FAISS服务实例"""
    model_manager = get_model_manager()
    return model_manager.get_faiss_service()

def get_ai_analysis_service():
    """获取AI分析服务实例"""
    global ai_analysis_service
    if ai_analysis_service is None:
        config = AIAnalysisConfig()
        ai_analysis_service = AIAnalysisService(config)
    return ai_analysis_service

def get_analysis_repository():
    """获取分析记录仓储实例"""
    global analysis_repository
    if analysis_repository is None:
        db = getMedicalDBWithTableName("analysis_records")._database
        analysis_repository = AnalysisRecordRepository(db)
        # 确保表存在
        analysis_repository.create_table_if_not_exists()
    return analysis_repository

def get_segment_repository():
    """获取分割记录仓储实例"""
    global segment_repository
    if segment_repository is None:
        db = getMedicalDBWithTableName("segment_records")._database
        segment_repository = SegmentRecordRepository(db)
        # 确保表存在
        segment_repository.create_table_if_not_exists()
    return segment_repository


def get_user_service():
    """获取用户服务实例"""
    global user_service
    if user_service is None:
        user_service = UserService()
    return user_service

def calculate_image_hash(image_path: str) -> str:
    """计算图像的哈希值"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
        return hashlib.sha256(image_data).hexdigest()

def _generate_analysis_result_from_labels(metadata: dict) -> str:
    """从metadata中的labels生成analysis_result文本"""
    labels = metadata.get('labels', [])
    if not labels:
        return "无可用的分析结果"
    
    analysis_parts = []
    for label in labels:
        layer = label.get('layer', '未知层')
        exists = label.get('exists', False)
        location = label.get('location', '未知位置')
        features = label.get('ultrasound_features', '无特征描述')
        
        if exists:
            analysis_parts.append(f"{layer}：存在于{location}，超声特征为{features}")
        else:
            analysis_parts.append(f"{layer}：未检测到")
    
    return "；".join(analysis_parts)

def generate_image_filename(user_id: int, image_hash: str) -> str:
    """生成图片文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"user_{user_id}_{timestamp}_{image_hash[:8]}.png"

def save_uploaded_image(image_data: bytes, user_id: int, image_hash: str) -> str:
    """保存上传的图片到固定存储路径"""
    # 确保上传目录存在
    upload_dir = os.path.join("uploads", "images")
    os.makedirs(upload_dir, exist_ok=True)
    
    # 生成文件名
    filename = generate_image_filename(user_id, image_hash)
    file_path = os.path.join(upload_dir, filename)
    
    # 保存文件
    with open(file_path, "wb") as f:
        f.write(image_data)
    
    return file_path

def resize_to_original(pred_mask, original_image_path: str):
    """将分割输出调整为与原图一致的尺寸。
    使用最近邻插值以避免类别标签被平滑。"""
    try:
        original = cv2.imread(original_image_path)
        if original is None:
            # 原图不可读则保持原样
            return pred_mask
        h, w = original.shape[:2]
        mh, mw = pred_mask.shape[:2]
        if (h != mh) or (w != mw):
            return cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return pred_mask
    except Exception as e:
        # 出错时不影响主流程，打印告警并返回原始分割结果
        print(f"[segment] resize_to_original warning: {e}")
        return pred_mask

def set_inference_model(model):
    """设置推理模型实例"""
    # 使用模型管理器重置模型
    model_manager = get_model_manager()
    if hasattr(model, '_model_path'):
        model_manager.reset_model(model._model_path)
    else:
        # 如果传入的是路径字符串
        if isinstance(model, str):
            model_manager.reset_model(model)

def set_faiss_service(service):
    """设置FAISS服务实例"""
    # 使用模型管理器，不需要手动设置FAISS服务
    # FAISS服务会在模型重置后自动重新初始化
    pass

def set_ai_analysis_service(service):
    """设置AI分析服务实例"""
    global ai_analysis_service
    ai_analysis_service = service

@segment_router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    image: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    分析图像，检索相似图像并识别层信息
    
    Args:
        image: 上传的图像文件
        current_user: 从JWT token解析的当前用户信息
        
    Returns:
        AnalysisResponse: 分析结果
    """
    # 从JWT token中获取用户ID
    user_id = int(get_user_id_from_payload(current_user))
    
    # 检查用户是否存在
    user_service = get_user_service()
    try:
        user_result = user_service.get_user_by_id(user_id)
        if not user_result or not user_result.get('success'):
            return AnalysisResponse(
                success=False,
                error_message="用户不存在"
            )
        
        # 从结果中获取用户信息字典
        user_dict = user_result.get('user', {})
        
    except Exception as e:
        return AnalysisResponse(
            success=False,
            error_message=f"用户验证失败: {str(e)}"
        )
    
    try:
        # 读取图片数据并计算哈希
        image_data = await image.read()
        image_hash = hashlib.sha256(image_data).hexdigest()
        
        # 保存图片到固定存储文件夹
        image_path = save_uploaded_image(image_data, user_id, image_hash)
        
        # 检查是否已经分析过这张图像
        analysis_repository = get_analysis_repository()
        existing_record = analysis_repository.find_by_image_hash(image_hash)
        
        if existing_record and existing_record.status == "completed":
            # 返回已有的分析结果
            return AnalysisResponse(
                success=True,
                record_id=existing_record.id,
                detected_layers=[layer.to_dict() for layer in existing_record.detected_layers],
                ai_analysis_summary=existing_record.ai_analysis_result
            )
        
        # 创建新的分析记录
        record = AnalysisRecord(
            original_image_path=image_path,
            original_image_hash=image_hash,
            user_id=user_id,
            status="analyzing",
            is_confirmed=-1  # 初次调用大模型分析时设置为未确认状态
        )
        
        # 保存记录到数据库
        record_id = analysis_repository.save_record(record)
        record.id = record_id
        
        # 1. 使用FAISS检索相似图像（阻塞操作，放入线程池以避免阻塞事件循环）
        faiss_service = get_faiss_service()
        loop = asyncio.get_running_loop()
        similar_results = await loop.run_in_executor(None, lambda: faiss_service.search_faiss(image_path, k=5))
        
        # 转换为SimilarImage对象
        similar_images = []
        for result in similar_results:
            # 从result中直接提取labels生成analysis_result
            labels = result.get('labels', [])
            analysis_result = _generate_analysis_result_from_labels({'labels': labels})
            
            similar_image = SimilarImage(
                image_path=result.get('image_path', ''),  # 从FAISS结果中获取image_path
                similarity_score=result.get('score', 0.0),
                metadata={'labels': labels},
                analysis_result=analysis_result
            )
            similar_images.append(similar_image)
            record.add_similar_image(similar_image)
        
        # 检查用户token余额是否为负数，如果是负数则不允许执行
        rest_token_count = user_dict.get('rest_token_count', 0)
        if rest_token_count < 0:
            return AnalysisResponse(
                success=False,
                error_message=f"Token余额为负数({rest_token_count})，请充值后再使用"
            )
        
        # 2. 使用AI分析层信息（调用大模型API，阻塞操作，放入线程池以避免阻塞其他接口）
        ai_service = get_ai_analysis_service()
        detected_layers, token_usage = await loop.run_in_executor(
            None,
            lambda: ai_service.analyze_ultrasound_layers(image_path, similar_images)
        )

        # 根据实际token消耗扣除用户token
        total_tokens_consumed = token_usage.get('total_tokens', 0)

        print("use token count:"+str(total_tokens_consumed))

        if total_tokens_consumed > 0:
            # 直接扣除实际消耗的token，允许余额变成负数（允许用户拖欠一次查询）
            user_service.consume_tokens(user_id, Decimal(str(total_tokens_consumed)))
        
        # 添加检测到的层信息
        for layer in detected_layers:
            record.add_detected_layer(layer)
        
        # 3. 生成AI分析摘要
        ai_summary = ai_service.generate_layer_summary(detected_layers)
        record.ai_analysis_result = ai_summary
        record.update_status("completed")
        
        # 更新数据库记录
        analysis_repository.update_record(record)
        
        return AnalysisResponse(
            success=True,
            record_id=record.id,
            detected_layers=[layer.to_dict() for layer in detected_layers],
            ai_analysis_summary=ai_summary
        )
        
    except Exception as e:
        # 只有在AI分析失败且已经扣除token的情况下才回滚token
        # 检查是否是AI分析阶段的失败（detected_layers未定义说明AI分析失败）
        ai_analysis_failed = 'detected_layers' not in locals()
        
        if ai_analysis_failed and 'total_tokens_consumed' in locals() and total_tokens_consumed > 0:
            try:
                user_service.add_tokens(user_id, Decimal(str(total_tokens_consumed)))
                print(f"AI分析失败，已回滚token: {total_tokens_consumed}")
            except Exception as rollback_error:
                # 记录回滚失败的日志，但不影响主要错误的返回
                print(f"Token回滚失败: {str(rollback_error)}")
        elif not ai_analysis_failed:
            # 如果AI分析成功但后续步骤失败，不回滚token，因为AI服务已经消耗了资源
            print(f"AI分析成功但后续步骤失败，不回滚token: {total_tokens_consumed}")
        
        # 更新记录状态为失败
        if 'record' in locals() and record.id:
            analysis_repository = get_analysis_repository()
            analysis_repository.update_status(record.id, "failed")
        
        return AnalysisResponse(
            success=False,
            error_message=f"图像分析失败: {str(e)}"
        )
    
    finally:
        # 不再删除图片文件，因为已经保存到固定存储路径
        pass

@segment_router.post("/segment-with-layers", response_model=SegmentResponse)
async def segment_with_selected_layers(request: LayerSelectionRequest):
    """
    根据用户选择的层进行图像分割
    
    Args:
        request: 包含分析记录ID和选择层的请求
        
    Returns:
        SegmentResponse: 分割结果
    """
    try:
        # 获取分析记录
        analysis_repository = get_analysis_repository()
        analysis_record = analysis_repository.find_by_id(request.analysis_record_id)
        
        if not analysis_record:
            return SegmentResponse(
                success=False,
                error_message="未找到指定的分析记录"
            )
        
        # 创建分割记录
        segment_repository = get_segment_repository()
        segment_record = SegmentRecord(
            original_image_path=analysis_record.original_image_path,
            original_image_hash=analysis_record.original_image_hash,
            selected_layers=request.selected_layers,
            user_id=analysis_record.user_id,
            analysis_record_id=analysis_record.id,
            status="segmenting"
        )
        
        # 保存分割记录
        segment_record_id = segment_repository.save_record(segment_record)
        segment_record.id = segment_record_id
        
        # 执行分割
        model = get_inference_model()
        pred_mask = model.interface(analysis_record.original_image_path, request.selected_layers, visualize=True)
        # 保证与原图同尺寸
        pred_mask = resize_to_original(pred_mask, analysis_record.original_image_path)
        
        # 保存分割结果
        result_filename = f"segment_result_{segment_record.id}_{analysis_record.original_image_hash[:8]}.png"
        result_path = os.path.join("uploads", "results", result_filename)
        
        # 确保结果目录存在
        os.makedirs(os.path.join("uploads", "results"), exist_ok=True)
        
        # 保存分割结果图像
        cv2.imwrite(result_path, pred_mask)
        
        # 更新分割记录
        segment_record.segment_result_path = result_path
        segment_record.status = "segmented"
        segment_repository.update_record(segment_record)
        
        return SegmentResponse(
            success=True,
            record_id=segment_record.id,
            segment_result_path=result_path
        )
        
    except Exception as e:
        return SegmentResponse(
            success=False,
            error_message=f"分割失败: {str(e)}"
        )

 

@segment_router.get("/result/{record_id}")
async def get_segment_result(record_id: int):
    """
    获取分割结果图像
    
    Args:
        record_id: 记录ID
        
    Returns:
        StreamingResponse: 分割结果图像流
    """
    try:
        repository = get_segment_repository()
        record = repository.find_by_id(record_id)
        
        if not record or not record.segment_result_path:
            raise HTTPException(
                status_code=404,
                detail="未找到分割结果"
            )
        
        if not os.path.exists(record.segment_result_path):
            raise HTTPException(
                status_code=404,
                detail="分割结果文件不存在"
            )
        
        # 读取图像文件
        with open(record.segment_result_path, "rb") as f:
            image_data = f.read()
        
        return StreamingResponse(
            io.BytesIO(image_data),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=segment_result_{record_id}.png"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取分割结果失败: {str(e)}"
        )
@segment_router.post("/")
async def segment_image(
    image: UploadFile = File(...),
    descriptions: str = Form(...),
):
    """
    接收图像作为参数，返回分割结果（原有接口，保持兼容性）
    
    Args:
        image: 上传的图像文件
        descriptions: 分割描述数组的JSON字符串
        
    Returns:
        StreamingResponse: 分割后的图像流
    """
    # 计算图片哈希
    image_data = await image.read()
    image_hash = hashlib.md5(image_data).hexdigest()
    
    # 保存图片到固定存储路径
    saved_image_path = save_uploaded_image(image_data, 1, image_hash)  # 使用默认用户ID 1
    
    try:
        # 解析descriptions JSON字符串为数组
        import json
        try:
            descriptions_list = json.loads(descriptions)
            if not isinstance(descriptions_list, list):
                raise ValueError("descriptions必须是数组格式")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail={
                    'success': False,
                    'message': f'descriptions参数格式错误: {str(e)}',
                    'code': ErrorCode.INVALID_PARAMETER
                }
            )

        # 获取推理模型并运行预测
        model = get_inference_model()
        pred_mask = model.interface(saved_image_path, descriptions_list, visualize=True)
        # 保证与原图同尺寸
        pred_mask = resize_to_original(pred_mask, saved_image_path)
        
        # 保存分割结果到固定路径
        result_filename = f"segment_result_{image_hash[:8]}.png"
        result_path = os.path.join("uploads", "results", result_filename)
        
        # 确保结果目录存在
        os.makedirs(os.path.join("uploads", "results"), exist_ok=True)
        
        # 保存分割结果图像到文件
        cv2.imwrite(result_path, pred_mask)
        
        # 创建分割记录（不需要关联分析记录，因为这是直接分割）
        segment_repository = get_segment_repository()
        record = SegmentRecord(
            original_image_path=saved_image_path,
            original_image_hash=image_hash,
            user_id=1,  # 使用默认用户ID
            status="segmented",
            segment_result_path=result_path,
            selected_layers=descriptions_list,
            analysis_record_id=None  # 直接分割不关联分析记录
        )
        segment_repository.save_record(record)
        
        # 编码预测结果用于返回
        success, encoded_image = cv2.imencode('.png', pred_mask)
        if not success:
            raise HTTPException(
                status_code=500, 
                detail={
                    'success': False,
                    'message': get_error_message(ErrorCode.SYSTEM_ERROR),
                    'code': ErrorCode.SYSTEM_ERROR
                }
            )
        
        # 转化为字节流
        image_bytes = encoded_image.tobytes()
        
        # 返回图像作为流式响应
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=segmented_image.png"}
        )
    
    finally:
        # 不再删除图片文件，因为已经保存到固定存储路径
        pass