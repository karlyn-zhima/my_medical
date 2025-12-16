# -*- coding: utf-8 -*-
"""
Confirm Controller
确认相关控制器，提供未确认分析记录查询等接口
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
import os

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from domain.analysis_record import AnalysisRecord, LayerInfo
from domain.repository.analysis_record_repository import AnalysisRecordRepository
from infrastructure.mysql import getMedicalDBWithTableName
from infrastructure.auth.fastapi_auth import get_current_user, get_user_id_from_payload
from service.ai_analysis import AIAnalysisService
from service.translation import get_translation_service
from service.model_manager import get_model_manager


# 创建路由器
confirm_router = APIRouter(prefix="/confirm", tags=["确认管理"])


# 响应模型
class UnconfirmedAnalysisItem(BaseModel):
    id: int
    original_image_path: str
    thumbnail_base64: Optional[str] = None
    ai_analysis_result: Optional[str] = None
    detected_layers: List[dict] = []
    status: str
    is_confirmed: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class UnconfirmedAnalysisListResponse(BaseModel):
    success: bool
    records: List[UnconfirmedAnalysisItem] = []
    count: int
    error_message: Optional[str] = None


class SimilarImageDetail(BaseModel):
    image_path: str
    similarity_score: float
    metadata: Dict[str, Any]
    analysis_result: Optional[str] = None
    image_base64: Optional[str] = None


class AnalysisDetailResponse(BaseModel):
    success: bool
    id: Optional[int] = None
    original_image_path: Optional[str] = None
    original_image_base64: Optional[str] = None
    ai_analysis_result: Optional[str] = None
    detected_layers: List[dict] = []
    similar_images: List[SimilarImageDetail] = []
    status: Optional[str] = None
    is_confirmed: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    error_message: Optional[str] = None


class UpdateDetectedLayersRequest(BaseModel):
    """更新检测到的层信息的请求体（PUT版本，已保留）"""
    detected_layers: List[Dict[str, Any]] = []
    ai_analysis_result: Optional[str] = None


class UpdateLayersPostRequest(BaseModel):
    """更新检测到的层信息的请求体（POST版本，不修改摘要）"""
    record_id: int
    detected_layers: List[Dict[str, Any]] = []


class AddToRagByIdRequest(BaseModel):
    """根据记录ID将数据加入RAG索引的请求体"""
    record_id: int
    skip_duplicates: Optional[bool] = True
    label_dir: Optional[str] = None
    mask_dir: Optional[str] = None


class AddToRagByIdResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    added_count: Optional[int] = None
    total_count: Optional[int] = None
    # 新增：奖励token信息
    added_tokens: Optional[int] = None
    rest_token_count: Optional[float] = None
    error_message: Optional[str] = None


# 全局仓储实例
analysis_repository = None


def get_analysis_repository():
    """获取分析记录仓储实例"""
    global analysis_repository
    if analysis_repository is None:
        db = getMedicalDBWithTableName("analysis_records")._database
        analysis_repository = AnalysisRecordRepository(db)
        analysis_repository.create_table_if_not_exists()
    return analysis_repository


# ----------------------
# 辅助方法：从层信息生成RAG标签JSON结构
# ----------------------
def _labels_from_detected_layers(layers: List[LayerInfo]) -> List[Dict[str, Any]]:
    """将分析记录中的 LayerInfo 列表转换为标签数组结构。
    输出示例：
    [
      {"layer": "皮肤", "exists": True, "location": "上部", "ultrasound_features": "..."},
      ...
    ]
    """
    results: List[Dict[str, Any]] = []
    for layer in layers or []:
        exists = True
        desc = layer.layer_description or ""
        # 简单规则：若描述包含“未检测到”，标记为不存在
        if "未检测到" in desc:
            exists = False
        # 从 features 字典中提取位置与特征；如无则尝试从描述中粗略解析
        features_dict = layer.features or {}
        location_val = features_dict.get('location') or ""
        features_text = None
        try:
            if features_dict:
                # 合成为“k: v, k2: v2”的字符串
                features_text = ", ".join([f"{k}: {v}" for k, v in features_dict.items() if v is not None])
        except Exception:
            features_text = None

        if not location_val:
            # 粗略解析“位置: xxx”或“位置：xxx”
            import re
            m = re.search(r"位置[:：]\s*([^,，\n]+)", desc)
            if m:
                location_val = m.group(1).strip()

        if not features_text:
            # 粗略解析“特征: xxx”或“特征：xxx”
            import re
            m = re.search(r"特征[:：]\s*([^\n]+)", desc)
            if m:
                features_text = m.group(1).strip()
        if not features_text:
            features_text = ""

        results.append({
            "layer": layer.layer_name or "",
            "exists": bool(exists),
            "location": location_val,
            "ultrasound_features": features_text
        })
    return results


def _ensure_dirs(root: str) -> Dict[str, str]:
    """确保RAG目标目录结构存在，返回子目录路径。"""
    import os
    images_dir = os.path.join(root, 'images')
    masks_dir = os.path.join(root, 'masks')
    labels_dir = os.path.join(root, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    return {"images": images_dir, "masks": masks_dir, "labels": labels_dir}


def _copy_image_to_dir(src_path: str, dest_dir: str, preferred_base: str = None) -> str:
    """将原始图像复制到目标目录，返回复制后的绝对路径。"""
    import os, shutil
    base_name = preferred_base or os.path.splitext(os.path.basename(src_path))[0]
    ext = os.path.splitext(src_path)[1] or '.png'
    dest_path = os.path.abspath(os.path.join(dest_dir, f"{base_name}{ext}"))
    shutil.copy2(src_path, dest_path)
    return dest_path


def _save_json(data: Union[Dict[str, Any], List[Any]], dest_path: str) -> None:
    """保存JSON到文件（UTF-8, 缩进）"""
    import json, os
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@confirm_router.get("/unconfirmed", response_model=UnconfirmedAnalysisListResponse)
async def get_unconfirmed_analysis_results(
    limit: int = 50,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    获取当前用户调用AI分析的全部未确认分析结果

    Args:
        limit: 返回数量上限
        offset: 分页偏移量
        current_user: 认证用户信息

    Returns:
        UnconfirmedAnalysisListResponse: 未确认分析记录列表
    """
    try:
        user_id = int(get_user_id_from_payload(current_user))
        repo = get_analysis_repository()
        records = repo.find_unconfirmed_by_user(user_id, limit=limit, offset=offset)

        def _to_item(record: AnalysisRecord) -> UnconfirmedAnalysisItem:
            thumb_b64 = _encode_image_to_thumbnail_base64(record.original_image_path)
            recomposed_summary = _compose_summary_without_confidence(record.detected_layers or [])
            return UnconfirmedAnalysisItem(
                id=record.id or 0,
                original_image_path=record.original_image_path,
                thumbnail_base64=thumb_b64,
                ai_analysis_result=recomposed_summary,
                detected_layers=[layer.to_dict() for layer in (record.detected_layers or [])],
                status=record.status,
                is_confirmed=record.is_confirmed,
                created_at=record.created_at,
                updated_at=record.updated_at
            )

        items = [_to_item(r) for r in records]
        return UnconfirmedAnalysisListResponse(success=True, records=items, count=len(items))
    except Exception as e:
        return UnconfirmedAnalysisListResponse(success=False, records=[], count=0, error_message=str(e))


def _encode_image_to_base64(image_path: str) -> Optional[str]:
    """将图像文件编码为base64字符串，如果文件不存在则返回None"""
    try:
        if not image_path:
            return None
        with open(image_path, "rb") as f:
            import base64
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

def _encode_image_to_thumbnail_base64(image_path: str, max_size: int = 128) -> Optional[str]:
    """将图像生成缩略图并编码为base64，如果失败则回退为原图base64。

    优化点：
    - 默认最大边长缩小为128，减小图像体积；
    - 采用JPEG有损压缩（quality≈70，optimize=True），相比PNG显著降低大小；
    - 统一转换到RGB，处理含透明通道的图片以避免保存JPEG失败。
    """
    try:
        if not image_path:
            return None
        from io import BytesIO
        from PIL import Image
        import base64
        with Image.open(image_path) as img:
            # 生成缩略图，保持比例
            img.thumbnail((max_size, max_size))

            # 处理透明通道：将RGBA/LA转为白底RGB，其他模式统一转RGB
            try:
                if img.mode in ("RGBA", "LA"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert("RGB")
            except Exception:
                # 转换失败时，继续尝试原图保存
                img = img.convert("RGB")

            buf = BytesIO()
            # 使用JPEG压缩以显著减小体积
            img.save(buf, format="JPEG", quality=70, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        # 缩略图生成失败时，回退为原图base64
        return _encode_image_to_base64(image_path)

def _compose_summary_without_confidence(layers: List[LayerInfo]) -> str:
    """根据detected_layers拼接AI描述结果，不包含置信度信息"""
    if not layers:
        return "未检测到明显的层结构"
    summary = f"检测到 {len(layers)} 个可能的解剖层结构：\n\n"
    for i, layer in enumerate(layers, 1):
        summary += f"{i}. {layer.layer_name}\n"
        summary += f"   描述: {layer.layer_description}\n"
        try:
            if layer.features:
                features_desc = ', '.join([f"{k}: {v}" for k, v in layer.features.items()])
                summary += f"   特征: {features_desc}\n"
        except Exception:
            pass
        summary += "\n"
    return summary


@confirm_router.get("/detail/{record_id}", response_model=AnalysisDetailResponse)
async def get_analysis_detail(
    record_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    获取指定分析记录的详细信息，包括原始图像的base64编码

    Args:
        record_id: 分析记录ID
        current_user: 认证用户信息

    Returns:
        AnalysisDetailResponse: 分析记录详情
    """
    try:
        user_id = int(get_user_id_from_payload(current_user))
        repo = get_analysis_repository()
        record = repo.find_by_id(record_id)

        if record is None:
            return AnalysisDetailResponse(success=False, error_message="记录不存在")

        # 权限校验：仅允许访问自己的记录
        if record.user_id is not None and int(record.user_id) != user_id:
            return AnalysisDetailResponse(success=False, error_message="无权访问该记录")

        original_image_b64 = _encode_image_to_base64(record.original_image_path)

        # 处理相似图像（如有），附带可用的base64
        similar_details: List[SimilarImageDetail] = []
        for img in (record.similar_images or []):
            similar_details.append(
                SimilarImageDetail(
                    image_path=img.image_path,
                    similarity_score=img.similarity_score,
                    metadata=img.metadata,
                    analysis_result=img.analysis_result,
                    image_base64=_encode_image_to_base64(img.image_path)
                )
            )

        return AnalysisDetailResponse(
            success=True,
            id=record.id,
            original_image_path=record.original_image_path,
            original_image_base64=original_image_b64,
            ai_analysis_result=record.ai_analysis_result,
            detected_layers=[layer.to_dict() for layer in (record.detected_layers or [])],
            similar_images=similar_details,
            status=record.status,
            is_confirmed=record.is_confirmed,
            created_at=record.created_at,
            updated_at=record.updated_at
        )
    except Exception as e:
        return AnalysisDetailResponse(success=False, error_message=str(e))


@confirm_router.post("/add-to-rag", response_model=AddToRagByIdResponse)
async def add_record_to_rag(
    request: AddToRagByIdRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    根据记录ID将该记录的原始图像加入到RAG索引。

    要求：仅允许当前用户操作自己的记录；支持可选的标签与掩码目录；
    元数据结构参考 faiss_meta.json（包含 id、image_path、mask_path、label_path、labels）。
    """
    try:
        # 1) 权限与记录校验
        user_id = int(get_user_id_from_payload(current_user))
        repo = get_analysis_repository()
        record = repo.find_by_id(request.record_id)

        if record is None:
            return AddToRagByIdResponse(success=False, error_message="记录不存在")
        if record.user_id is not None and int(record.user_id) != user_id:
            return AddToRagByIdResponse(success=False, error_message="无权修改该记录")
        # 已完成确认则禁止重复确认
        if int(record.is_confirmed or 0) == 1:
            return AddToRagByIdResponse(success=False, error_message="该记录已完成确认，禁止重复确认")

        # 2) 目标RAG目录：从配置文件读取相对路径（默认 'rag_data'）
        try:
            import json
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            config_path = os.path.join(base_dir, 'config', 'config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            rag_rel = (
                (cfg.get('paths') or {}).get('rag_data_root')
                or cfg.get('rag_data_root')
                or 'rag_data'
            )
            rag_root = os.path.abspath(os.path.join(base_dir, rag_rel))
        except Exception:
            # 兜底：若配置不可用，使用项目根下的 'rag_data'
            rag_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rag_data'))
        dirs = _ensure_dirs(rag_root)

        # 3) 复制原图到 rag_data/images
        preferred_base = f"rec_{record.id}" if record.id else None
        copied_img_path = _copy_image_to_dir(record.original_image_path, dirs["images"], preferred_base=preferred_base)

        # 4) 生成 labels JSON（来自数据库的 detected_layers）并保存到 rag_data/labels
        labels = _labels_from_detected_layers(record.detected_layers or [])
        label_path = None
        if labels:
            label_filename = (preferred_base or "image") + ".json"
            label_path = os.path.abspath(os.path.join(dirs["labels"], label_filename))
            _save_json(labels, label_path)

        # 5) 自动生成 mask 图像并保存到 rag_data/masks
        # 使用模型的 interface，并将描述列表取为层名称列表（存在的层）
        try:
            from service.model_manager import get_model_manager
            infer = get_model_manager().get_inference_model()
            # 使用数据库中存储的英文拼接来生成分割提示
            descriptions: List[str] = []
            for layer in (record.detected_layers or []):
                # 通过英文/中文描述判断该层是否存在
                desc_en = (getattr(layer, 'layer_description_en', None) or '').strip()
                desc_cn = (getattr(layer, 'layer_description', None) or '').strip()
                exists = True
                if desc_en and ('not detected' in desc_en.lower()):
                    exists = False
                elif ('未检测到' in desc_cn):
                    exists = False

                if not exists:
                    continue

                name_en = (getattr(layer, 'layer_name_en', None) or '').strip()
                # 组装英文提示：优先使用英文层名 + 英文描述
                phrase_parts = [name_en if name_en else '', desc_en if desc_en else '']
                phrase = ' '.join([p for p in phrase_parts if p])
                if not phrase:
                    # 兜底：使用中文层名
                    phrase = (getattr(layer, 'layer_name', None) or '').strip()
                if phrase:
                    descriptions.append(phrase)

            pred_mask = infer.interface(record.original_image_path, descriptions, visualize=True)
        except Exception as e:
            # 如果分割失败，允许继续，仅不生成mask
            pred_mask = None

        mask_path = None
        if pred_mask is not None:
            import cv2
            mask_filename = (preferred_base or "image") + ".png"
            mask_path = os.path.abspath(os.path.join(dirs["masks"], mask_filename))
            cv2.imwrite(mask_path, pred_mask)

        # 6) 调用FAISS服务写入索引与元数据（指向rag_data下的路径）
        service = get_model_manager().get_faiss_service()
        result = service.add_to_index(
            image_paths=[copied_img_path],
            skip_duplicates=bool(request.skip_duplicates),
            label_dir=dirs["labels"],
            mask_dir=dirs["masks"]
        )
        if not result.get("success"):
            return AddToRagByIdResponse(success=False, error_message=result.get("message"))

        # 7) 更新记录为已确认
        record.set_confirmation_status(1)
        repo.update_record(record)

        # 8) 为当前用户奖励Token（3000）并返回余额
        from decimal import Decimal
        try:
            from service.user import UserService
            award_tokens = Decimal('3000')
            user_service = UserService()
            award_result = user_service.add_tokens(user_id, award_tokens)
            rest_token_cnt: Optional[float] = None
            try:
                user_info = (award_result or {}).get('user') or {}
                raw_cnt = user_info.get('rest_token_count')
                if raw_cnt is not None:
                    # 兼容Decimal/字符串
                    rest_token_cnt = float(str(raw_cnt))
            except Exception:
                rest_token_cnt = None
        except Exception:
            # 奖励失败不影响主要流程
            award_result = None
            rest_token_cnt = None

        return AddToRagByIdResponse(
            success=True,
            message="加入RAG并生成与迁移完成",
            added_count=result.get("added_count"),
            total_count=result.get("total_count"),
            added_tokens=3000,
            rest_token_count=rest_token_cnt
        )
    except Exception as e:
        return AddToRagByIdResponse(success=False, error_message=str(e))


@confirm_router.put("/detail/{record_id}/layers", response_model=AnalysisDetailResponse)
async def update_detected_layers(
    record_id: int,
    request: UpdateDetectedLayersRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    更新指定分析记录的 detected_layers 信息及（可选）摘要描述。

    Args:
        record_id: 分析记录ID
        request: 包含新的层信息列表与可选的AI摘要
        current_user: 认证用户信息

    Returns:
        AnalysisDetailResponse: 更新后的分析记录详情
    """
    try:
        user_id = int(get_user_id_from_payload(current_user))
        repo = get_analysis_repository()
        record = repo.find_by_id(record_id)

        if record is None:
            return AnalysisDetailResponse(success=False, error_message="记录不存在")

        # 权限校验：仅允许修改自己的记录
        if record.user_id is not None and int(record.user_id) != user_id:
            return AnalysisDetailResponse(success=False, error_message="无权修改该记录")
        # 已完成确认则禁止保存
        if int(record.is_confirmed or 0) == 1:
            return AnalysisDetailResponse(success=False, error_message="该记录已完成确认，无法保存层信息")

        # 将请求体中的字典列表转换为 LayerInfo 对象列表
        # 先构建上一版本层的映射，以便复用英文翻译
        prev_layers: List[LayerInfo] = record.detected_layers or []
        prev_by_name: Dict[str, LayerInfo] = {}
        prev_by_pair: Dict[tuple[str, str], LayerInfo] = {}
        for pl in prev_layers:
            key_name = (pl.layer_name or '').strip()
            key_pair = ((pl.layer_name or '').strip(), (pl.layer_description or '').strip())
            if key_name and key_name not in prev_by_name:
                prev_by_name[key_name] = pl
            if key_pair not in prev_by_pair:
                prev_by_pair[key_pair] = pl

        def _contains_chinese(s: str) -> bool:
            s = (s or '').strip()
            return any('\u4e00' <= ch <= '\u9fff' for ch in s)

        new_layers: List[LayerInfo] = []
        for item in (request.detected_layers or []):
            try:
                # 与 LayerInfo.to_dict 对齐的键集合，允许缺省值
                name_cn = item.get('layer_name', '')
                desc_cn = item.get('layer_description', '')

                # 若英文字段缺失，调用后端翻译服务补齐
                ts = get_translation_service()
                # 优先复用旧记录中相同中文内容的英文翻译，避免重复翻译
                name_en = (item.get('layer_name_en') or '').strip()
                desc_en = (item.get('layer_description_en') or '').strip()

                # 复用策略：先按(中文名, 中文描述)精确匹配，再按中文名匹配
                pair_key = (name_cn.strip(), desc_cn.strip())
                prev_pair = prev_by_pair.get(pair_key)
                prev_name = prev_by_name.get(name_cn.strip())

                if not name_en:
                    candidate = None
                    if prev_pair and (prev_pair.layer_name_en or '').strip():
                        candidate = prev_pair.layer_name_en.strip()
                    elif prev_name and (prev_name.layer_name_en or '').strip():
                        candidate = prev_name.layer_name_en.strip()
                    # 仅当候选不存在中文且不与中文相同，才复用
                    if candidate and not _contains_chinese(candidate) and candidate != name_cn.strip():
                        name_en = candidate
                    else:
                        name_en = ts.translate_text(name_cn, source='zh', target='en').text

                if not desc_en:
                    candidate = None
                    if prev_pair and (prev_pair.layer_description_en or '').strip():
                        candidate = prev_pair.layer_description_en.strip()
                    elif prev_name and (prev_name.layer_description_en or '').strip():
                        prev_desc_cn = (prev_name.layer_description or '').strip()
                        if prev_desc_cn == desc_cn.strip() and (prev_name.layer_description_en or '').strip():
                            candidate = prev_name.layer_description_en.strip()
                    # 仅当候选不存在中文且不与中文相同，才复用
                    if candidate and not _contains_chinese(candidate) and candidate != desc_cn.strip():
                        desc_en = candidate
                    else:
                        desc_en = ts.translate_text(desc_cn, source='zh', target='en').text

                # 保存时在服务端打印英文翻译
                try:
                    print(f"[Translation][PUT] record_id={record_id} | '{name_cn}' -> '{name_en}' | desc: '{desc_cn}' -> '{desc_en}'")
                except Exception:
                    pass

                layer = LayerInfo(
                    layer_name=name_cn,
                    layer_description=desc_cn,
                    confidence=float(item.get('confidence', 0.0)),
                    features=item.get('features', {}) or {},
                    layer_name_en=name_en,
                    layer_description_en=desc_en
                )
                new_layers.append(layer)
            except Exception:
                # 跳过非法项，确保接口健壮性
                continue

        record.detected_layers = new_layers

        # 更新（或重新生成）摘要
        if request.ai_analysis_result is not None:
            record.ai_analysis_result = request.ai_analysis_result
        else:
            ai_service = AIAnalysisService()
            record.ai_analysis_result = ai_service.generate_layer_summary(new_layers)

        record.updated_at = datetime.now()

        # 持久化更新
        repo.update_record(record)

        # 返回更新后的详情
        original_image_b64 = _encode_image_to_base64(record.original_image_path)
        similar_details: List[SimilarImageDetail] = []
        for img in (record.similar_images or []):
            similar_details.append(
                SimilarImageDetail(
                    image_path=img.image_path,
                    similarity_score=img.similarity_score,
                    metadata=img.metadata,
                    analysis_result=img.analysis_result,
                    image_base64=_encode_image_to_base64(img.image_path)
                )
            )

        return AnalysisDetailResponse(
            success=True,
            id=record.id,
            original_image_path=record.original_image_path,
            original_image_base64=original_image_b64,
            ai_analysis_result=record.ai_analysis_result,
            detected_layers=[layer.to_dict() for layer in (record.detected_layers or [])],
            similar_images=similar_details,
            status=record.status,
            is_confirmed=record.is_confirmed,
            created_at=record.created_at,
            updated_at=record.updated_at
        )
    except Exception as e:
        return AnalysisDetailResponse(success=False, error_message=str(e))


@confirm_router.post("/update-layers", response_model=AnalysisDetailResponse)
async def update_detected_layers_post(
    request: UpdateLayersPostRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    使用POST方法更新指定分析记录的 detected_layers 信息。
    要求：仅允许修改 user_id 为当前用户的记录；不修改 ai_analysis_result。
    """
    try:
        user_id = int(get_user_id_from_payload(current_user))
        repo = get_analysis_repository()
        record = repo.find_by_id(request.record_id)

        if record is None:
            return AnalysisDetailResponse(success=False, error_message="记录不存在")

        # 权限校验：仅允许修改自己的记录
        if record.user_id is not None and int(record.user_id) != user_id:
            return AnalysisDetailResponse(success=False, error_message="无权修改该记录")
        # 已完成确认则禁止保存
        if int(record.is_confirmed or 0) == 1:
            return AnalysisDetailResponse(success=False, error_message="该记录已完成确认，无法保存层信息")

        # 转换 detected_layers
        # 先构建上一版本层的映射，以便复用英文翻译
        prev_layers: List[LayerInfo] = record.detected_layers or []
        prev_by_name: Dict[str, LayerInfo] = {}
        prev_by_pair: Dict[tuple[str, str], LayerInfo] = {}
        for pl in prev_layers:
            key_name = (pl.layer_name or '').strip()
            key_pair = ((pl.layer_name or '').strip(), (pl.layer_description or '').strip())
            if key_name and key_name not in prev_by_name:
                prev_by_name[key_name] = pl
            if key_pair not in prev_by_pair:
                prev_by_pair[key_pair] = pl

        def _contains_chinese(s: str) -> bool:
            s = (s or '').strip()
            return any('\u4e00' <= ch <= '\u9fff' for ch in s)

        new_layers: List[LayerInfo] = []
        for item in (request.detected_layers or []):
            try:
                name_cn = item.get('layer_name', '')
                desc_cn = item.get('layer_description', '')

                ts = get_translation_service()
                # 优先复用旧记录中相同中文内容的英文翻译，避免重复翻译
                name_en = (item.get('layer_name_en') or '').strip()
                desc_en = (item.get('layer_description_en') or '').strip()

                pair_key = (name_cn.strip(), desc_cn.strip())
                prev_pair = prev_by_pair.get(pair_key)
                prev_name = prev_by_name.get(name_cn.strip())

                if not name_en:
                    candidate = None
                    if prev_pair and (prev_pair.layer_name_en or '').strip():
                        candidate = prev_pair.layer_name_en.strip()
                    elif prev_name and (prev_name.layer_name_en or '').strip():
                        candidate = prev_name.layer_name_en.strip()
                    if candidate and not _contains_chinese(candidate) and candidate != name_cn.strip():
                        name_en = candidate
                    else:
                        name_en = ts.translate_text(name_cn, source='zh', target='en').text

                if not desc_en:
                    candidate = None
                    if prev_pair and (prev_pair.layer_description_en or '').strip():
                        candidate = prev_pair.layer_description_en.strip()
                    elif prev_name and (prev_name.layer_description_en or '').strip():
                        prev_desc_cn = (prev_name.layer_description or '').strip()
                        if prev_desc_cn == desc_cn.strip() and (prev_name.layer_description_en or '').strip():
                            candidate = prev_name.layer_description_en.strip()
                    if candidate and not _contains_chinese(candidate) and candidate != desc_cn.strip():
                        desc_en = candidate
                    else:
                        desc_en = ts.translate_text(desc_cn, source='zh', target='en').text

                # 保存时在服务端打印英文翻译
                try:
                    print(f"[Translation][POST] record_id={request.record_id} | '{name_cn}' -> '{name_en}' | desc: '{desc_cn}' -> '{desc_en}'")
                except Exception:
                    pass

                layer = LayerInfo(
                    layer_name=name_cn,
                    layer_description=desc_cn,
                    confidence=float(item.get('confidence', 0.0)),
                    features=item.get('features', {}) or {},
                    layer_name_en=name_en,
                    layer_description_en=desc_en
                )
                new_layers.append(layer)
            except Exception:
                continue

        # 仅更新 detected_layers 与更新时间，不触碰摘要
        record.detected_layers = new_layers
        record.updated_at = datetime.now()
        repo.update_record(record)

        # 返回详情
        original_image_b64 = _encode_image_to_base64(record.original_image_path)
        similar_details: List[SimilarImageDetail] = []
        for img in (record.similar_images or []):
            similar_details.append(
                SimilarImageDetail(
                    image_path=img.image_path,
                    similarity_score=img.similarity_score,
                    metadata=img.metadata,
                    analysis_result=img.analysis_result,
                    image_base64=_encode_image_to_base64(img.image_path)
                )
            )

        return AnalysisDetailResponse(
            success=True,
            id=record.id,
            original_image_path=record.original_image_path,
            original_image_base64=original_image_b64,
            ai_analysis_result=record.ai_analysis_result,  # 保持不变
            detected_layers=[layer.to_dict() for layer in (record.detected_layers or [])],
            similar_images=similar_details,
            status=record.status,
            is_confirmed=record.is_confirmed,
            created_at=record.created_at,
            updated_at=record.updated_at
        )
    except Exception as e:
        return AnalysisDetailResponse(success=False, error_message=str(e))