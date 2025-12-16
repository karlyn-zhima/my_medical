# -*- coding: utf-8 -*-
"""
Analysis Record Domain Model
分析记录领域模型
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import json


@dataclass
class LayerInfo:
    """层信息"""
    layer_name: str
    layer_description: str
    confidence: float
    features: Dict[str, Any]
    # 新增英文版本字段
    layer_name_en: Optional[str] = None
    layer_description_en: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'layer_name': self.layer_name,
            'layer_description': self.layer_description,
            'confidence': self.confidence,
            'features': self.features,
            'layer_name_en': self.layer_name_en,
            'layer_description_en': self.layer_description_en
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerInfo':
        return cls(
            layer_name=data['layer_name'],
            layer_description=data['layer_description'],
            confidence=data['confidence'],
            features=data['features'],
            layer_name_en=data.get('layer_name_en'),
            layer_description_en=data.get('layer_description_en')
        )


@dataclass
class SimilarImage:
    """相似图像信息"""
    image_path: str
    similarity_score: float
    metadata: Dict[str, Any]
    analysis_result: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'image_path': self.image_path,
            'similarity_score': self.similarity_score,
            'metadata': self.metadata,
            'analysis_result': self.analysis_result
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimilarImage':
        return cls(
            image_path=data['image_path'],
            similarity_score=data['similarity_score'],
            metadata=data['metadata'],
            analysis_result=data.get('analysis_result')
        )


@dataclass
class AnalysisRecord:
    """分析记录领域模型"""
    id: Optional[int] = None
    original_image_path: str = ""
    original_image_hash: str = ""
    similar_images: List[SimilarImage] = None
    detected_layers: List[LayerInfo] = None
    ai_analysis_result: Optional[str] = None
    status: str = "pending"  # pending, analyzing, completed, failed
    is_confirmed: int = -1  # -1: 未确认, 0: 医生拒绝, 1: 医生确认
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    user_id: Optional[int] = None
    
    def __post_init__(self):
        if self.similar_images is None:
            self.similar_images = []
        if self.detected_layers is None:
            self.detected_layers = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'original_image_path': self.original_image_path,
            'original_image_hash': self.original_image_hash,
            'similar_images': json.dumps([img.to_dict() for img in self.similar_images]),
            'detected_layers': json.dumps([layer.to_dict() for layer in self.detected_layers]),
            'ai_analysis_result': self.ai_analysis_result,
            'status': self.status,
            'is_confirmed': self.is_confirmed,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'user_id': self.user_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisRecord':
        """从字典创建实例"""
        similar_images = []
        if data.get('similar_images'):
            similar_images_data = json.loads(data['similar_images'])
            similar_images = [SimilarImage.from_dict(img) for img in similar_images_data]
        
        detected_layers = []
        if data.get('detected_layers'):
            detected_layers_data = json.loads(data['detected_layers'])
            detected_layers = [LayerInfo.from_dict(layer) for layer in detected_layers_data]
        
        return cls(
            id=data.get('id'),
            original_image_path=data.get('original_image_path', ''),
            original_image_hash=data.get('original_image_hash', ''),
            similar_images=similar_images,
            detected_layers=detected_layers,
            ai_analysis_result=data.get('ai_analysis_result'),
            status=data.get('status', 'pending'),
            is_confirmed=data.get('is_confirmed', -1),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            user_id=data.get('user_id')
        )
    
    def add_similar_image(self, image: SimilarImage):
        """添加相似图像"""
        self.similar_images.append(image)
        self.updated_at = datetime.now()
    
    def add_detected_layer(self, layer: LayerInfo):
        """添加检测到的层"""
        self.detected_layers.append(layer)
        self.updated_at = datetime.now()
    
    def update_status(self, status: str):
        """更新状态"""
        self.status = status
        self.updated_at = datetime.now()
    
    def set_analysis_result(self, result: str):
        """设置分析结果"""
        self.ai_analysis_result = result
        self.updated_at = datetime.now()
    
    def set_confirmation_status(self, is_confirmed: int):
        """设置确认状态"""
        self.is_confirmed = is_confirmed
        self.updated_at = datetime.now()