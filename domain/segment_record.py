# -*- coding: utf-8 -*-
"""
Segment Record Domain Model
分割记录领域模型
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import json


@dataclass
class SegmentRecord:
    """分割记录领域模型 - 专门处理图像分割相关数据"""
    id: Optional[int] = None
    original_image_path: str = ""
    original_image_hash: str = ""
    selected_layers: List[str] = None
    segment_result_path: Optional[str] = None
    status: str = "pending"  # pending, segmenting, segmented, failed
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    user_id: Optional[int] = None
    analysis_record_id: Optional[int] = None  # 关联的分析记录ID
    
    def __post_init__(self):
        if self.selected_layers is None:
            self.selected_layers = []
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
            'selected_layers': json.dumps(self.selected_layers),
            'segment_result_path': self.segment_result_path,
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'user_id': self.user_id,
            'analysis_record_id': self.analysis_record_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SegmentRecord':
        """从字典创建实例"""
        selected_layers = []
        if data.get('selected_layers'):
            selected_layers = json.loads(data['selected_layers'])
        
        return cls(
            id=data.get('id'),
            original_image_path=data.get('original_image_path', ''),
            original_image_hash=data.get('original_image_hash', ''),
            selected_layers=selected_layers,
            segment_result_path=data.get('segment_result_path'),
            status=data.get('status', 'pending'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            user_id=data.get('user_id'),
            analysis_record_id=data.get('analysis_record_id')
        )
    
    def set_selected_layers(self, layers: List[str]):
        """设置用户选择的层"""
        self.selected_layers = layers
        self.updated_at = datetime.now()
    
    def update_status(self, status: str):
        """更新状态"""
        self.status = status
        self.updated_at = datetime.now()
    
    def set_segment_result(self, result_path: str):
        """设置分割结果路径"""
        self.segment_result_path = result_path
        self.updated_at = datetime.now()
    
    def link_analysis_record(self, analysis_record_id: int):
        """关联分析记录"""
        self.analysis_record_id = analysis_record_id
        self.updated_at = datetime.now()