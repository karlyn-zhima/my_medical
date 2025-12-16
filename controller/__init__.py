# -*- coding: utf-8 -*-
"""
Controller Package
控制器包，包含所有API端点的控制器
"""

from .segment_controller import segment_router
from .search_controller import search_router
from .model_controller import model_router
from .user_controller import user_router
from .confirm_controller import confirm_router
from .video_controller import video_router

__all__ = [
    'segment_router',
    'search_router', 
    'model_router',
    'user_router',
    'confirm_router',
    'video_router'
]