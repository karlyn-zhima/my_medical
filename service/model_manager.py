# -*- coding: utf-8 -*-
"""
Model Manager Service
模型管理服务，实现单例模式和模型缓存，避免重复加载模型
"""

import os
import threading
from typing import Optional, Dict, Any
import json
from inference.inference import Inference
from service.faiss import FaissService


class ModelManager:
    """模型管理器，实现单例模式"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._inference_model: Optional[Inference] = None
            self._faiss_service: Optional[FaissService] = None
            self._model_path: Optional[str] = None
            self._ref_count = 0
            self._initialized = True
    
    def _read_default_paths(self) -> Dict[str, Optional[str]]:
        """从配置文件读取默认模型路径，找不到则使用合理的内置默认值。"""
        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            config_path = os.path.join(base_dir, 'config', 'config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            model_path = cfg.get('default_model_path') or cfg.get('model_path') or 'ckpt/multi_seg_best_1105.weight'
            rag_path = cfg.get('rag_branch_model_path') or 'ckpt/rag_branch_best.weight'
            return { 'model_path': model_path, 'rag_path': rag_path }
        except Exception:
            return { 'model_path': 'ckpt/multi_seg_best_1105.weight', 'rag_path': 'ckpt/rag_branch_best.weight' }

    def get_inference_model(self, model_path: Optional[str] = None, rag_path: Optional[str] = 'ckpt/rag_branch_best_attention.weight') -> Inference:
        """
        获取推理模型实例，如果模型未加载或路径发生变化则重新加载
        
        Args:
            model_path: 模型文件路径
            rag_path: RAG分支模型文件路径
            
        Returns:
            Inference: 推理模型实例
        """
        # 优先复用已加载的模型（不传入路径时不触发切换）
        if self._inference_model is not None and (model_path is None or self._model_path == model_path):
            self._ref_count += 1
            print(f"复用已加载模型，引用计数+1，当前引用计数: {self._ref_count}")
            return self._inference_model

        # 需要加载或切换模型
        if model_path is None:
            defaults = self._read_default_paths()
            model_path = defaults['model_path']
            rag_path = rag_path if rag_path is not None else defaults['rag_path']

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        if rag_path and os.path.exists(rag_path):
            print(f"加载RAG分支模型: {rag_path}")
        else:
            rag_path = None

        print(f"加载模型: {model_path}")
        self._inference_model = Inference(model_path, rag_path)
        self._model_path = model_path
        self._ref_count = 1
        print(f"模型加载完成，引用计数初始化为: {self._ref_count}")

        # 模型更新后，需要重置FAISS服务
        self._faiss_service = None

        return self._inference_model
    
    def release_inference_model(self, force: bool = False):
        """
        释放推理模型以回收显存
        
        Args:
            force: 是否强制释放（忽略引用计数）
        """
        if self._inference_model:
            if not force:
                self._ref_count -= 1
                print(f"释放模型引用，引用计数-1，当前引用计数: {self._ref_count}")
                if self._ref_count > 0:
                    print("引用计数大于0，暂不释放显存")
                    return

            print("正在释放推理模型及显存...")
            del self._inference_model
            self._inference_model = None
            self._model_path = None
            self._ref_count = 0
            
            # 同时也需要清理FAISS服务
            self._faiss_service = None
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 清理PyTorch显存缓存
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("已释放推理模型及显存")

    def get_faiss_service(
        self, 
        index_path: str = "faiss.index",
        meta_path: str = "faiss_meta.json",
        pool: str = 'attn',
        normalize: bool = True,
    ) -> FaissService:
        """
        获取FAISS服务实例，如果服务未初始化则创建
        
        Args:
            index_path: FAISS索引文件路径
            meta_path: 元数据文件路径
            pool: 池化方法
            normalize: 是否归一化
            
        Returns:
            FaissService: FAISS服务实例
        """
        # 如果FAISS服务未初始化，则创建
        if self._faiss_service is None:
            # 复用已加载的模型；如未加载则按配置默认值加载一次
            inference_model = self.get_inference_model()
            self._faiss_service = FaissService(
                index_path=index_path,
                meta_path=meta_path,
                infer=inference_model,
                pool=pool,
                normalize=normalize,
            )
        
        return self._faiss_service
    
    def reset_model(self, model_path: str, rag_path: Optional[str] = None) -> None:
        """
        重置模型，强制重新加载
        
        Args:
            model_path: 新的模型文件路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"重置模型: {model_path}")
        # 重置时允许指定RAG路径
        defaults = self._read_default_paths()
        rag_resolved = rag_path if rag_path is not None else defaults['rag_path']
        if rag_resolved and not os.path.exists(rag_resolved):
            rag_resolved = None
        self._inference_model = Inference(model_path, rag_resolved)
        self._model_path = model_path
        
        # 重置FAISS服务
        self._faiss_service = None
    
    def is_model_loaded(self) -> bool:
        """
        检查模型是否已加载
        
        Returns:
            bool: 模型是否已加载
        """
        return self._inference_model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            "model_loaded": self.is_model_loaded(),
            "model_path": self._model_path,
            "faiss_service_initialized": self._faiss_service is not None
        }


# 全局模型管理器实例
_model_manager = None

def get_model_manager() -> ModelManager:
    """获取全局模型管理器实例"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager