# -*- coding: utf-8 -*-
"""
模型工厂：统一、灵活地创建并加载多种模型（TransUNet 与 ViT-Seg）。

用法示例：

from model import create_model
model, cfg = create_model(
    model_type="transunet",          # or "vit", "vit-seg"
    num_classes=6,
    img_size=(256, 256),              # 支持 int 或 (H, W)
    vit_name="R50-ViT-B_16",
    n_skip=3,
    vit_patches_size=16,
    pretrained_path=None,             # 如需加载预训练权重，传入 .npz 路径
    device="cuda"                    # 可选：自动放置到设备
)
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union

import os
import numpy as np
import torch

from .multi_guide import vit_seg_modeling as vit_mod
from .multi_guide import transunet as trans_mod
from .multi_guide.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

SupportedName = Union[str, None]

# 允许的名称映射（大小写不敏感）
_NAME_ALIASES = {
    "vit": "vit",
    "multi-seg": "vit",
    "multi_seg": "vit",
    "multi_seg_modeling": "vit",
    "transunet": "transunet",
    "trans-unet": "transunet",
    "trans_unet": "transunet",
}


def _normalize_size(img_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """将 img_size 统一为 (H, W)。"""
    if isinstance(img_size, int):
        return img_size, img_size
    if isinstance(img_size, (tuple, list)) and len(img_size) == 2:
        h, w = int(img_size[0]), int(img_size[1])
        return h, w
    raise ValueError("img_size 必须为 int 或 (H, W) 二元组")


def _prepare_config(module, vit_name: str, num_classes: int, n_skip: int,
                    img_size: Tuple[int, int], vit_patches_size: int):
    """根据目标模块(vit/trans)准备配置对象。"""
    if vit_name not in module.CONFIGS:
        raise KeyError(f"未找到 vit_name: {vit_name}，可选项：{list(module.CONFIGS.keys())}")
    config = module.CONFIGS[vit_name]
    config.n_classes = num_classes
    config.n_skip = n_skip

    h, w = img_size
    # R50 变体需要设定 grid
    if "R50" in vit_name:
        config.patches.grid = (w // vit_patches_size, h // vit_patches_size)
    return config


def create_model(
    model_type: SupportedName,
    *,
    num_classes: int,
    img_size: Union[int, Tuple[int, int]] = 224,
    vit_name: str = "R50-ViT-B_16",
    n_skip: int = 3,
    vit_patches_size: int = 16,
    pretrained_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.nn.Module, Any]:
    """创建并返回模型与其配置。

    参数：
        model_type: 模型类型，支持别名："vit"/"vit-seg" 与 "transunet" 等。
        num_classes: 分类/分割类别数。
        img_size: 输入尺寸（int 或 (H, W)）。
        vit_name: ViT 配置名，例如 "R50-ViT-B_16"。
        n_skip: 跳连数量。
        vit_patches_size: ViT patch 大小或 grid 计算使用的 stride（R50 变体）。
        pretrained_path: 可选，npz 权重路径（使用 .load_from 加载）。
        device: 可选，放置设备（例如 "cuda"、"cpu"）。

    返回：
        (model, config)
    """
    if model_type is None:
        raise ValueError("必须提供 model_type，例如 'vit' 或 'transunet'")
    key = _NAME_ALIASES.get(str(model_type).lower())
    if key is None:
        raise ValueError(f"不支持的 model_type: {model_type}")

    h, w = _normalize_size(img_size)

    if key == "vit":
        module = vit_mod
    elif key == "transunet":
        module = trans_mod
    else:
        raise ValueError(f"未匹配的 model_type: {model_type}")

    config = _prepare_config(module, vit_name, num_classes, n_skip, (h, w), vit_patches_size)

    # 这两个实现的 VisionTransformer 构造签名一致
    model = module.VisionTransformer(config, img_size=h, num_classes=config.n_classes)


    # 可选：加载预训练权重（npz）
    # 可选：加载预训练权重（npz）
    resolved_pretrained: Optional[str] = None
    if pretrained_path is not None:
        pp = str(pretrained_path).strip()
        # 允许通过传入空串/none/null 来显式跳过
        if pp and pp.lower() not in {"none", "null"}:
            # 将相对路径解析为项目根目录下的绝对路径
            if not os.path.isabs(pp):
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                cand = os.path.normpath(os.path.join(project_root, pp))
                if os.path.exists(cand):
                    resolved_pretrained = cand
                elif os.path.exists(pp):
                    resolved_pretrained = pp
            else:
                if os.path.exists(pp):
                    resolved_pretrained = pp
    if resolved_pretrained:
        print(f"加载预训练权重: {resolved_pretrained}")
        model.load_from(weights=np.load(resolved_pretrained),)
    else:
        if pretrained_path:
            print(f"预训练权重未找到或无效，跳过加载: {pretrained_path}")
        else:
            print("未配置预训练权重，跳过加载。")

    if device is not None:
        model = model.to(device)

    return model, config


def create_model_from_config(cfg: Dict[str, Any]) -> Tuple[torch.nn.Module, Any]:
    """从参数字典创建模型，字段对齐 train/main_aug.py 的命名习惯。

    识别字段：
        - name: 模型名（如 'TransUnet' 或 'vit'）
        - num_class, input_h, input_w, vit_name, n_skip, vit_patches_size, pretrained_path, device
    """
    model_type = cfg.get("name")
    num_classes = int(cfg.get("num_class") or cfg.get("n_classes"))
    h = int(cfg.get("input_h", 224))
    w = int(cfg.get("input_w", h))
    vit_name = cfg.get("vit_name", "R50-ViT-B_16")
    n_skip = int(cfg.get("n_skip", 3))
    vit_patches_size = int(cfg.get("vit_patches_size", 16))
    pretrained_path = cfg.get("pretrained_path")
    device = cfg.get("device")

    return create_model(
        model_type,
        num_classes=num_classes,
        img_size=(h, w),
        vit_name=vit_name,
        n_skip=n_skip,
        vit_patches_size=vit_patches_size,
        pretrained_path=pretrained_path,
        device=device,
    )


def list_supported_models() -> Dict[str, str]:
    """返回受支持名称及其归一化后键。"""
    return dict(_NAME_ALIASES)


__all__ = [
    "create_model",
    "create_model_from_config",
    "list_supported_models",
]