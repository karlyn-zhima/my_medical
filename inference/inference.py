import os
import re
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from PIL import Image, ImageDraw
from clip import clip
from model.multi_guide import vit_seg_modeling
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import torch.nn.functional as F
from model.rag_embedding.network import RAGIndexAttentionBranch ,MultiDatasetRAGModel
# from .visual_bge.visual_bge.modeling import Visualized_BGE

class Inference:
    def __init__(self, ckpt,ckpt_rag=r'ckpt\rag_branch_best.weight'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        config = vit_seg_modeling.CONFIGS["R50-ViT-B_16"]
        config.n_classes = 1
        config.n_skip = 3
        config.patches.grid = (16, 16)
        self.model = vit_seg_modeling.VisionTransformer(config,img_size=256,num_classes=config.n_classes).to(self.device)
        self.text_model, self.text_preprocess = clip.load("ViT-B/32",device=self.device)
        if os.path.isfile(ckpt):
            state_dict = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print('\nLoaded checkpoint:', ckpt)
        self.embedding = MultiDatasetRAGModel(
                pretrained_backbone=self.model.transformer,
                rag_branch=RAGIndexAttentionBranch().cuda(),
            ).cuda()
        if os.path.isfile(ckpt_rag):
            state_dict = self._safe_torch_load(ckpt_rag, map_location=self.device)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                self.embedding.load_state_dict(state_dict['model_state_dict'])
            elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
                self.embedding.load_state_dict(state_dict['state_dict'])
            else:
                self.embedding.load_state_dict(state_dict)
            print('\nLoaded rag checkpoint:', ckpt_rag)
        self.embedding.eval()
        
        # 设置模型为评估模式，禁用dropout等训练时的随机性
        self.model.eval()
        self.text_model.eval()
        # self.embedding_model = Visualized_BGE(model_name_bge=r"BAAI/bge-base-en-v1.5",model_weight=r"C:\Users\Administrator\Desktop\demo\inference\Visualized_base_en_v1.5.pth")

        # Pre-allocate normalization tensors on device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

    def encode_text(self, text):
        """
        Encode text description to feature vector
        """
        token = clip.tokenize(text).to(self.device)
        text_feature = self.text_model.encode_text(token).to(self.device).float()
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature.unsqueeze(0).to(self.device)
        return text_feature

    def img_prepare(self, img_input):
        """
        对图像进行预处理，包括resize、归一化、正则化
        :param img_input: 输入图像路径 str 或 numpy array (RGB)
        :return: 预处理后的图像tensor (1, 3, 256, 256)
        """
        if isinstance(img_input, str):
            img = Image.open(img_input).convert('RGB')
        elif isinstance(img_input, np.ndarray):
            img = Image.fromarray(img_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(img_input)}")

        img = img.resize((256, 256))
        img_np = np.array(img)
        # 图像正则化处理
        # Create tensor directly from numpy
        img_tensor = torch.from_numpy(img_np).permute(2,0,1).float() / 255.0
        
        # Use pre-allocated mean/std
        img_tensor = (img_tensor.to(self.device) - self.mean) / self.std
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    def interface(self, img_input, destribe, visualize=False, text_feature=None):
        """
        推理接口，支持可视化结果保存
        :param img_input: 输入图像路径 或 numpy array
        :param destribe: 文本描述
        :param visualize: 是否可视化结果
        :param text_feature: 预计算的文本特征，如果提供则跳过编码
        :return: 预测 mask (numpy array)
        """
        img_tensor = self.img_prepare(img_input)
        
        if text_feature is None:
            text_feature = self.encode_text(destribe)
            
        try:
            with torch.no_grad():
                pred, _ = self.model(img_tensor, text_feature)
                pred = torch.argmax(pred, dim=1).squeeze(0)
                pred_np = pred.cpu().numpy()
             
            if visualize:
                # 重新读取并缩放原图以匹配预测尺寸，用于可视化叠加
                if isinstance(img_input, str):
                    img = Image.open(img_input).convert('RGB')
                elif isinstance(img_input, np.ndarray):
                    img = Image.fromarray(img_input)
                    
                img = img.resize((256, 256))
                img_np = np.array(img)
                return self.conv_to_mask(pred_np, img_np)
            return pred_np
        finally:
            # Explicitly delete tensors to free CUDA memory
            del img_tensor
            if 'pred' in locals():
                del pred
    
    def img_embedding(self, img_path,mode="vit"):
        """
        只使用模型的embedding部分对图像进行编码，作为向量数据库的key
        :param img_path: 输入图像路径
        :return: 图像embedding tensor (1, n_patches, hidden_size)，默认配置约为 (1, 256, 768)
        """
        # if mode=="bge":
        #     img_emb = self.embedding_model.encode(image=img_path)
        #     img_emb = img_emb.squeeze(0)
        #     img_emb = img_emb.detach().cpu().numpy()
        #     return img_emb
        img_tensor = self.img_prepare(img_path)
        with torch.no_grad():
            img_embedding, _, _ = self.model.transformer(img_tensor)
        return img_embedding

    def test_img_embedding(self, img_path=None):
        """
        简单测试：运行 img_embedding 并校验输出维度与类型。
        :param img_path: 可选，传入图像路径；不传则生成一张随机 256x256 RGB 测试图像。
        :return: embedding tensor
        """
        created_temp = False
        if img_path is None:
            tmp_path = os.path.join(os.path.dirname(__file__), "tmp_test_img.png")
            img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
            img.save(tmp_path)
            img_path = tmp_path
            created_temp = True

        emb = self.img_embedding(img_path)

        # 依据当前配置计算期望的 patch 数
        config = self.model.config
        img_size = 256
        expected_n_patches = None
        if hasattr(config, "patches") and config.patches.get("grid") is not None:
            grid_x, grid_y = config.patches["grid"]
            patch_size = (img_size // 16 // grid_x, img_size // 16 // grid_y)
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            expected_n_patches = (img_size // patch_size_real[0]) * (img_size // patch_size_real[1])
        else:
            size = config.patches["size"]
            expected_n_patches = (img_size // size[0]) * (img_size // size[1])

        hidden = config.hidden_size
        assert emb.shape[0] == 1 and emb.shape[1] == expected_n_patches and emb.shape[2] == hidden, \
            f"Unexpected embedding shape: {tuple(emb.shape)}; expected (1, {expected_n_patches}, {hidden})"
        assert emb.dtype == torch.float32, f"Unexpected dtype: {emb.dtype}"

        print(f"img_embedding OK. Shape: {tuple(emb.shape)}, dtype: {emb.dtype}")
        if created_temp:
            try:
                os.remove(img_path)
            except Exception:
                pass
        return emb

    def conv_to_mask(self, pred_mask, img_np):
        """
        转换预测 mask 为多值 mask
        :param pred_mask: 预测 mask numpy array (H, W)，值为类别索引
        :param img_np: 原图 numpy array (H, W, 3)
        :return: 图片覆盖mask
        """
        palette = np.array([
            [0, 0, 0],       # 0 背景 黑色
            [255, 0, 0],     # 1 红色
            [0, 255, 0],     # 2 绿色
            [0, 0, 255],     # 3 蓝色
            [255, 255, 0],   # 4 黄色
            [255, 0, 255],   # 5 品红
            [0, 255, 255],   # 6 青色
            [255, 128, 0],   # 7 橙色
            [128, 0, 255],   # 8 紫色
        ], dtype=np.uint8)

        h, w = pred_mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls in np.unique(pred_mask):
            if cls < len(palette):
                color_mask[pred_mask == cls] = palette[cls]

        # 叠加图
        overlay = cv2.addWeighted(img_np, 0.6, color_mask, 0.4, 0)
        return overlay

    def visualize_multiclass(self, img_np, pred_mask, destribe, save_path=None):
        """
        可视化多类分割结果：原图、彩色 mask、叠加图
        :param img_np: 原图 numpy array (H, W, 3)
        :param pred_mask: 多类预测 mask numpy array (H, W)，值为类别索引
        :param destribe: 文本描述
        :param save_path: 保存路径，为 None 则只显示不保存
        :return: 彩色 mask 图像 (RGB)
        """
        # 定义调色板（可扩展）
        palette = np.array([
            [0, 0, 0],       # 0 背景 黑色
            [255, 0, 0],     # 1 红色
            [0, 255, 0],     # 2 绿色
            [0, 0, 255],     # 3 蓝色
            [255, 255, 0],   # 4 黄色
            [255, 0, 255],   # 5 品红
            [0, 255, 255],   # 6 青色
            [255, 128, 0],   # 7 橙色
            [128, 0, 255],   # 8 紫色
        ], dtype=np.uint8)

        h, w = pred_mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls in np.unique(pred_mask):
            if cls < len(palette):
                color_mask[pred_mask == cls] = palette[cls]

        # 叠加图
        overlay = cv2.addWeighted(img_np, 0.6, color_mask, 0.4, 0)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(color_mask)
        plt.title("Colored Mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title("Overlay\n" + destribe)
        plt.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        plt.close()

        return color_mask

    def get_global_embedding(self, img_path, pool='mean', normalize=True, return_numpy=True, use_rag=True):
        """
        将多个 patch 的 768 维特征聚合为单个 768 维的全图向量，用作向量数据库索引。
        :param img_path: 输入图像路径
        :param pool: 聚合方式，可选 'mean'、'max'、'gem' 或 'attn'
        :param normalize: 是否进行 L2 归一化
        :param return_numpy: True 返回 numpy.ndarray；False 返回 torch.Tensor
        :return: 形状为 (768,) 的向量
        """
        img = self._to_pil_rgb(img_path)
        img = img.resize((256, 256))
        img_np = np.array(img)
        img_tensor = torch.tensor(img_np).permute(2,0,1).to(dtype=torch.float32)
        img_tensor = img_tensor.unsqueeze(0).cuda()
        with torch.no_grad():
            rag_features = self.embedding(img_tensor).squeeze(0).cpu().numpy()
        return rag_features

    def _safe_torch_load(self, path, map_location=None):
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=map_location)

    def _to_pil_rgb(self, src):
        if isinstance(src, Image.Image):
            img = src
        elif isinstance(src, np.ndarray):
            arr = src
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating):
                    arr = np.clip(arr, 0.0, 1.0)
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        elif hasattr(src, 'read'):
            img = Image.open(src)
        elif isinstance(src, (str, os.PathLike)):
            img = Image.open(src)
        elif isinstance(src, (bytes, bytearray)):
            img = Image.open(io.BytesIO(src))
        else:
            raise TypeError('Unsupported image input type')
        return img.convert('RGB')