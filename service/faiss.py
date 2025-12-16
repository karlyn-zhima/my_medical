import os
import faiss
from inference.inference import Inference
import json
import cv2
import base64
import numpy as np
import glob

class FaissService:
    def __init__(self, index_path, meta_path, infer: Inference, pool='gem', normalize=True):
        self.index_path = index_path
        self.meta_path = meta_path
        self.infer = infer
        self.pool = pool
        self.normalize = normalize
        self._index = None
        self._meta = None
        self._load_resources()

    def _load_resources(self):
        """Lazy load or reload index and metadata"""
        try:
            if os.path.isfile(self.index_path):
                self._index = faiss.read_index(self.index_path)
            if os.path.isfile(self.meta_path):
                with open(self.meta_path, 'r', encoding='utf-8') as f:
                    self._meta = json.load(f)
        except Exception as e:
            print(f"Error loading FAISS resources: {e}")

    def reload(self):
        """Force reload of index and metadata"""
        self._load_resources()

    def search_faiss(self, query_img_path, k=5):
        """
        加载已保存的索引与元数据，对单张查询图像执行检索，返回 Top-k 结果。
        """

        if self._index is None or self._meta is None:
            self._load_resources()
            
        if self._index is None:
            print(f"索引文件不存在或加载失败: {self.index_path}")
            return []
        if self._meta is None:
            print(f"元数据文件不存在或加载失败: {self.meta_path}")
            return []

        index = self._index
        meta = self._meta

        if not os.path.isfile(query_img_path):
            print(f"查询图像不存在: {query_img_path}")
            return []

        q = self.infer.get_global_embedding(query_img_path, pool=self.pool, normalize=self.normalize, return_numpy=True).astype('float32')
        q = q[None, :]  # (1, 768)
        D, I = index.search(q, k=min(k, index.ntotal))

        results = []
        for score, id_ in zip(D[0], I[0]):
            val = meta.get(str(int(id_))) or meta.get(int(id_))
            image_path = val.get('image_path')
            mask_path = val.get('mask_path')
            label_path = val.get('label_path')
            labels = val.get('labels')
            # 从对应路径读取image和mask
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # 将image和mask编码为base64字符串
            _, image_encoded = cv2.imencode('.png', image)
            _, mask_encoded = cv2.imencode('.png', mask)
            image_base64 = base64.b64encode(image_encoded).decode('utf-8')
            mask_base64 = base64.b64encode(mask_encoded).decode('utf-8')
            result = {
                "score": float(score),
                "image_path": image_path,
                "labels": labels,
                "image": image_base64,
                "mask": mask_base64,
            }
            results.append(result)
        return results
    
    def _find_mask_path(self, base_name, mask_dir):
        """根据图像基名扫描 mask_dir 下可能的掩码文件，返回第一个匹配到的路径；无则返回默认的 .png 路径但不创建文件。"""
        candidates = [
            os.path.join(mask_dir, base_name + ext)
            for ext in ('.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff')
        ]
        for c in candidates:
            if os.path.isfile(c):
                return os.path.abspath(c)
        # 若都不存在，返回首选的 .png 路径（仅记录路径，不写文件）
        return os.path.abspath(os.path.join(mask_dir, base_name + '.png'))

    def _find_label_path(self, base_name, label_dir):
        """扫描 label_dir 下以图像基名开头的 JSON 标签文件，返回第一个匹配到的路径；无则返回 None。"""
        if not label_dir:
            return None
        # 优先精确匹配 base.json，其次匹配 base*.json
        candidates = [
            os.path.join(label_dir, base_name + '.json')
        ]
        try:
            candidates.extend(glob.glob(os.path.join(label_dir, base_name + '*.json')))
        except Exception:
            pass
        for c in candidates:
            if os.path.isfile(c):
                return os.path.abspath(c)
        return None

    def add_to_index(self, image_paths, skip_duplicates=True, label_dir=None, mask_dir=None):
        """
        在已有索引基础上增量添加新的图片向量，并更新 FAISS 元数据与对应掩码路径。
        不修改用户已有标签文件：只读标签并统一到全局元数据。
        
        Args:
            image_paths: 要添加的图像路径列表
            skip_duplicates: 是否跳过已存在于元数据中的图片路径
            label_dir: 标签目录（只读）
            mask_dir: 掩码/可视化图像目录（仅记录路径，不生成或修改文件）
            
        Returns:
            dict: 包含成功状态和添加数量的结果
        """
        if not os.path.isfile(self.index_path):
            return {"success": False, "message": f"索引文件不存在: {self.index_path}"}
        if not os.path.isfile(self.meta_path):
            return {"success": False, "message": f"元数据文件不存在: {self.meta_path}"}

        # 读取现有索引与元数据
        index = faiss.read_index(self.index_path)
        # 若读出的索引不支持 add_with_ids，则用 IDMap 包装
        if not hasattr(index, "add_with_ids"):
            index = faiss.IndexIDMap(index)

        with open(self.meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        # 现有 ID 与路径集合（用于去重与生成新 ID）
        try:
            existing_ids = [int(k) for k in meta.keys()]
        except Exception:
            existing_ids = []
        next_id = (max(existing_ids) + 1) if existing_ids else 0
        existing_paths = set()
        for v in meta.values():
            if isinstance(v, str):
                existing_paths.add(os.path.abspath(v))
            elif isinstance(v, dict):
                p = v.get('image_path')
                if p:
                    existing_paths.add(os.path.abspath(p))

        vectors, ids, valid_paths = [], [], []
        for p in image_paths:
            if not os.path.isfile(p):
                print(f"跳过不存在的文件: {p}")
                continue
            ap = os.path.abspath(p)
            if skip_duplicates and ap in existing_paths:
                print(f"已存在于元数据，跳过: {ap}")
                continue
            vec = self.infer.get_global_embedding(ap, pool=self.pool, normalize=self.normalize, return_numpy=True).astype('float32')
            vectors.append(vec)
            ids.append(next_id)
            valid_paths.append(ap)
            next_id += 1

        if len(vectors) == 0:
            return {"success": False, "message": "没有需要增量添加的向量"}

        xb = np.stack(vectors, axis=0).astype('float32')
        index.add_with_ids(xb, np.array(ids, dtype='int64'))

        # 保存更新后的索引
        faiss.write_index(index, self.index_path)
        
        # 更新元数据
        for id_, p in zip(ids, valid_paths):
            base = os.path.splitext(os.path.basename(p))[0]
            mask_path = self._find_mask_path(base, mask_dir) if mask_dir else None
            label_path = self._find_label_path(base, label_dir) if label_dir else None
            labels = None
            if label_path:
                try:
                    with open(label_path, 'r', encoding='utf-8') as lf:
                        labels = json.load(lf)
                except Exception as e:
                    print(f"读取标签失败: {label_path}，错误: {e}")
            record = {"id": int(id_), "image_path": p, "mask_path": mask_path}
            if label_path:
                record["label_path"] = label_path
            if labels is not None:
                record["labels"] = labels
            meta[str(int(id_))] = record
            
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return {
            "success": True, 
            "message": f"增量添加完成: 新增 {len(ids)} 条。当前索引向量总数: {index.ntotal}",
            "added_count": len(ids),
            "total_count": index.ntotal
        }

    def get_best_match_label(self, query_img, mapping=None):
        """
        查询最佳匹配图片的标签（layer_name），并支持中英文映射
        :param query_img: 查询图像（路径或numpy array）
        :param mapping: 中英文映射字典 {chinese: english}
        :return: 翻译后的标签或 None
        """
        if self._index is None or self._meta is None:
            self._load_resources()
            
        if self._index is None or self._index.ntotal == 0:
            return None
            
        try:
            # Get embedding
            q = self.infer.get_global_embedding(query_img, pool=self.pool, normalize=self.normalize, return_numpy=True).astype('float32')
            q = q[None, :]
            
            # Search k=1
            D, I = self._index.search(q, k=1)
            
            if len(I) > 0 and len(I[0]) > 0:
                id_ = I[0][0]
                if id_ == -1: 
                    return None
                    
                val = self._meta.get(str(int(id_))) or self._meta.get(int(id_))
                if val:
                    labels = val.get('labels')
                    # print(labels)
                    # Extract layer_name
                    layer_name = None
                    layers=[]
                    if isinstance(labels, list):
                        # Assuming list of shapes/layers, take the first one or logic?
                        # User said "rag return content is its content", implying 1-1 match or just take what's there.
                        # I'll take the first layer_name found.
                        for item in labels:
                            if isinstance(item, dict) and 'layer' in item:
                                layer_name = item['layer']
                                if item['exists']:
                                    layers.append(layer_name)
                    elif isinstance(labels, dict):
                        layer_name = labels.get('layer')
                        if layer_name:
                            layers.append(layer_name)
                    # print(layers)
                    if len(layers) > 0:
                        # print(mapping)
                        if mapping:
                            return [mapping.get(layer_name, layer_name) for layer_name in layers], layers
                        return layers, layers
        except Exception as e:
            print(f"Error in get_best_match_label: {e}")
            
        return None
    