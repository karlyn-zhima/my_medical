import torch
import torch.nn as nn
import torch.nn.functional as F

class RAGIndexBranch(nn.Module):
    def __init__(self, input_dim=256*768, hidden_dims=[1024, 512], output_dim=512, 
                 dropout_rate=0.2, use_batch_norm=True):
        super(RAGIndexBranch, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 特征预处理层
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入形状: N × 256 × 768
        batch_size = x.shape[0]
        
        # 展平特征
        x_flat = x.view(batch_size, -1)
        
        # 通过网络
        features = self.network(x_flat)
        
        # L2归一化用于余弦相似度
        normalized_features = F.normalize(features, p=2, dim=1)
        
        return normalized_features

class RAGIndexConvBranch(nn.Module):
    """
    卷积版的 RAG 索引分支：
    - 输入: N × 256 × 768，视为 16×16 的空间网格，每个位置 768 维特征
    - 处理: 通过多层卷积逐步进行空间聚合，提取全局语义
    - 输出: N × 512 的 L2 归一化嵌入，用于余弦相似度
    """
    def __init__(
        self,
        output_dim: int = 512,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        channels_reduction: int = 512,
    ):
        super(RAGIndexConvBranch, self).__init__()

        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        in_channels = 768
        mid_channels = channels_reduction  # 将通道维度从 768 降到 512

        def bn(c):
            return nn.BatchNorm2d(c) if use_batch_norm else nn.Identity()

        def dp():
            return nn.Dropout2d(dropout_rate) if dropout_rate and dropout_rate > 0 else nn.Identity()

        # 逐层卷积与下采样：16x16 -> 8x8 -> 4x4 -> 2x2
        self.stage = nn.Sequential(
            # 先用深度可分离卷积高效对通道和空间特征进行融合（保持 16x16）
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            bn(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            bn(mid_channels),
            nn.ReLU(inplace=True),
            dp(),

            # 下采样到 8x8
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            bn(mid_channels),
            nn.ReLU(inplace=True),
            dp(),

            # 保持 8x8，进一步提特征
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            bn(mid_channels),
            nn.ReLU(inplace=True),

            # 下采样到 4x4（使用深度可分离卷积）
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False),
            bn(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            bn(mid_channels),
            nn.ReLU(inplace=True),
            dp(),

            # 下采样到 2x2
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            bn(mid_channels),
            nn.ReLU(inplace=True),
        )

        # 将 2x2 特征进行全局聚合
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 输出映射到指定维度
        self.proj = nn.Sequential(
            nn.Linear(mid_channels, output_dim, bias=True),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状: N × 256 × 768，将其视为 16 × 16 × 768 的图像特征
        # 先确保维度符合要求
        batch_size = x.shape[0]
        assert x.shape[1] == 256 and x.shape[2] == 768, "输入应为 N×256×768"

        # 展开为图像特征: (N, 16, 16, 768) -> (N, 768, 16, 16)
        x_img = x.view(batch_size, 16, 16, 768).permute(0, 3, 1, 2).contiguous()

        # 通过卷积网络逐步提取到全局
        feats = self.stage(x_img)

        # 全局平均池化 -> (N, C, 1, 1) -> (N, C)
        pooled = self.global_pool(feats).view(batch_size, -1)

        # 映射到输出维度并进行 L2 归一化
        out = self.proj(pooled)
        normalized_features = F.normalize(out, p=2, dim=1)
        return normalized_features

class RAGIndexAttentionBranch(nn.Module):
    def __init__(self, input_dim=768, index_dim=512, num_queries=8, hidden_dim=768):
        """
        Args:
            num_queries (int): 前景原型的数量 (K)。建议设为 4, 8 或 16。
                               代表网络能识别多少种不同类型的前景模式。
        """
        super().__init__()
        
        self.num_queries = num_queries
        
        # 1. 定义 K 个可学习的前景 Query
        # Shape: (1, K, D)
        self.foreground_queries = nn.Parameter(torch.randn(1, num_queries, input_dim))
        
        self.scale = input_dim ** -0.5

        # 2. MLP 投影层
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, index_dim)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.foreground_queries, std=0.02)
        for m in self.projector:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: ViT 输出特征, Shape: (B, L, D) (例如 B, 256, 768)
        Returns:
            index_vector: (B, index_dim)
            max_attn_weights: (B, 1, L) 用于可视化的聚合注意力图
        """
        B, L, D = x.shape

        # --- 步骤 1: 多头注意力加权 ---
        
        # 扩展 Queries: (1, K, D) -> (B, K, D)
        qs = self.foreground_queries.expand(B, -1, -1)
        
        # 计算注意力分数: (B, K, D) @ (B, D, L) -> (B, K, L)
        # 代表 K 个 Query 对 L 个 Patch 的关注程度
        attn_scores = torch.bmm(qs, x.transpose(1, 2)) * self.scale
        
        # Softmax (在空间维度 L 上归一化)
        attn_weights = F.softmax(attn_scores, dim=-1) # (B, K, L)
        
        # 加权聚合: (B, K, L) @ (B, L, D) -> (B, K, D)
        # 此时我们得到了 K 个不同的前景特征向量
        multi_prototype_features = torch.bmm(attn_weights, x)

        # --- 步骤 2: 原型聚合 (Max Pooling) ---
        
        # 在 K (num_queries) 维度上取最大值
        # 含义：对于每个特征通道，保留所有原型中反应最强烈的信号
        # Shape: (B, K, D) -> (B, D)
        aggregated_features, _ = torch.max(multi_prototype_features, dim=1)
        
        # 我们想知道最终网络关注了哪些区域，取 K 个注意力图的最大值
        # (B, K, L) -> (B, L) -> (B, 1, L)
        max_attn_weights, _ = torch.max(attn_weights, dim=1)
        max_attn_weights = max_attn_weights.unsqueeze(1)

        # --- 步骤 3: 投影与归一化 ---
        projected = self.projector(aggregated_features) # (B, index_dim)
        index_vector = F.normalize(projected, p=2, dim=1)

        return index_vector

class MultiDatasetRAGModel(nn.Module):
    def __init__(self, pretrained_backbone, rag_branch):
        super(MultiDatasetRAGModel, self).__init__()
        self.pretrained_backbone = pretrained_backbone
        self.rag_branch = rag_branch
        
        # 冻结预训练主干网络
        for param in self.pretrained_backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # 前向传播时禁用预训练主干的梯度
        with torch.no_grad():
            backbone_features,_,_ = self.pretrained_backbone(x)
        

        # 仅RAG分支参与梯度计算
        rag_features = self.rag_branch(backbone_features)
        
        return rag_features