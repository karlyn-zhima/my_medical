import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticOppositeGenerator(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=256):
        """
        可训练的语义相反token生成模块
        
        参数：
            embed_dim (int): 词向量维度，默认512
            hidden_dim (int): 隐藏层维度，默认256
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # 可学习参数矩阵
        self.projector = nn.Sequential(
            nn.Linear(embed_dim*2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """参数初始化"""
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, tokens):
        """
        前向传播
        
        参数：
            tokens (Tensor): 输入token序列，形状(B, N, D)
            
        返回：
            opposite_token (Tensor): 生成的相反token，形状(B, 1, D)
        """
        B, N, D = tokens.shape
        
        # 步骤1：计算统计特征
        mean_vector = tokens.mean(dim=1)  # (B, D)
        max_vector = tokens.max(dim=1).values  # (B, D)
        
        # 步骤2：构建特征输入
        combined_feature = torch.cat([mean_vector, max_vector], dim=1)  # (B, 2D)
        
        # 步骤3：生成初始相反向量
        raw_output = self.projector(combined_feature)  # (B, D)
        
        # 步骤4：正交化处理
        normalized_tokens = F.normalize(tokens, p=2, dim=-1)
        similarity_matrix = torch.matmul(
            raw_output.unsqueeze(1),  # (B, 1, D)
            normalized_tokens.transpose(1,2)  # (B, D, N)
        )  # (B, 1, N)
        
        # 动态正交加权
        weights = F.softmax(-similarity_matrix, dim=-1)  # (B, 1, N)
        orthogonal_component = torch.matmul(weights, normalized_tokens)  # (B, 1, D)
        
        # 步骤5：组合最终结果
        final_vector = F.normalize(raw_output.unsqueeze(1) - 0.5*orthogonal_component)
        opposite_token = F.normalize(final_vector, p=2, dim=-1)
        
        return opposite_token
