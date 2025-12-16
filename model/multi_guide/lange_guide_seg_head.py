import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from model.multi_guide.semantic_opposite_generator import SemanticOppositeGenerator

class HighGuide_CrossAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super(HighGuide_CrossAttention, self).__init__()
        self.query_transform = nn.Linear(512, embed_dim)
        self.query_transform_y = nn.Linear(512, 256)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,dropout=0.2)
        self.relu=nn.PReLU()

    def forward(self, query, key, value):
        query=query.float()
        queryx = self.query_transform(query).unsqueeze(-2)  # (N, B, 1,768)
        queryy = self.query_transform_y(query).unsqueeze(-1) #(N, B ,256,1)
        queryx = self.relu(queryx)
        queryy = self.relu(queryy)
        querys =queryx*queryy
        list=[]
        for query in querys:
            query = query.permute(1,0,2)
            attn_output, _ = self.attn(query=query, key=key, value=value)  # (256, B, 768)
            attn_output = self.relu(attn_output)
            list.append(attn_output.permute(1,0,2))
        res=torch.stack(list, dim=0)
        return res

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads,dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embed dim must be divisible by the number of heads."

        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.sig=nn.PReLU()

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_linear(output)
        output = self.dropout(output)  # 应用dropout
        output = self.sig(output)

        return output

# 定义交叉注意力模块
class CrossAttention(nn.Module):
    def __init__(self, feature_dim=1, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = MultiHeadAttention(d_model, nhead)
        self.linear = nn.Linear(d_model, feature_dim)
        self.norm = nn.LayerNorm(d_model)
        # self.linear_a = nn.Linear(512, 512)
        self.linear_b = nn.Linear(256, 512)
        self.dropout = nn.Dropout(dropout)
        self.sig=nn.PReLU()
    
    def forward(self, query, key):
        # 调整张量形状
        query = query.to(dtype=torch.float32)  # 调整为 (seq_len, batch)
        key = key.permute(0 ,1,2)     # 调整为 (seq_len, batch)
        key=self.linear_b(key)
        key=self.sig(key)
        attn_output= self.multihead_attn(query, key, key)
        attn_output = self.dropout(attn_output)  # 应用dropout

        attn_output = attn_output+query
        attn_output = self.sig(attn_output)
        return attn_output

class HighGuide(nn.Module):
    def __init__(self):
        super(HighGuide, self).__init__()
        self.bottom_attention_1=HighGuide_CrossAttention()
        self.bottom_attention_2=HighGuide_CrossAttention()
        self.bottom_attention_3=HighGuide_CrossAttention()
        self.relu=nn.SiLU()
    def forward(self, mins,text_features):
        text_features=text_features.permute(1,0,2)
        mins=mins.permute(1,0,2)
        res = self.bottom_attention_1(text_features,mins,mins)
        res=torch.mean(res,dim=0)
        res = res.permute(1,0,2)
        res = self.bottom_attention_2(text_features,res,res)
        res=torch.mean(res,dim=0)
        res = res.permute(1,0,2)
        res = self.bottom_attention_3(text_features,res,res)
        # res=res+mins.permute(1,0,2)
        res=torch.mean(res,dim=0)
        # res=self.relu(res)
        return res


from timm.models.layers import CondConv2d

class DynamicConv(nn.Module):
    """ Dynamic Conv layer
    """
    def __init__(self, routing_features,in_features, out_features, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, num_experts=8):
        super().__init__()
        self.routing = nn.Linear(routing_features, num_experts)
        self.cond_conv = CondConv2d(in_features, in_features*2, kernel_size, stride, padding, dilation,
                 groups, bias, num_experts)
        self.relu = nn.PReLU()
        self.cond_conv_output = CondConv2d(in_features*2, in_features, kernel_size, stride, padding, dilation,
                 groups, bias, num_experts)
        self.self_routing = nn.Linear(in_features*2, num_experts)
        self.conv_output = torch.nn.Conv2d(in_features, 1, 1, stride, 0)
        
    def forward(self, x, routing):
        routing_weights = self.routing(routing)
        routing_weights = torch.sigmoid(routing_weights)
        resmap = []
        for routing_weight in routing_weights.permute(1,0,2):
            x_c = self.relu(self.cond_conv(x, routing_weight))
            pooled_inputs = F.adaptive_avg_pool2d(x_c, 1).flatten(1)
            self_routing_weight = self.self_routing(pooled_inputs)
            x_c = self.relu(self.cond_conv_output(x_c,self_routing_weight))
            x_c = self.conv_output(x_c)
            resmap.append(x_c)
        x = torch.cat(resmap, dim=1)
        return x

class LGSHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        super(LGSHead, self).__init__()
        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 768),
            nn.PReLU(),
        )
        self.out_channels=out_channels
        self.in_channels=in_channels
        self.cross_attention = CrossAttention()
        self.bg_gen=SemanticOppositeGenerator()
        self.out_conv = DynamicConv(512,in_channels,1,kernel_size,stride=1,padding=1)
    
    
    def forward(self, mins,input,text_features):
        # print(self.in_channels)
        B, n_patch, hidden = mins.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        mins = mins.permute(0, 2, 1)
        mins = mins.contiguous().view(B, hidden, h, w)
        x_feat = self.GAP(mins)
        x_feat_cat=x_feat.view(x_feat.shape[0],x_feat.shape[1], -1)
        bg_feature=self.bg_gen(text_features)
        text_features=torch.cat([bg_feature,text_features],dim=1)
        cross_res=self.cross_attention(text_features,x_feat_cat)
        x=self.out_conv(input,cross_res)
        x=torch.softmax(x,dim=1)
        return x,text_features
    