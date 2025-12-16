from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
from .transform import *
import torch
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer
import random

class SingleData(Dataset):
    def __init__(self, image_size=(256,256), class_num=27, data_path='./data/BUS/',ranks=['upper skin','fatty layer','glandular','muscle','breast tumors'], random_add_rank =None,mode='train',argue=True):
        super(SingleData, self).__init__()
        self.ranksAll = ['upper skin','fatty layer','glandular','muscle','breast tumors','thyroid nodule']
        self.img_width,self.img_height=image_size
        self.class_num=class_num
        self.data_path=data_path
        self.path_to_img=os.path.join(data_path,'image/')
        self.path_to_label=os.path.join(data_path,'mask/')
        self.path_to_describe = [os.path.join(data_path,'des_ali/'),os.path.join(data_path,'des_zhipu/')]
        path_to_train=os.path.join(data_path,'train.txt')
        path_to_val=os.path.join(data_path,'val.txt')
        self.mode=mode
        self.ranks=ranks
        self.text_model, self.text_preprocess = clip.load("ViT-B/32",device="cpu")
        with open(path_to_train,'r') as f:
            ls=f.readlines()
        self.trainlist=[l.rstrip('\n') for l in ls]
        with open(path_to_val,'r') as f:
            ls=f.readlines()
        self.vallist=[l.rstrip('\n') for l in ls]
        if mode=='train':
            self.data=self.trainlist
        if mode=='val':
            self.data=self.vallist
        self.size_list=[]
        self.trans_train = [Compose(
            [
                ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                HorizontalFlip(),
                RandomScale((0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
                RandomCrop(size=[256,256]),
            ]
        ),Compose(
            [
                ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                HorizontalFlip(),
                RandomScale((0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
                RandomCrop(size=[256,128]),
            ]
        ),Compose(
            [
                ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                HorizontalFlip(),
                RandomScale((0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
                RandomCrop(size=[128,256]),
            ]
        )]
        self.argue=argue
        self.random_add_rank = random_add_rank
        self.text_features_cache = {}  # 添加文本特征缓存
        
    def __getitem__(self, item):
        item=item%len(self.data)
        img_path = self.path_to_img + self.data[item]
        label_path = self.path_to_label + self.data[item]
        describe = self.path_to_describe[random.randint(0, len(self.path_to_describe)-1)] + self.data[item].replace('jpg','txt').replace('png','txt').replace('bmp','txt')
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        img = img.resize((self.img_height, self.img_width))
        label = label.resize((self.img_height, self.img_width))
        lines = []
        # 读取描述并在训练时随机选择使用“增强特征(描述)”或“正常特征(ranks)”
        lines_file = []
        try:
            with open(describe, "r", encoding="utf-8") as file:
                lines_file = [line.strip() for line in file.readlines()]
        except:
            lines_file = []
        # 随机交换特征使用增强特征还是正常特征（针对每个值独立随机，而不是整体切换）
        if self.mode == 'train' and len(lines_file) == len(self.ranks):
            lines = []
            for i in range(len(self.ranks)):
                use_aug_text_i = (random.random() < 0.5)
                lines.append(lines_file[i] if use_aug_text_i else self.ranks[i])
        else:
            lines = lines_file if len(lines_file) == len(self.ranks) else self.ranks.copy()

        if self.mode == 'train' and self.random_add_rank:
            if random.random() < 0.5:
                ind = random.randint(0, len(self.random_add_rank)-1)
                lines.append(self.random_add_rank[ind])
            else:
                lines.append("empty")

        # 使用缓存或计算文本特征
        text_features_list = []
        for line in lines:
            if line not in self.text_features_cache:
                text_tokens = clip.tokenize([line])
                with torch.no_grad():
                    text_features = self.text_model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                self.text_features_cache[line] = text_features
            text_features_list.append(self.text_features_cache[line])
        text_features = torch.cat(text_features_list, dim=0)  # 合并为一个张量
        
        # 训练时：随机丢弃一个 rank，将对应的文本特征置为 0（保持长度一致，便于 batch 堆叠）
        drop_idx = None
        if self.mode == 'train' and len(lines) > 1:
            drop_idx = random.randint(0, len(lines) - 1)
            text_features[drop_idx, :] = 0
            

        if self.mode=='train' and self.argue:
            im_lb=dict(im=img,lb=label)
            i=random.randint(0, 2)
            im_lb=self.trans_train[i](im_lb)
            img,label=im_lb['im'],im_lb['lb']
            img = img.resize((self.img_height, self.img_width))
            label = label.resize((self.img_height, self.img_width))
        img = np.array(img)
        label = np.array(label)
        
        # 训练时：将被丢弃 rank 对应的 mask 类别置为 0（假设类别索引为 1..len(ranks)，0 为背景）
        if drop_idx is not None:
            cls_to_zero = drop_idx + 1
            label[label == cls_to_zero] = 0
        
        img=torch.tensor(img).permute(2,0,1).to(dtype=torch.float32)
        label=torch.tensor(label).to(dtype=torch.float32)
        # 定义一个5x5的列表，表示矩阵的数据
        matrix_data = [ [1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0.1, 0.2, 0, 0.7]]

        # 将列表转换为PyTorch张量
        matrix_tensor = torch.tensor(matrix_data)
        return img, label,text_features,matrix_tensor

    def __len__(self):
     return len(self.data)
