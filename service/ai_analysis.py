# -*- coding: utf-8 -*-
"""
AI Analysis Service
AI分析服务，用于分析相似图像并识别层信息
"""

import json
import logging
import base64
import os
import time
import ssl
import urllib3
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dashscope import MultiModalConversation

from domain.analysis_record import LayerInfo, SimilarImage

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 配置SSL环境变量和设置
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# 创建不验证SSL证书的SSL上下文
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


@dataclass
class AIAnalysisConfig:
    """AI分析配置"""
    api_key: str = "sk-031cbd771af24c07937602182ffe7993"  # 阿里云API密钥
    model_name: str = "qwen3-vl-plus"  # 使用阿里云多模态模型
    enable_thinking: bool = True
    thinking_budget: int = 50
    max_retries: int = 3  # 最大重试次数
    retry_delay: float = 1.0  # 重试延迟（秒）
    timeout: int = 30  # 请求超时时间（秒）


class AIAnalysisService:
    """AI分析服务"""
    
    def __init__(self, config: AIAnalysisConfig = None):
        self.config = config or AIAnalysisConfig()
        self._logger = logging.getLogger(__name__)
    
    def analyze_ultrasound_layers(
        self, 
        original_image_path: str, 
        similar_images: List[SimilarImage]
    ) -> Tuple[List[LayerInfo], Dict[str, int]]:
        """
        分析超声图像的层信息
        
        Args:
            original_image_path: 原始图像路径
            similar_images: 相似图像列表
            
        Returns:
            Tuple[List[LayerInfo], Dict[str, int]]: 检测到的层信息列表和token使用情况
        """
        try:
            # 读取prompt模板
            prompt_text = self._load_prompt_template()
            
            # 构建完整的分析输入，包含相似图像的分析结果
            full_prompt = self._build_analysis_prompt(prompt_text, similar_images)
            
            # 调用阿里云AI模型进行分析
            analysis_result, token_usage = self._call_alibaba_ai_model_with_similar_images(
                full_prompt, original_image_path, similar_images
            )
            
            # 如果AI分析失败，返回默认层信息
            if not analysis_result or analysis_result.strip() == "":
                self._logger.warning("AI分析失败，返回默认层信息")
                return self._get_default_ultrasound_layers(), token_usage
            
            # 解析分析结果
            layers = self._parse_analysis_result(analysis_result)
            
            # 如果解析失败或结果为空，返回默认层信息
            if not layers:
                self._logger.warning("AI分析结果解析失败，返回默认层信息")
                return self._get_default_ultrasound_layers(), token_usage
            
            self._logger.info(f"成功分析图像层信息，检测到 {len(layers)} 个层")
            return layers, token_usage
            
        except Exception as e:
            self._logger.error(f"分析超声图像层结构时发生异常: {e}")
            # 发生异常时返回默认层信息
            return self._get_default_ultrasound_layers(), {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    def _build_analysis_prompt(self, base_prompt: str, similar_images: List[SimilarImage]) -> str:
        """
        构建包含相似图像分析结果的完整prompt
        
        Args:
            base_prompt: 基础prompt模板
            similar_images: 相似图像列表
            
        Returns:
            str: 完整的分析prompt
        """
        # 构建相似图像的分析结果文本
        similar_analyses = []
        for i, similar_image in enumerate(similar_images[:3], 1):  # 最多使用3张相似图像
            if hasattr(similar_image, 'analysis_result') and similar_image.analysis_result:
                similar_analyses.append(f"图像{i}分析：{similar_image.analysis_result}")
            else:
                # 如果没有分析结果，提供基本的层信息
                similar_analyses.append(f"图像{i}分析：{self._generate_sample_analysis()}")
        
        # 组合完整的prompt
        full_prompt = base_prompt
        if similar_analyses:
            full_prompt += "\n\n相似图像的分析结果：\n" + "\n\n".join(similar_analyses)
        
        return full_prompt
    
    def _generate_sample_analysis(self) -> str:
        """生成示例分析结果"""
        sample_analysis = [
            {
                "layer": "皮肤层",
                "exists": True,
                "location": "图像上部",
                "ultrasound_features": "高回声线状结构，边界清晰"
            },
            {
                "layer": "脂肪层", 
                "exists": True,
                "location": "皮肤层下方",
                "ultrasound_features": "低回声区域，回声相对均匀"
            }
        ]
        return json.dumps(sample_analysis, ensure_ascii=False, indent=2)
    
    def _call_alibaba_ai_model_with_similar_images(
        self, prompt: str, original_image_path: str, similar_images: List[SimilarImage]
    ) -> Tuple[str, Dict[str, int]]:
        """
        调用阿里云多模态模型进行分析，包含原始图像和相似图像
        
        Args:
            prompt: 分析提示
            original_image_path: 原始图像路径
            similar_images: 相似图像列表
            
        Returns:
            Tuple[str, Dict[str, int]]: AI分析结果和token使用情况
        """
        for attempt in range(self.config.max_retries):
            try:
                # 编码原始图像
                original_image_base64 = self._encode_image_to_base64(original_image_path)
                
                # 构建消息内容
                content = [{"text": prompt}]
                print("calling prompt\n"+prompt)
                
                # 添加待分析的超声图像（第一张）
                content.append({"image": f"data:image/png;base64,{original_image_base64}"})
                
                # 添加相似图像（最多3张）
                for i, similar_image in enumerate(similar_images[:3]):
                    print(f"相似图片地址:{similar_image.image_path}")
                    if os.path.exists(similar_image.image_path):
                        similar_image_base64 = self._encode_image_to_base64(similar_image.image_path)
                        content.append({"image": f"data:image/png;base64,{similar_image_base64}"})
                
                # 构建消息
                messages = [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
                
                self._logger.info(f"尝试调用阿里云API (第{attempt + 1}次)，包含{len(similar_images[:3])}张相似图像")
                
                # 设置SSL相关的环境变量
                import requests
                from requests.adapters import HTTPAdapter
                from urllib3.util.retry import Retry
                
                # 创建session并配置SSL
                session = requests.Session()
                session.verify = False  # 禁用SSL验证
                
                # 配置重试策略
                retry_strategy = Retry(
                    total=3,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
                )
                adapter = HTTPAdapter(max_retries=retry_strategy)
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                
                # 调用模型
                response = MultiModalConversation.call(
                    api_key=self.config.api_key,
                    model=self.config.model_name,
                    messages=messages,
                    enable_thinking=self.config.enable_thinking,
                    thinking_budget=self.config.thinking_budget,
                )
                
                if response.status_code == 200:
                    # 提取token使用情况
                    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                    if hasattr(response, 'usage') and response.usage:
                        token_usage = {
                            "input_tokens": getattr(response.usage, 'input_tokens', 0),
                            "output_tokens": getattr(response.usage, 'output_tokens', 0),
                            "total_tokens": getattr(response.usage, 'total_tokens', 0)
                        }
                        self._logger.info(f"Token使用情况: {token_usage}")
                    
                    content = response.output.choices[0].message.content[0]
                    if isinstance(content, dict) and "text" in content:
                        self._logger.info("阿里云API调用成功")
                        return content["text"], token_usage
                    elif isinstance(content, str):
                        self._logger.info("阿里云API调用成功")
                        return content, token_usage
                    else:
                        self._logger.info("阿里云API调用成功，转换内容格式")
                        return str(content), token_usage
                else:
                    error_msg = f"阿里云API调用失败: status_code={response.status_code}"
                    if hasattr(response, 'message'):
                        error_msg += f", message={response.message}"
                    self._logger.error(error_msg)
                    
                    # 如果不是最后一次尝试，继续重试
                    if attempt < self.config.max_retries - 1:
                        self._logger.info(f"等待{self.config.retry_delay}秒后重试...")
                        time.sleep(self.config.retry_delay)
                        continue
                    else:
                        return "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                        
            except ssl.SSLError as e:
                error_msg = f"SSL连接错误 (第{attempt + 1}次尝试): {e}"
                self._logger.error(error_msg)
                
                if attempt < self.config.max_retries - 1:
                    self._logger.info(f"SSL错误，等待{self.config.retry_delay * (attempt + 1)}秒后重试...")
                    time.sleep(self.config.retry_delay * (attempt + 1))  # 递增延迟
                    continue
                else:
                    self._logger.error("SSL连接持续失败，已达到最大重试次数")
                    return "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                    
            except Exception as e:
                error_msg = f"阿里云AI模型调用异常 (第{attempt + 1}次尝试): {e}"
                self._logger.error(error_msg)
                
                if attempt < self.config.max_retries - 1:
                    self._logger.info(f"调用异常，等待{self.config.retry_delay}秒后重试...")
                    time.sleep(self.config.retry_delay)
                    continue
                else:
                    self._logger.error("API调用持续失败，已达到最大重试次数")
                    return "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        return "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    def _load_prompt_template(self) -> str:
        """加载提示模板"""
        
        # 读取通用提示文件
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'ultrasound_analysis.txt')
        if os.path.exists(prompt_path):
            try:
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                logging.warning(f"无法读取提示文件: {e}")
        
        # 如果所有文件都不存在，返回一个基本的提示
        logging.error("所有提示文件都不存在，使用基本提示")
        return "请分析超声图像中的层结构，以JSON格式返回结果。"
    
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """将图像文件编码为base64字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _call_alibaba_ai_model(self, prompt: str, image_path: str) -> Tuple[str, Dict[str, int]]:
        """调用阿里云多模态模型进行分析，包含重试机制和错误处理"""
        
        for attempt in range(self.config.max_retries):
            try:
                # 编码图像
                image_base64 = self._encode_image_to_base64(image_path)
                
                # 构建消息
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt},
                            {"image": f"data:image/png;base64,{image_base64}"}
                        ]
                    }
                ]
                
                self._logger.info(f"尝试调用阿里云API (第{attempt + 1}次)")
                
                # 设置SSL相关的环境变量
                import requests
                from requests.adapters import HTTPAdapter
                from urllib3.util.retry import Retry
                
                # 创建session并配置SSL
                session = requests.Session()
                session.verify = False  # 禁用SSL验证
                
                # 配置重试策略
                retry_strategy = Retry(
                    total=3,
                    status_forcelist=[429, 500, 502, 503, 504],
                    method_whitelist=["HEAD", "GET", "OPTIONS", "POST"]
                )
                adapter = HTTPAdapter(max_retries=retry_strategy)
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                
                # 调用模型
                response = MultiModalConversation.call(
                    api_key=self.config.api_key,
                    model=self.config.model_name,
                    messages=messages,
                    enable_thinking=self.config.enable_thinking,
                    thinking_budget=self.config.thinking_budget,
                )
                
                if response.status_code == 200:
                    # 提取token使用情况
                    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                    if hasattr(response, 'usage') and response.usage:
                        token_usage = {
                            "input_tokens": getattr(response.usage, 'input_tokens', 0),
                            "output_tokens": getattr(response.usage, 'output_tokens', 0),
                            "total_tokens": getattr(response.usage, 'total_tokens', 0)
                        }
                        self._logger.info(f"Token使用情况: {token_usage}")
                    
                    content = response.output.choices[0].message.content[0]
                    if isinstance(content, dict) and "text" in content:
                        self._logger.info("阿里云API调用成功")
                        return content["text"], token_usage
                    elif isinstance(content, str):
                        self._logger.info("阿里云API调用成功")
                        return content, token_usage
                    else:
                        self._logger.info("阿里云API调用成功，转换内容格式")
                        return str(content), token_usage
                else:
                    error_msg = f"阿里云API调用失败: status_code={response.status_code}"
                    if hasattr(response, 'message'):
                        error_msg += f", message={response.message}"
                    self._logger.error(error_msg)
                    
                    # 如果不是最后一次尝试，继续重试
                    if attempt < self.config.max_retries - 1:
                        self._logger.info(f"等待{self.config.retry_delay}秒后重试...")
                        time.sleep(self.config.retry_delay)
                        continue
                    else:
                        return "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                        
            except ssl.SSLError as e:
                error_msg = f"SSL连接错误 (第{attempt + 1}次尝试): {e}"
                self._logger.error(error_msg)
                
                if attempt < self.config.max_retries - 1:
                    self._logger.info(f"SSL错误，等待{self.config.retry_delay * (attempt + 1)}秒后重试...")
                    time.sleep(self.config.retry_delay * (attempt + 1))  # 递增延迟
                    continue
                else:
                    self._logger.error("SSL连接持续失败，已达到最大重试次数")
                    return "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                    
            except Exception as e:
                error_msg = f"阿里云AI模型调用异常 (第{attempt + 1}次尝试): {e}"
                self._logger.error(error_msg)
                
                if attempt < self.config.max_retries - 1:
                    self._logger.info(f"调用异常，等待{self.config.retry_delay}秒后重试...")
                    time.sleep(self.config.retry_delay)
                    continue
                else:
                    self._logger.error("API调用持续失败，已达到最大重试次数")
                    return "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        return "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    def _parse_analysis_result(self, analysis_result: str) -> List[LayerInfo]:
        """解析AI分析结果"""
        try:
            # 首先尝试直接解析JSON
            if analysis_result.strip().startswith('[') or analysis_result.strip().startswith('{'):
                data = json.loads(analysis_result)
            else:
                # 尝试从结果中提取JSON
                start_idx = analysis_result.find('[')
                if start_idx == -1:
                    start_idx = analysis_result.find('{')
                
                if start_idx != -1:
                    if analysis_result.strip().startswith('['):
                        end_idx = analysis_result.rfind(']') + 1
                    else:
                        end_idx = analysis_result.rfind('}') + 1
                    
                    if end_idx > start_idx:
                        json_str = analysis_result[start_idx:end_idx]
                        data = json.loads(json_str)
                    else:
                        raise ValueError("无法找到完整的JSON结构")
                else:
                    raise ValueError("无法找到JSON开始标记")
            
            layers = []
            
            # 处理数组格式的结果
            if isinstance(data, list):
                layer_data_list = data
            elif isinstance(data, dict) and 'layers' in data:
                layer_data_list = data['layers']
            else:
                # 如果是单个对象，转换为数组
                layer_data_list = [data]
            
            for layer_data in layer_data_list:
                # 适配新的双语JSON格式
                layer_name = layer_data.get('layer', layer_data.get('layer_name', '甲状腺结节'))
                layer_name_en = layer_data.get('layer_en', '')
                exists = layer_data.get('exists', True)
                location = layer_data.get('location', '')
                location_en = layer_data.get('location_en', '')
                ultrasound_features = layer_data.get('ultrasound_features', layer_data.get('features', {}))
                ultrasound_features_en = layer_data.get('ultrasound_features_en', '')
                
                # 构建中文描述
                if exists:
                    description = f"位置: {location}"
                    if isinstance(ultrasound_features, dict):
                        features_desc = ', '.join([f'{k}: {v}' for k, v in ultrasound_features.items()])
                        description += f", 特征: {features_desc}"
                    elif isinstance(ultrasound_features, str):
                        description += f", 特征: {ultrasound_features}"
                else:
                    description = "未检测到该结构"
                
                # 构建英文描述
                if exists:
                    description_en = f"Location: {location_en}"
                    if ultrasound_features_en:
                        description_en += f", Features: {ultrasound_features_en}"
                else:
                    description_en = "Structure not detected"
                
                layer = LayerInfo(
                    layer_name=layer_name,
                    layer_description=description,
                    confidence=0.9 if exists else 0.1,
                    features=ultrasound_features if isinstance(ultrasound_features, dict) else {},
                    layer_name_en=layer_name_en,
                    layer_description_en=description_en
                )
                layers.append(layer)
            
            return layers if layers else self._get_default_ultrasound_layers()
                
        except json.JSONDecodeError as e:
            self._logger.error(f"解析AI分析结果JSON失败: {e}")
            self._logger.debug(f"原始结果: {analysis_result}")
            return self._get_default_ultrasound_layers()
        except Exception as e:
            self._logger.error(f"解析AI分析结果异常: {e}")
            self._logger.debug(f"原始结果: {analysis_result}")
            return self._get_default_ultrasound_layers()
    
    def _get_default_ultrasound_layers(self) -> List[LayerInfo]:
        """获取默认的乳腺超声层信息"""
        self._logger.info("使用默认的乳腺超声层信息")
        return [
            LayerInfo(
                layer_name="皮肤层",
                layer_description="图像上部的高回声线性结构",
                confidence=0.8,
                features={
                    "echogenicity": "高回声",
                    "location": "图像上部",
                    "boundary": "清晰"
                },
                layer_name_en="Skin Layer",
                layer_description_en="High echogenic linear structure in the upper part of the image"
            ),
            LayerInfo(
                layer_name="脂肪层", 
                layer_description="皮肤层下方的低回声区域",
                confidence=0.8,
                features={
                    "echogenicity": "低回声",
                    "location": "上部至中部",
                    "uniformity": "均匀"
                },
                layer_name_en="Fat Layer",
                layer_description_en="Low echogenic area below the skin layer"
            ),
            LayerInfo(
                layer_name="腺体层",
                layer_description="中等回声的腺体组织",
                confidence=0.7,
                features={
                    "echogenicity": "中等回声",
                    "location": "中部",
                    "texture": "不均匀"
                },
                layer_name_en="Glandular Layer",
                layer_description_en="Medium echogenic glandular tissue"
            ),
            LayerInfo(
                layer_name="肌肉层",
                layer_description="图像下部的低回声肌肉组织",
                confidence=0.7,
                features={
                    "echogenicity": "低回声",
                    "location": "下部",
                    "texture": "相对均匀"
                },
                layer_name_en="Muscle Layer",
                layer_description_en="Low echogenic muscle tissue in the lower part of the image"
            )
        ]
    
    def generate_layer_summary(self, layers: List[LayerInfo]) -> str:
        """生成层信息摘要"""
        if not layers:
            return "未检测到明显的层结构"
        
        summary = f"检测到 {len(layers)} 个可能的解剖层结构：\n\n"
        
        for i, layer in enumerate(layers, 1):
            summary += f"{i}. {layer.layer_name} (置信度: {layer.confidence:.2f})\n"
            summary += f"   描述: {layer.layer_description}\n"
            if layer.features:
                summary += f"   特征: {', '.join([f'{k}: {v}' for k, v in layer.features.items()])}\n"
            summary += "\n"
        
        return summary