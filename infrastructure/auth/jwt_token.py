"""
JWT Token 加密组件
提供安全的token生成、验证和解析功能
"""

import jwt
import datetime
from typing import Dict, Any, Optional
import os
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import json


class JWTTokenManager:
    """JWT Token 管理器"""
    
    def __init__(self, secret_key: Optional[str] = None, algorithm: str = "HS256"):
        """
        初始化JWT Token管理器
        
        Args:
            secret_key: 密钥，如果为None则从环境变量或配置文件获取
            algorithm: 加密算法，默认HS256
        """
        self.algorithm = algorithm
        self.secret_key = secret_key or self._get_secret_key()
        
        # Token过期时间配置
        self.access_token_expire_hours = 24  # 访问token 24小时过期
        self.refresh_token_expire_days = 7   # 刷新token 7天过期
    
    def _get_secret_key(self) -> str:
        """获取密钥"""
        # 优先从环境变量获取
        secret = os.getenv('JWT_SECRET_KEY')
        if secret:
            return secret
            
        # 从配置文件获取
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('jwt_secret_key', 'default_secret_key_change_in_production')
        except:
            # 默认密钥（生产环境中应该更改）
            return 'default_secret_key_change_in_production_2024'
    
    def generate_access_token(self, user_data: Dict[str, Any]) -> str:
        """
        生成访问token
        
        Args:
            user_data: 用户数据字典，包含用户ID等信息
            
        Returns:
            JWT token字符串
        """
        now = datetime.datetime.utcnow()
        payload = {
            'user_id': user_data.get('id') or user_data.get('uid'),
            'username': user_data.get('username'),
            'email': user_data.get('email'),
            'iat': now,  # 签发时间
            'exp': now + datetime.timedelta(hours=self.access_token_expire_hours),  # 过期时间
            'type': 'access'  # token类型
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def generate_refresh_token(self, user_id: str) -> str:
        """
        生成刷新token
        
        Args:
            user_id: 用户ID
            
        Returns:
            JWT refresh token字符串
        """
        now = datetime.datetime.utcnow()
        payload = {
            'user_id': user_id,
            'iat': now,
            'exp': now + datetime.timedelta(days=self.refresh_token_expire_days),
            'type': 'refresh'
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        验证token并返回payload
        
        Args:
            token: JWT token字符串
            
        Returns:
            解析后的payload字典，验证失败返回None
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            # Token已过期
            return None
        except jwt.InvalidTokenError:
            # Token无效
            return None
    
    def is_token_expired(self, token: str) -> bool:
        """
        检查token是否过期
        
        Args:
            token: JWT token字符串
            
        Returns:
            True表示已过期，False表示未过期
        """
        try:
            jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return False
        except jwt.ExpiredSignatureError:
            return True
        except jwt.InvalidTokenError:
            return True
    
    def get_user_id_from_token(self, token: str) -> Optional[str]:
        """
        从token中获取用户ID
        
        Args:
            token: JWT token字符串
            
        Returns:
            用户ID，获取失败返回None
        """
        payload = self.verify_token(token)
        if payload:
            return payload.get('user_id')
        return None
    
    def refresh_access_token(self, refresh_token: str, user_data: Dict[str, Any]) -> Optional[str]:
        """
        使用刷新token生成新的访问token
        
        Args:
            refresh_token: 刷新token
            user_data: 用户数据
            
        Returns:
            新的访问token，失败返回None
        """
        payload = self.verify_token(refresh_token)
        if payload and payload.get('type') == 'refresh':
            return self.generate_access_token(user_data)
        return None


# 全局token管理器实例
jwt_manager = JWTTokenManager()


def generate_token(user_data: Dict[str, Any]) -> Dict[str, str]:
    """
    生成token对（访问token和刷新token）
    
    Args:
        user_data: 用户数据
        
    Returns:
        包含access_token和refresh_token的字典
    """
    user_id = str(user_data.get('id') or user_data.get('uid'))
    
    return {
        'access_token': jwt_manager.generate_access_token(user_data),
        'refresh_token': jwt_manager.generate_refresh_token(user_id)
    }


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    验证token
    
    Args:
        token: JWT token字符串
        
    Returns:
        解析后的payload，验证失败返回None
    """
    return jwt_manager.verify_token(token)


def get_user_id_from_token(token: str) -> Optional[str]:
    """
    从token中获取用户ID
    
    Args:
        token: JWT token字符串
        
    Returns:
        用户ID字符串，如果token无效则返回None
    """
    payload = verify_token(token)
    if payload:
        return str(payload.get('user_id'))
    return None