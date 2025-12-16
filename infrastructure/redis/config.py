# -*- coding: utf-8 -*-
"""
Redis Configuration Management
Redis配置管理模块
"""

import os
import json
from typing import Dict, Any, Optional
from .redis import RedisConfig, RedisPoolConfig


class RedisConfigManager:
    """Redis配置管理器"""
    
    @staticmethod
    def from_env() -> RedisConfig:
        """从环境变量创建Redis配置"""
        return RedisConfig(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            db=int(os.getenv('REDIS_DB', '0')),
            password=os.getenv('REDIS_PASSWORD', None),
            decode_responses=os.getenv('REDIS_DECODE_RESPONSES', 'true').lower() == 'true',
            socket_timeout=float(os.getenv('REDIS_SOCKET_TIMEOUT', '5.0')),
            socket_connect_timeout=float(os.getenv('REDIS_SOCKET_CONNECT_TIMEOUT', '5.0')),
            socket_keepalive=os.getenv('REDIS_SOCKET_KEEPALIVE', 'true').lower() == 'true',
            socket_keepalive_options={}
        )
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> RedisConfig:
        """从字典创建Redis配置"""
        return RedisConfig(
            host=config_dict.get('host', 'localhost'),
            port=config_dict.get('port', 6379),
            db=config_dict.get('db', 0),
            password=config_dict.get('password', None),
            decode_responses=config_dict.get('decode_responses', True),
            socket_timeout=config_dict.get('socket_timeout', 5.0),
            socket_connect_timeout=config_dict.get('socket_connect_timeout', 5.0),
            socket_keepalive=config_dict.get('socket_keepalive', True),
            socket_keepalive_options=config_dict.get('socket_keepalive_options', {})
        )
    
    @staticmethod
    def from_json_file(file_path: str) -> RedisConfig:
        """从JSON文件创建Redis配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return RedisConfigManager.from_dict(config_dict)


class RedisPoolConfigManager:
    """Redis连接池配置管理器"""
    
    @staticmethod
    def from_env() -> RedisPoolConfig:
        """从环境变量创建连接池配置"""
        return RedisPoolConfig(
            max_connections=int(os.getenv('REDIS_POOL_MAX_CONNECTIONS', '50')),
            retry_on_timeout=os.getenv('REDIS_POOL_RETRY_ON_TIMEOUT', 'true').lower() == 'true',
            health_check_interval=int(os.getenv('REDIS_POOL_HEALTH_CHECK_INTERVAL', '30')),
            socket_keepalive=os.getenv('REDIS_POOL_SOCKET_KEEPALIVE', 'true').lower() == 'true',
            socket_keepalive_options={}
        )
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> RedisPoolConfig:
        """从字典创建连接池配置"""
        return RedisPoolConfig(
            max_connections=config_dict.get('max_connections', 50),
            retry_on_timeout=config_dict.get('retry_on_timeout', True),
            health_check_interval=config_dict.get('health_check_interval', 30),
            socket_keepalive=config_dict.get('socket_keepalive', True),
            socket_keepalive_options=config_dict.get('socket_keepalive_options', {})
        )
    
    @staticmethod
    def from_json_file(file_path: str) -> RedisPoolConfig:
        """从JSON文件创建连接池配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return RedisPoolConfigManager.from_dict(config_dict)


class RedisConfigFactory:
    """Redis配置工厂类"""
    
    @staticmethod
    def create_default_config() -> RedisConfig:
        """创建默认配置"""
        return RedisConfig(
            host='localhost',
            port=6379,
            db=0,
            password=None,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            socket_keepalive=True,
            socket_keepalive_options={}
        )
    
    @staticmethod
    def create_default_pool_config() -> RedisPoolConfig:
        """创建默认连接池配置"""
        return RedisPoolConfig(
            max_connections=50,
            retry_on_timeout=True,
            health_check_interval=30,
            socket_keepalive=True,
            socket_keepalive_options={}
        )