# -*- coding: utf-8 -*-
"""
Database Configuration Management
数据库配置管理模块
"""

import os
import json
from typing import Dict, Any, Optional
from .mysql import DatabaseConfig, ConnectionPoolConfig


class DatabaseConfigManager:
    """数据库配置管理器"""
    
    @staticmethod
    def from_env() -> DatabaseConfig:
        """从环境变量创建数据库配置"""
        return DatabaseConfig(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '3306')),
            database=os.getenv('DB_NAME', 'test_db'),
            username=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', ''),
            charset=os.getenv('DB_CHARSET', 'utf8mb4')
        )
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> DatabaseConfig:
        """从字典创建数据库配置"""
        return DatabaseConfig(
            host=config_dict.get('host', 'localhost'),
            port=config_dict.get('port', 3306),
            database=config_dict.get('database', 'test_db'),
            username=config_dict.get('username', 'root'),
            password=config_dict.get('password', ''),
            charset=config_dict.get('charset', 'utf8mb4')
        )
    
    @staticmethod
    def from_json_file(file_path: str) -> DatabaseConfig:
        """从JSON文件创建数据库配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return DatabaseConfigManager.from_dict(config_dict)


class ConnectionPoolConfigManager:
    """连接池配置管理器"""
    
    @staticmethod
    def from_env() -> ConnectionPoolConfig:
        """从环境变量创建连接池配置"""
        return ConnectionPoolConfig(
            min_connections=int(os.getenv('DB_POOL_MIN_CONNECTIONS', '5')),
            max_connections=int(os.getenv('DB_POOL_MAX_CONNECTIONS', '20')),
            connection_timeout=int(os.getenv('DB_POOL_TIMEOUT', '30')),
            retry_attempts=int(os.getenv('DB_POOL_RETRY_ATTEMPTS', '3')),
            retry_delay=float(os.getenv('DB_POOL_RETRY_DELAY', '1.0'))
        )
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> ConnectionPoolConfig:
        """从字典创建连接池配置"""
        return ConnectionPoolConfig(
            min_connections=config_dict.get('min_connections', 5),
            max_connections=config_dict.get('max_connections', 20),
            connection_timeout=config_dict.get('connection_timeout', 30),
            retry_attempts=config_dict.get('retry_attempts', 3),
            retry_delay=config_dict.get('retry_delay', 1.0)
        )


# 默认配置实例
DEFAULT_DB_CONFIG = DatabaseConfig(
    host='localhost',
    port=3306,
    database='demo_db',
    username='root',
    password='password'
)

DEFAULT_POOL_CONFIG = ConnectionPoolConfig(
    min_connections=5,
    max_connections=20,
    connection_timeout=30
)