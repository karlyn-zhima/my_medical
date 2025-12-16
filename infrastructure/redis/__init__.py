# -*- coding: utf-8 -*-
"""
Redis Infrastructure Module
Redis基础设施模块

提供Redis连接、配置管理和仓储实现
"""

import json
import os

from .redis import (
    # 配置类
    RedisConfig,
    RedisPoolConfig,
    
    # 连接和客户端
    RedisConnection,
    RedisClient,
    RedisConnectionPool,
    
    # 仓储接口
    IRedisRepository,
    IRedisHashRepository,
    IRedisListRepository,
    IRedisSetRepository,
    
    # 仓储实现
    RedisRepository,
    RedisHashRepository,
    RedisListRepository,
    RedisSetRepository,
    
    # 工厂类
    RedisFactory
)

from .config import (
    # 配置管理器
    RedisConfigManager,
    RedisPoolConfigManager,
    RedisConfigFactory
)

# 版本信息
__version__ = '1.0.0'
__author__ = 'Infrastructure Team'

# 导出的公共接口
__all__ = [
    # 配置类
    'RedisConfig',
    'RedisPoolConfig',
    
    # 连接和客户端
    'RedisConnection',
    'RedisClient',
    'RedisConnectionPool',
    
    # 仓储接口
    'IRedisRepository',
    'IRedisHashRepository',
    'IRedisListRepository',
    'IRedisSetRepository',
    
    # 仓储实现
    'RedisRepository',
    'RedisHashRepository',
    'RedisListRepository',
    'RedisSetRepository',
    
    # 工厂类
    'RedisFactory',
    
    # 配置管理器
    'RedisConfigManager',
    'RedisPoolConfigManager',
    'RedisConfigFactory',
    
    # 全局客户端获取方法
    'getRedisClient',
    'getRedisRepository',
    'getRedisHashRepository',
    'getRedisListRepository',
    'getRedisSetRepository'
]

# 定义全局Redis客户端变量
global redisClient
redisClient = None

def initRedisClient():
    """初始化Redis客户端"""
    # 从json配置文件加载Redis的信息
    # 获取当前文件的目录，然后构建配置文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', '..', 'config', 'config.json')
    config_path = os.path.normpath(config_path)  # 规范化路径
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # 取出其中的redis配置
        redis_config = config.get('redis', {})
        
        # 使用配置文件中的Redis配置
        config_obj = RedisConfig(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            password=redis_config.get('password', None),
            decode_responses=redis_config.get('decode_responses', True)
        )
        
        pool_config = RedisPoolConfig(
            max_connections=redis_config.get('max_connections', 10),
            retry_on_timeout=redis_config.get('retry_on_timeout', True),
            health_check_interval=redis_config.get('health_check_interval', 30)
        )
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        # 如果配置文件不存在或格式错误，使用默认配置
        config_obj = RedisConfigFactory.create_default_config()
        pool_config = RedisConfigFactory.create_default_pool_config()
    
    # 初始化Redis客户端实例
    global redisClient
    redisClient = RedisFactory.create_client(config_obj, pool_config)

def getRedisClient() -> RedisClient:
    """获取Redis客户端，支持懒加载初始化"""
    if redisClient is None:
        initRedisClient()
    return redisClient

def getRedisRepository() -> RedisRepository:
    """获取基础Redis仓储"""
    client = getRedisClient()
    return RedisFactory.create_repository(client)

def getRedisHashRepository() -> RedisHashRepository:
    """获取Redis Hash仓储"""
    client = getRedisClient()
    return RedisFactory.create_hash_repository(client)

def getRedisListRepository() -> RedisListRepository:
    """获取Redis List仓储"""
    client = getRedisClient()
    return RedisFactory.create_list_repository(client)

def getRedisSetRepository() -> RedisSetRepository:
    """获取Redis Set仓储"""
    client = getRedisClient()
    return RedisFactory.create_set_repository(client)

# 保留原有的工厂方法以保持向后兼容性
def create_default_redis_client() -> RedisClient:
    """创建默认的Redis客户端"""
    config = RedisConfigFactory.create_default_config()
    pool_config = RedisConfigFactory.create_default_pool_config()
    return RedisFactory.create_client(config, pool_config)

def create_redis_client_from_env() -> RedisClient:
    """从环境变量创建Redis客户端"""
    config = RedisConfigManager.from_env()
    pool_config = RedisPoolConfigManager.from_env()
    return RedisFactory.create_client(config, pool_config)

def create_redis_repositories(client: RedisClient) -> dict:
    """创建所有类型的Redis仓储"""
    return {
        'basic': RedisFactory.create_repository(client),
        'hash': RedisFactory.create_hash_repository(client),
        'list': RedisFactory.create_list_repository(client),
        'set': RedisFactory.create_set_repository(client)
    }