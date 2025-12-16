# -*- coding: utf-8 -*-
"""
Infrastructure Layer Package
基础设施层包 - 提供数据库连接和仓储实现
"""

import json
import os

from .mysql import (
    # 值对象
    DatabaseConfig,
    ConnectionPoolConfig,
    
    # 实体
    DatabaseConnection,
    
    # 领域服务
    ConnectionPool,
    
    # 接口
    IRepository,
    IUnitOfWork,
    
    # 实现
    MySQLDatabase,
    BaseRepository,
    MySQLUnitOfWork,
    
    # 工厂
    MySQLDatabaseFactory
)

__all__ = [
    # 值对象
    'DatabaseConfig',
    'ConnectionPoolConfig',
    
    # 实体
    'DatabaseConnection',
    
    # 领域服务
    'ConnectionPool',
    
    # 接口
    'IRepository',
    'IUnitOfWork',
    
    # 实现
    'MySQLDatabase',
    'BaseRepository',
    'MySQLUnitOfWork',
    
    # 工厂
    'MySQLDatabaseFactory'
]

# 定义初始化数据库工厂的全局变量
global medicalDB
medicalDB = None

def initMedicalDB():
    """使用示例初始化医疗数据库"""
    # 从json配置文件加载数据库的信息
    # 获取当前文件的目录，然后构建配置文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', '..', 'config', 'config.json')
    config_path = os.path.normpath(config_path)  # 规范化路径
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    # 取出其中的mysql配置
    mysql_config = config['mysql']

    # 初始化数据库实例
    global medicalDB
    medicalDB = MySQLDatabaseFactory.create_database(
        host=mysql_config['host'],
        port=mysql_config['port'],
        database=mysql_config['database'],
        username=mysql_config['username'],
        password=mysql_config['password'],
        charset=mysql_config['charset']
    )

def getMedicalDBWithTableName(table_name: str):
    if medicalDB is None:
        initMedicalDB()
    repository = MySQLDatabaseFactory.create_repository(medicalDB, table_name)
    return repository
