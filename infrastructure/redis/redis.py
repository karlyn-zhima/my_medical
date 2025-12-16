# -*- coding: utf-8 -*-
"""
Redis Infrastructure Layer - DDD Architecture Implementation
基于领域驱动设计(DDD)的Redis基础设施层实现
"""

import logging
import json
import pickle
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Generator, Callable
import redis
from redis.connection import ConnectionPool
from redis.exceptions import RedisError, ConnectionError, TimeoutError
import threading
import time


# ================================
# 值对象 (Value Objects)
# ================================

@dataclass(frozen=True)
class RedisConfig:
    """Redis配置值对象"""
    host: str
    port: int
    db: int = 0
    password: Optional[str] = None
    decode_responses: bool = True
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.socket_keepalive_options is None:
            object.__setattr__(self, 'socket_keepalive_options', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        config = {
            'host': self.host,
            'port': self.port,
            'db': self.db,
            'decode_responses': self.decode_responses,
            'socket_timeout': self.socket_timeout,
            'socket_connect_timeout': self.socket_connect_timeout,
            'socket_keepalive': self.socket_keepalive,
            'socket_keepalive_options': self.socket_keepalive_options
        }
        if self.password:
            config['password'] = self.password
        return config


@dataclass(frozen=True)
class RedisPoolConfig:
    """Redis连接池配置值对象"""
    max_connections: int = 50
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.socket_keepalive_options is None:
            object.__setattr__(self, 'socket_keepalive_options', {})


# ================================
# 实体 (Entities)
# ================================

class RedisConnection:
    """Redis连接实体"""
    
    def __init__(self, connection: redis.Redis):
        self._connection = connection
        self._created_at = time.time()
        self._last_used = time.time()
    
    @property
    def connection(self) -> redis.Redis:
        """获取Redis连接"""
        self._last_used = time.time()
        return self._connection
    
    @property
    def created_at(self) -> float:
        """获取创建时间"""
        return self._created_at
    
    @property
    def last_used(self) -> float:
        """获取最后使用时间"""
        return self._last_used
    
    def is_alive(self) -> bool:
        """检查连接是否存活"""
        try:
            self._connection.ping()
            return True
        except RedisError:
            return False


# ================================
# 仓储接口 (Repository Interfaces)
# ================================

class IRedisRepository(ABC):
    """Redis仓储接口"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取值"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """设置值"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除键"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        pass
    
    @abstractmethod
    def expire(self, key: str, time: int) -> bool:
        """设置过期时间"""
        pass
    
    @abstractmethod
    def ttl(self, key: str) -> int:
        """获取剩余生存时间"""
        pass


class IRedisHashRepository(ABC):
    """Redis哈希仓储接口"""
    
    @abstractmethod
    def hget(self, name: str, key: str) -> Optional[Any]:
        """获取哈希字段值"""
        pass
    
    @abstractmethod
    def hset(self, name: str, key: str, value: Any) -> bool:
        """设置哈希字段值"""
        pass
    
    @abstractmethod
    def hdel(self, name: str, key: str) -> bool:
        """删除哈希字段"""
        pass
    
    @abstractmethod
    def hgetall(self, name: str) -> Dict[str, Any]:
        """获取所有哈希字段"""
        pass


class IRedisListRepository(ABC):
    """Redis列表仓储接口"""
    
    @abstractmethod
    def lpush(self, name: str, *values: Any) -> int:
        """从左侧推入值"""
        pass
    
    @abstractmethod
    def rpush(self, name: str, *values: Any) -> int:
        """从右侧推入值"""
        pass
    
    @abstractmethod
    def lpop(self, name: str) -> Optional[Any]:
        """从左侧弹出值"""
        pass
    
    @abstractmethod
    def rpop(self, name: str) -> Optional[Any]:
        """从右侧弹出值"""
        pass
    
    @abstractmethod
    def llen(self, name: str) -> int:
        """获取列表长度"""
        pass


class IRedisSetRepository(ABC):
    """Redis集合仓储接口"""
    
    @abstractmethod
    def sadd(self, name: str, *values: Any) -> int:
        """添加成员到集合"""
        pass
    
    @abstractmethod
    def srem(self, name: str, *values: Any) -> int:
        """从集合移除成员"""
        pass
    
    @abstractmethod
    def smembers(self, name: str) -> set:
        """获取集合所有成员"""
        pass
    
    @abstractmethod
    def sismember(self, name: str, value: Any) -> bool:
        """检查是否为集合成员"""
        pass


# ================================
# 基础设施实现 (Infrastructure Implementation)
# ================================

class RedisConnectionPool:
    """Redis连接池"""
    
    def __init__(self, config: RedisConfig, pool_config: RedisPoolConfig):
        self._config = config
        self._pool_config = pool_config
        self._pool = ConnectionPool(**config.to_dict(), max_connections=pool_config.max_connections)
        self._logger = logging.getLogger(__name__)
    
    def get_connection(self) -> RedisConnection:
        """获取连接"""
        try:
            redis_client = redis.Redis(connection_pool=self._pool)
            return RedisConnection(redis_client)
        except RedisError as e:
            self._logger.error(f"获取Redis连接失败: {e}")
            raise
    
    def close(self):
        """关闭连接池"""
        try:
            self._pool.disconnect()
        except Exception as e:
            self._logger.error(f"关闭Redis连接池失败: {e}")


class RedisClient:
    """Redis客户端"""
    
    def __init__(self, config: RedisConfig, pool_config: Optional[RedisPoolConfig] = None):
        self._config = config
        self._pool_config = pool_config or RedisPoolConfig()
        self._connection_pool = RedisConnectionPool(config, self._pool_config)
        self._logger = logging.getLogger(__name__)
    
    @contextmanager
    def get_connection(self) -> Generator[redis.Redis, None, None]:
        """获取连接上下文管理器"""
        connection = None
        try:
            redis_connection = self._connection_pool.get_connection()
            connection = redis_connection.connection
            yield connection
        except RedisError as e:
            self._logger.error(f"Redis操作失败: {e}")
            raise
        finally:
            # Redis连接会自动返回到连接池，无需手动关闭
            pass
    
    def ping(self) -> bool:
        """测试连接"""
        try:
            with self.get_connection() as conn:
                return conn.ping()
        except RedisError:
            return False
    
    def close(self):
        """关闭客户端"""
        self._connection_pool.close()


class RedisRepository(IRedisRepository):
    """Redis仓储实现"""
    
    def __init__(self, client: RedisClient, serializer: Optional[Callable] = None, deserializer: Optional[Callable] = None):
        self._client = client
        self._serializer = serializer or self._default_serializer
        self._deserializer = deserializer or self._default_deserializer
        self._logger = logging.getLogger(__name__)
    
    def _default_serializer(self, value: Any) -> str:
        """默认序列化器"""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        return json.dumps(value, ensure_ascii=False)
    
    def _default_deserializer(self, value: str) -> Any:
        """默认反序列化器"""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    
    def get(self, key: str) -> Optional[Any]:
        """获取值"""
        try:
            with self._client.get_connection() as conn:
                value = conn.get(key)
                return self._deserializer(value) if value is not None else None
        except RedisError as e:
            self._logger.error(f"获取键 {key} 失败: {e}")
            return None
    
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """设置值"""
        try:
            with self._client.get_connection() as conn:
                serialized_value = self._serializer(value)
                return conn.set(key, serialized_value, ex=ex)
        except RedisError as e:
            self._logger.error(f"设置键 {key} 失败: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """删除键"""
        try:
            with self._client.get_connection() as conn:
                return bool(conn.delete(key))
        except RedisError as e:
            self._logger.error(f"删除键 {key} 失败: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            with self._client.get_connection() as conn:
                return bool(conn.exists(key))
        except RedisError as e:
            self._logger.error(f"检查键 {key} 存在性失败: {e}")
            return False
    
    def expire(self, key: str, time: int) -> bool:
        """设置过期时间"""
        try:
            with self._client.get_connection() as conn:
                return conn.expire(key, time)
        except RedisError as e:
            self._logger.error(f"设置键 {key} 过期时间失败: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """获取剩余生存时间"""
        try:
            with self._client.get_connection() as conn:
                return conn.ttl(key)
        except RedisError as e:
            self._logger.error(f"获取键 {key} TTL失败: {e}")
            return -1


class RedisHashRepository(IRedisHashRepository):
    """Redis哈希仓储实现"""
    
    def __init__(self, client: RedisClient, serializer: Optional[Callable] = None, deserializer: Optional[Callable] = None):
        self._client = client
        self._serializer = serializer or json.dumps
        self._deserializer = deserializer or json.loads
        self._logger = logging.getLogger(__name__)
    
    def hget(self, name: str, key: str) -> Optional[Any]:
        """获取哈希字段值"""
        try:
            with self._client.get_connection() as conn:
                value = conn.hget(name, key)
                return self._deserializer(value) if value is not None else None
        except (RedisError, json.JSONDecodeError) as e:
            self._logger.error(f"获取哈希 {name} 字段 {key} 失败: {e}")
            return None
    
    def hset(self, name: str, key: str, value: Any) -> bool:
        """设置哈希字段值"""
        try:
            with self._client.get_connection() as conn:
                serialized_value = self._serializer(value)
                return bool(conn.hset(name, key, serialized_value))
        except (RedisError, TypeError) as e:
            self._logger.error(f"设置哈希 {name} 字段 {key} 失败: {e}")
            return False
    
    def hdel(self, name: str, key: str) -> bool:
        """删除哈希字段"""
        try:
            with self._client.get_connection() as conn:
                return bool(conn.hdel(name, key))
        except RedisError as e:
            self._logger.error(f"删除哈希 {name} 字段 {key} 失败: {e}")
            return False
    
    def hgetall(self, name: str) -> Dict[str, Any]:
        """获取所有哈希字段"""
        try:
            with self._client.get_connection() as conn:
                data = conn.hgetall(name)
                return {k: self._deserializer(v) for k, v in data.items()}
        except (RedisError, json.JSONDecodeError) as e:
            self._logger.error(f"获取哈希 {name} 所有字段失败: {e}")
            return {}


class RedisListRepository(IRedisListRepository):
    """Redis列表仓储实现"""
    
    def __init__(self, client: RedisClient, serializer: Optional[Callable] = None, deserializer: Optional[Callable] = None):
        self._client = client
        self._serializer = serializer or json.dumps
        self._deserializer = deserializer or json.loads
        self._logger = logging.getLogger(__name__)
    
    def lpush(self, name: str, *values: Any) -> int:
        """从左侧推入值"""
        try:
            with self._client.get_connection() as conn:
                serialized_values = [self._serializer(v) for v in values]
                return conn.lpush(name, *serialized_values)
        except (RedisError, TypeError) as e:
            self._logger.error(f"左推入列表 {name} 失败: {e}")
            return 0
    
    def rpush(self, name: str, *values: Any) -> int:
        """从右侧推入值"""
        try:
            with self._client.get_connection() as conn:
                serialized_values = [self._serializer(v) for v in values]
                return conn.rpush(name, *serialized_values)
        except (RedisError, TypeError) as e:
            self._logger.error(f"右推入列表 {name} 失败: {e}")
            return 0
    
    def lpop(self, name: str) -> Optional[Any]:
        """从左侧弹出值"""
        try:
            with self._client.get_connection() as conn:
                value = conn.lpop(name)
                return self._deserializer(value) if value is not None else None
        except (RedisError, json.JSONDecodeError) as e:
            self._logger.error(f"左弹出列表 {name} 失败: {e}")
            return None
    
    def rpop(self, name: str) -> Optional[Any]:
        """从右侧弹出值"""
        try:
            with self._client.get_connection() as conn:
                value = conn.rpop(name)
                return self._deserializer(value) if value is not None else None
        except (RedisError, json.JSONDecodeError) as e:
            self._logger.error(f"右弹出列表 {name} 失败: {e}")
            return None
    
    def llen(self, name: str) -> int:
        """获取列表长度"""
        try:
            with self._client.get_connection() as conn:
                return conn.llen(name)
        except RedisError as e:
            self._logger.error(f"获取列表 {name} 长度失败: {e}")
            return 0


class RedisSetRepository(IRedisSetRepository):
    """Redis集合仓储实现"""
    
    def __init__(self, client: RedisClient, serializer: Optional[Callable] = None, deserializer: Optional[Callable] = None):
        self._client = client
        self._serializer = serializer or json.dumps
        self._deserializer = deserializer or json.loads
        self._logger = logging.getLogger(__name__)
    
    def sadd(self, name: str, *values: Any) -> int:
        """添加成员到集合"""
        try:
            with self._client.get_connection() as conn:
                serialized_values = [self._serializer(v) for v in values]
                return conn.sadd(name, *serialized_values)
        except (RedisError, TypeError) as e:
            self._logger.error(f"添加成员到集合 {name} 失败: {e}")
            return 0
    
    def srem(self, name: str, *values: Any) -> int:
        """从集合移除成员"""
        try:
            with self._client.get_connection() as conn:
                serialized_values = [self._serializer(v) for v in values]
                return conn.srem(name, *serialized_values)
        except (RedisError, TypeError) as e:
            self._logger.error(f"从集合 {name} 移除成员失败: {e}")
            return 0
    
    def smembers(self, name: str) -> set:
        """获取集合所有成员"""
        try:
            with self._client.get_connection() as conn:
                members = conn.smembers(name)
                return {self._deserializer(m) for m in members}
        except (RedisError, json.JSONDecodeError) as e:
            self._logger.error(f"获取集合 {name} 成员失败: {e}")
            return set()
    
    def sismember(self, name: str, value: Any) -> bool:
        """检查是否为集合成员"""
        try:
            with self._client.get_connection() as conn:
                serialized_value = self._serializer(value)
                return conn.sismember(name, serialized_value)
        except (RedisError, TypeError) as e:
            self._logger.error(f"检查集合 {name} 成员失败: {e}")
            return False


# ================================
# 工厂类 (Factory Classes)
# ================================

class RedisFactory:
    """Redis工厂类"""
    
    @staticmethod
    def create_client(config: RedisConfig, pool_config: Optional[RedisPoolConfig] = None) -> RedisClient:
        """创建Redis客户端"""
        return RedisClient(config, pool_config)
    
    @staticmethod
    def create_repository(client: RedisClient) -> RedisRepository:
        """创建Redis仓储"""
        return RedisRepository(client)
    
    @staticmethod
    def create_hash_repository(client: RedisClient) -> RedisHashRepository:
        """创建Redis哈希仓储"""
        return RedisHashRepository(client)
    
    @staticmethod
    def create_list_repository(client: RedisClient) -> RedisListRepository:
        """创建Redis列表仓储"""
        return RedisListRepository(client)
    
    @staticmethod
    def create_set_repository(client: RedisClient) -> RedisSetRepository:
        """创建Redis集合仓储"""
        return RedisSetRepository(client)