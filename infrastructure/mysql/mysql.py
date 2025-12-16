# -*- coding: utf-8 -*-
"""
MySQL Infrastructure Layer - DDD Architecture Implementation
基于领域驱动设计(DDD)的MySQL基础设施层实现
"""

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Generator
import pymysql
from pymysql.cursors import DictCursor
from pymysql.connections import Connection
from pymysql.err import MySQLError
import threading
import time
from queue import Queue, Empty, Full


# ================================
# 值对象 (Value Objects)
# ================================

@dataclass(frozen=True)
class DatabaseConfig:
    """数据库配置值对象"""
    host: str
    port: int
    database: str
    username: str
    password: str
    charset: str = 'utf8mb4'
    autocommit: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.username,
            'password': self.password,
            'charset': self.charset,
            'autocommit': self.autocommit,
            'cursorclass': DictCursor,
            'connect_timeout': 10,  # 连接超时10秒
            'read_timeout': 30,     # 读取超时30秒
            'write_timeout': 30     # 写入超时30秒
        }


@dataclass(frozen=True)
class ConnectionPoolConfig:
    """连接池配置值对象"""
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


# ================================
# 实体 (Entities)
# ================================

class DatabaseConnection:
    """数据库连接实体"""
    
    def __init__(self, connection: Connection, created_at: float):
        self._connection = connection
        self._created_at = created_at
        self._last_used = created_at
        self._is_active = True
    
    @property
    def connection(self) -> Connection:
        return self._connection
    
    @property
    def created_at(self) -> float:
        return self._created_at
    
    @property
    def last_used(self) -> float:
        return self._last_used
    
    @property
    def is_active(self) -> bool:
        """检查连接是否活跃"""
        if not self._is_active:
            return False
        try:
            # 使用ping方法检查连接是否真正可用
            self._connection.ping(reconnect=False)
            return self._connection.open
        except Exception:
            self._is_active = False
            return False
    
    def mark_used(self):
        """标记连接被使用"""
        self._last_used = time.time()
    
    def close(self):
        """关闭连接"""
        if self._connection and self._connection.open:
            self._connection.close()
        self._is_active = False


# ================================
# 领域服务 (Domain Services)
# ================================

class ConnectionPool:
    """数据库连接池领域服务"""
    
    def __init__(self, db_config: DatabaseConfig, pool_config: ConnectionPoolConfig):
        self._db_config = db_config
        self._pool_config = pool_config
        self._pool: Queue[DatabaseConnection] = Queue(maxsize=pool_config.max_connections)
        self._lock = threading.RLock()
        self._current_connections = 0
        self._logger = logging.getLogger(__name__)
        
        # 初始化最小连接数
        self._initialize_pool()
    
    def _initialize_pool(self):
        """初始化连接池"""
        for _ in range(self._pool_config.min_connections):
            try:
                conn = self._create_connection()
                self._pool.put_nowait(conn)
                self._current_connections += 1
            except Exception as e:
                self._logger.error(f"初始化连接池失败: {e}")
                break
    
    def _create_connection(self) -> DatabaseConnection:
        """创建新的数据库连接"""
        retry_count = 0
        while retry_count < self._pool_config.retry_attempts:
            try:
                raw_conn = pymysql.connect(**self._db_config.to_dict())
                return DatabaseConnection(raw_conn, time.time())
            except MySQLError as e:
                retry_count += 1
                self._logger.warning(f"创建数据库连接失败 (尝试 {retry_count}/{self._pool_config.retry_attempts}): {e}")
                if retry_count >= self._pool_config.retry_attempts:
                    self._logger.error(f"创建数据库连接失败，已达到最大重试次数: {e}")
                    raise
                time.sleep(self._pool_config.retry_delay)
    
    def get_connection(self) -> DatabaseConnection:
        """获取数据库连接"""
        with self._lock:
            # 尝试从池中获取连接
            try:
                conn = self._pool.get_nowait()
                if conn.is_active:
                    conn.mark_used()
                    return conn
                else:
                    # 连接已失效，创建新连接
                    self._current_connections -= 1
            except Empty:
                pass
            
            # 如果池中没有可用连接且未达到最大连接数，创建新连接
            if self._current_connections < self._pool_config.max_connections:
                conn = self._create_connection()
                self._current_connections += 1
                conn.mark_used()
                return conn
            
            # 等待连接可用
            try:
                conn = self._pool.get(timeout=self._pool_config.connection_timeout)
                if conn.is_active:
                    conn.mark_used()
                    return conn
                else:
                    self._current_connections -= 1
                    raise MySQLError("获取的连接已失效")
            except Empty:
                raise MySQLError("获取数据库连接超时")
    
    def return_connection(self, conn: DatabaseConnection):
        """归还数据库连接"""
        if conn.is_active:
            try:
                self._pool.put_nowait(conn)
            except Full:
                # 池已满，关闭连接
                conn.close()
                with self._lock:
                    self._current_connections -= 1
        else:
            with self._lock:
                self._current_connections -= 1
    
    def close_all(self):
        """关闭所有连接"""
        with self._lock:
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                except Empty:
                    break
            self._current_connections = 0


# ================================
# 仓储接口 (Repository Interfaces)
# ================================

class IRepository(ABC):
    """仓储基础接口"""
    
    @abstractmethod
    def find_by_id(self, entity_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """根据ID查找实体"""
        pass
    
    @abstractmethod
    def find_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """查找所有实体"""
        pass
    
    @abstractmethod
    def save(self, entity: Dict[str, Any]) -> bool:
        """保存实体"""
        pass
    
    @abstractmethod
    def update(self, entity_id: Union[int, str], entity: Dict[str, Any]) -> bool:
        """更新实体"""
        pass
    
    @abstractmethod
    def delete(self, entity_id: Union[int, str]) -> bool:
        """删除实体"""
        pass


class IUnitOfWork(ABC):
    """工作单元接口"""
    
    @abstractmethod
    def begin(self):
        """开始事务"""
        pass
    
    @abstractmethod
    def commit(self):
        """提交事务"""
        pass
    
    @abstractmethod
    def rollback(self):
        """回滚事务"""
        pass


# ================================
# 基础设施实现 (Infrastructure Implementation)
# ================================

class MySQLDatabase:
    """MySQL数据库基础设施服务"""
    
    def __init__(self, db_config: DatabaseConfig, pool_config: Optional[ConnectionPoolConfig] = None):
        self._db_config = db_config
        self._pool_config = pool_config or ConnectionPoolConfig()
        self._connection_pool = ConnectionPool(db_config, self._pool_config)
        self._logger = logging.getLogger(__name__)
    
    @contextmanager
    def get_connection(self) -> Generator[Connection, None, None]:
        """获取数据库连接上下文管理器"""
        db_conn = None
        try:
            db_conn = self._connection_pool.get_connection()
            yield db_conn.connection
        except Exception as e:
            self._logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            if db_conn:
                self._connection_pool.return_connection(db_conn)
    
    def execute_query(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """执行查询SQL"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.fetchall()
    
    def execute_non_query(self, sql: str, params: Optional[tuple] = None) -> int:
        """执行非查询SQL（INSERT, UPDATE, DELETE）"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                affected_rows = cursor.execute(sql, params)
                conn.commit()
                return affected_rows
    
    def execute_batch(self, sql: str, params_list: List[tuple]) -> int:
        """批量执行SQL"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                affected_rows = cursor.executemany(sql, params_list)
                conn.commit()
                return affected_rows
    
    def close(self):
        """关闭数据库连接池"""
        self._connection_pool.close_all()


class BaseRepository(IRepository):
    """基础仓储实现"""
    
    def __init__(self, database: MySQLDatabase, table_name: str):
        self._database = database
        self._table_name = table_name
        self._logger = logging.getLogger(__name__)
    
    def find_by_id(self, entity_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """根据ID查找实体"""
        sql = f"SELECT * FROM {self._table_name} WHERE id = %s"
        results = self._database.execute_query(sql, (entity_id,))
        return results[0] if results else None
    
    def find_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """查找所有实体"""
        sql = f"SELECT * FROM {self._table_name}"
        params = []
        
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
            
        if offset is not None:
            sql += " OFFSET %s"
            params.append(offset)
        
        return self._database.execute_query(sql, tuple(params) if params else None)
    
    def save(self, entity: Dict[str, Any]) -> bool:
        """保存实体"""
        try:
            columns = ', '.join(entity.keys())
            placeholders = ', '.join(['%s'] * len(entity))
            sql = f"INSERT INTO {self._table_name} ({columns}) VALUES ({placeholders})"
            
            affected_rows = self._database.execute_non_query(sql, tuple(entity.values()))
            return affected_rows > 0
        except Exception as e:
            self._logger.error(f"保存实体失败: {e}")
            return False
    
    def update(self, entity_id: Union[int, str], entity: Dict[str, Any]) -> bool:
        """更新实体"""
        try:
            set_clause = ', '.join([f"{key} = %s" for key in entity.keys()])
            sql = f"UPDATE {self._table_name} SET {set_clause} WHERE id = %s"
            
            params = list(entity.values()) + [entity_id]
            affected_rows = self._database.execute_non_query(sql, tuple(params))
            return affected_rows > 0
        except Exception as e:
            self._logger.error(f"更新实体失败: {e}")
            return False
    
    def delete(self, entity_id: Union[int, str]) -> bool:
        """删除实体"""
        try:
            sql = f"DELETE FROM {self._table_name} WHERE id = %s"
            affected_rows = self._database.execute_non_query(sql, (entity_id,))
            return affected_rows > 0
        except Exception as e:
            self._logger.error(f"删除实体失败: {e}")
            return False


class MySQLUnitOfWork(IUnitOfWork):
    """MySQL工作单元实现"""
    
    def __init__(self, database: MySQLDatabase):
        self._database = database
        self._connection = None
        self._transaction_active = False
    
    def __enter__(self):
        self.begin()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()
    
    def begin(self):
        """开始事务"""
        if not self._transaction_active:
            self._connection = self._database._connection_pool.get_connection()
            self._connection.connection.begin()
            self._transaction_active = True
    
    def commit(self):
        """提交事务"""
        if self._transaction_active and self._connection:
            self._connection.connection.commit()
            self._database._connection_pool.return_connection(self._connection)
            self._transaction_active = False
            self._connection = None
    
    def rollback(self):
        """回滚事务"""
        if self._transaction_active and self._connection:
            self._connection.connection.rollback()
            self._database._connection_pool.return_connection(self._connection)
            self._transaction_active = False
            self._connection = None


# ================================
# 工厂类 (Factory)
# ================================

class MySQLDatabaseFactory:
    """MySQL数据库工厂类"""
    
    @staticmethod
    def create_database(
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        charset: str = 'utf8mb4',
        min_connections: int = 5,
        max_connections: int = 20
    ) -> MySQLDatabase:
        """创建MySQL数据库实例"""
        db_config = DatabaseConfig(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            charset=charset
        )
        
        pool_config = ConnectionPoolConfig(
            min_connections=min_connections,
            max_connections=max_connections
        )
        
        return MySQLDatabase(db_config, pool_config)
    
    @staticmethod
    def create_repository(database: MySQLDatabase, table_name: str) -> BaseRepository:
        """创建基础仓储实例"""
        return BaseRepository(database, table_name)



if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    pass