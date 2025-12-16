# -*- coding: utf-8 -*-
"""
Segment Record Repository
分割记录仓储实现
"""

from typing import List, Optional
from datetime import datetime
import logging

from infrastructure.mysql.mysql import BaseRepository, MySQLDatabase
from domain.segment_record import SegmentRecord


class SegmentRecordRepository(BaseRepository):
    """分割记录仓储"""
    
    def __init__(self, database: MySQLDatabase):
        super().__init__(database, 'segment_records')
        self._logger = logging.getLogger(__name__)
    
    def create_table_if_not_exists(self):
        """创建表（如果不存在）"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS segment_records (
            id INT AUTO_INCREMENT PRIMARY KEY,
            original_image_path VARCHAR(500) NOT NULL,
            original_image_hash VARCHAR(64) NOT NULL,
            selected_layers TEXT,
            segment_result_path VARCHAR(500),
            status VARCHAR(20) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            user_id INT,
            analysis_record_id INT,
            INDEX idx_image_hash (original_image_hash),
            INDEX idx_status (status),
            INDEX idx_user_id (user_id),
            INDEX idx_created_at (created_at),
            INDEX idx_analysis_record_id (analysis_record_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        try:
            self._database.execute_non_query(create_table_sql)
            self._logger.info("分割记录表创建成功或已存在")
        except Exception as e:
            self._logger.error(f"创建分割记录表失败: {e}")
            raise
    
    def save_record(self, record: SegmentRecord) -> int:
        """保存分割记录"""
        if record.id is None:
            # 插入新记录 - 使用同一连接确保LAST_INSERT_ID正确
            insert_sql = """
            INSERT INTO segment_records (
                original_image_path, original_image_hash, selected_layers,
                segment_result_path, status, user_id, analysis_record_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            record_dict = record.to_dict()
            params = (
                record_dict['original_image_path'],
                record_dict['original_image_hash'],
                record_dict['selected_layers'],
                record_dict['segment_result_path'],
                record_dict['status'],
                record_dict['user_id'],
                record_dict['analysis_record_id']
            )
            
            # 使用同一连接执行INSERT和LAST_INSERT_ID查询
            with self._database.get_connection() as conn:
                with conn.cursor() as cursor:
                    affected_rows = cursor.execute(insert_sql, params)
                    conn.commit()
                    
                    if affected_rows > 0:
                        # 在同一连接中获取插入的ID
                        cursor.execute("SELECT LAST_INSERT_ID() as id")
                        id_result = cursor.fetchall()
                        if id_result:
                            record.id = id_result[0]['id']
                            return record.id
            return 0
        else:
            # 更新现有记录
            return self.update_record(record)
    
    def update_record(self, record: SegmentRecord) -> int:
        """更新分割记录"""
        update_sql = """
        UPDATE segment_records SET
            original_image_path = %s,
            original_image_hash = %s,
            selected_layers = %s,
            segment_result_path = %s,
            status = %s,
            updated_at = CURRENT_TIMESTAMP,
            user_id = %s,
            analysis_record_id = %s
        WHERE id = %s
        """
        
        record_dict = record.to_dict()
        params = (
            record_dict['original_image_path'],
            record_dict['original_image_hash'],
            record_dict['selected_layers'],
            record_dict['segment_result_path'],
            record_dict['status'],
            record_dict['user_id'],
            record_dict['analysis_record_id'],
            record.id
        )
        
        return self._database.execute_non_query(update_sql, params)
    
    def find_by_id(self, record_id: int) -> Optional[SegmentRecord]:
        """根据ID查找记录"""
        sql = "SELECT * FROM segment_records WHERE id = %s"
        results = self._database.execute_query(sql, (record_id,))
        
        if results:
            return SegmentRecord.from_dict(results[0])
        return None
    
    def find_by_image_hash(self, image_hash: str) -> Optional[SegmentRecord]:
        """根据图像哈希查找记录"""
        sql = "SELECT * FROM segment_records WHERE original_image_hash = %s ORDER BY created_at DESC LIMIT 1"
        results = self._database.execute_query(sql, (image_hash,))
        
        if results:
            return SegmentRecord.from_dict(results[0])
        return None
    
    def find_by_user_id(self, user_id: int, limit: int = 50, offset: int = 0) -> List[SegmentRecord]:
        """根据用户ID查找记录"""
        sql = """
        SELECT * FROM segment_records 
        WHERE user_id = %s 
        ORDER BY created_at DESC 
        LIMIT %s OFFSET %s
        """
        results = self._database.execute_query(sql, (user_id, limit, offset))
        
        return [SegmentRecord.from_dict(row) for row in results]
    
    def find_by_status(self, status: str, limit: int = 50) -> List[SegmentRecord]:
        """根据状态查找记录"""
        sql = """
        SELECT * FROM segment_records 
        WHERE status = %s 
        ORDER BY created_at DESC 
        LIMIT %s
        """
        results = self._database.execute_query(sql, (status, limit))
        
        return [SegmentRecord.from_dict(row) for row in results]
    
    def update_status(self, record_id: int, status: str) -> int:
        """更新记录状态"""
        sql = """
        UPDATE segment_records 
        SET status = %s, updated_at = CURRENT_TIMESTAMP 
        WHERE id = %s
        """
        return self._database.execute_non_query(sql, (status, record_id))
    
    def delete_record(self, record_id: int) -> int:
        """删除记录"""
        sql = "DELETE FROM segment_records WHERE id = %s"
        return self._database.execute_non_query(sql, (record_id,))
    
    def count_by_user_id(self, user_id: int) -> int:
        """统计用户的记录数量"""
        sql = "SELECT COUNT(*) as count FROM segment_records WHERE user_id = %s"
        results = self._database.execute_query(sql, (user_id,))
        
        if results:
            return results[0]['count']
        return 0