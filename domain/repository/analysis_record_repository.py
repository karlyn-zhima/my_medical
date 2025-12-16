# -*- coding: utf-8 -*-
"""
Analysis Record Repository
分析记录仓储实现
"""

from typing import List, Optional
from datetime import datetime
import logging

from infrastructure.mysql.mysql import BaseRepository, MySQLDatabase
from domain.analysis_record import AnalysisRecord


class AnalysisRecordRepository(BaseRepository):
    """分析记录仓储"""
    
    def __init__(self, database: MySQLDatabase):
        super().__init__(database, 'analysis_records')
        self._logger = logging.getLogger(__name__)
    
    def create_table_if_not_exists(self):
        """创建表（如果不存在）"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS analysis_records (
            id INT AUTO_INCREMENT PRIMARY KEY,
            original_image_path VARCHAR(500) NOT NULL,
            original_image_hash VARCHAR(64) NOT NULL,
            similar_images TEXT,
            detected_layers TEXT,
            ai_analysis_result TEXT,
            status VARCHAR(20) DEFAULT 'pending',
            is_confirmed INT DEFAULT -1 COMMENT '-1: 未确认, 0: 医生拒绝, 1: 医生确认',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            user_id INT,
            INDEX idx_image_hash (original_image_hash),
            INDEX idx_status (status),
            INDEX idx_user_id (user_id),
            INDEX idx_created_at (created_at),
            INDEX idx_is_confirmed (is_confirmed)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        try:
            self._database.execute_non_query(create_table_sql)
            self._logger.info("分析记录表创建成功或已存在")
        except Exception as e:
            self._logger.error(f"创建分析记录表失败: {e}")
            raise
    
    def save_record(self, record: AnalysisRecord) -> int:
        """保存分析记录"""
        if record.id is None:
            # 插入新记录 - 使用同一连接确保LAST_INSERT_ID正确
            insert_sql = """
            INSERT INTO analysis_records (
                original_image_path, original_image_hash, similar_images,
                detected_layers, ai_analysis_result, status, is_confirmed, user_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            record_dict = record.to_dict()
            params = (
                record_dict['original_image_path'],
                record_dict['original_image_hash'],
                record_dict['similar_images'],
                record_dict['detected_layers'],
                record_dict['ai_analysis_result'],
                record_dict['status'],
                record_dict['is_confirmed'],
                record_dict['user_id']
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
    
    def update_record(self, record: AnalysisRecord) -> int:
        """更新分析记录"""
        update_sql = """
        UPDATE analysis_records SET
            original_image_path = %s,
            original_image_hash = %s,
            similar_images = %s,
            detected_layers = %s,
            ai_analysis_result = %s,
            status = %s,
            is_confirmed = %s,
            updated_at = %s,
            user_id = %s
        WHERE id = %s
        """
        
        record_dict = record.to_dict()
        params = (
            record_dict['original_image_path'],
            record_dict['original_image_hash'],
            record_dict['similar_images'],
            record_dict['detected_layers'],
            record_dict['ai_analysis_result'],
            record_dict['status'],
            record_dict['is_confirmed'],
            datetime.now(),
            record_dict['user_id'],
            record.id
        )
        
        return self._database.execute_non_query(update_sql, params)
    
    def find_by_id(self, record_id: int) -> Optional[AnalysisRecord]:
        """根据ID查找分析记录"""
        sql = "SELECT * FROM analysis_records WHERE id = %s"
        result = self._database.execute_query(sql, (record_id,))
        if result:
            return AnalysisRecord.from_dict(result[0])
        return None
    
    def find_by_image_hash(self, image_hash: str) -> Optional[AnalysisRecord]:
        """根据图像哈希查找分析记录"""
        sql = "SELECT * FROM analysis_records WHERE original_image_hash = %s ORDER BY created_at DESC LIMIT 1"
        result = self._database.execute_query(sql, (image_hash,))
        if result:
            return AnalysisRecord.from_dict(result[0])
        return None
    
    def find_by_user_id(self, user_id: int, limit: int = 50, offset: int = 0) -> List[AnalysisRecord]:
        """根据用户ID查找分析记录"""
        sql = """
        SELECT * FROM analysis_records 
        WHERE user_id = %s 
        ORDER BY created_at DESC 
        LIMIT %s OFFSET %s
        """
        results = self._database.execute_query(sql, (user_id, limit, offset))
        return [AnalysisRecord.from_dict(row) for row in results]
    
    def find_by_status(self, status: str, limit: int = 50) -> List[AnalysisRecord]:
        """根据状态查找分析记录"""
        sql = """
        SELECT * FROM analysis_records 
        WHERE status = %s 
        ORDER BY created_at DESC 
        LIMIT %s
        """
        results = self._database.execute_query(sql, (status, limit))
        return [AnalysisRecord.from_dict(row) for row in results]
    
    def update_status(self, record_id: int, status: str) -> int:
        """更新分析记录状态"""
        sql = "UPDATE analysis_records SET status = %s, updated_at = %s WHERE id = %s"
        return self._database.execute_non_query(sql, (status, datetime.now(), record_id))
    
    def update_confirmation_status(self, record_id: int, is_confirmed: int) -> int:
        """更新确认状态"""
        sql = "UPDATE analysis_records SET is_confirmed = %s, updated_at = %s WHERE id = %s"
        return self._database.execute_non_query(sql, (is_confirmed, datetime.now(), record_id))
    
    def delete_record(self, record_id: int) -> int:
        """删除分析记录"""
        sql = "DELETE FROM analysis_records WHERE id = %s"
        return self._database.execute_non_query(sql, (record_id,))
    
    def count_by_user_id(self, user_id: int) -> int:
        """统计用户的分析记录数量"""
        sql = "SELECT COUNT(*) as count FROM analysis_records WHERE user_id = %s"
        result = self._database.execute_query(sql, (user_id,))
        if result:
            return result[0]['count']
        return 0
    
    def find_unconfirmed_records(self, limit: int = 50) -> List[AnalysisRecord]:
        """查找未确认的分析记录"""
        sql = """
        SELECT * FROM analysis_records 
        WHERE is_confirmed = -1 AND status = 'completed'
        ORDER BY created_at ASC 
        LIMIT %s
        """
        results = self._database.execute_query(sql, (limit,))
        return [AnalysisRecord.from_dict(row) for row in results]

    def find_unconfirmed_by_user(self, user_id: int, limit: int = 50, offset: int = 0) -> List[AnalysisRecord]:
        """根据用户ID查找未确认的分析记录"""
        sql = """
        SELECT * FROM analysis_records 
        WHERE user_id = %s AND is_confirmed = -1 AND status = 'completed'
        ORDER BY created_at DESC 
        LIMIT %s OFFSET %s
        """
        results = self._database.execute_query(sql, (user_id, limit, offset))
        return [AnalysisRecord.from_dict(row) for row in results]