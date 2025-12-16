# -*- coding: utf-8 -*-
"""
验证码检查服务
Verification Check Service
"""

from typing import Dict, Any
from infrastructure.email import get_email_verification_service


class VerificationCheckService:
    """验证码检查服务类"""
    
    def __init__(self):
        self.email_service = get_email_verification_service()
    
    def send_email_verification_code(self, email: str) -> Dict[str, Any]:
        """
        发送邮箱验证码
        
        Args:
            email: 邮箱地址
            
        Returns:
            Dict: 包含发送结果的字典
        """
        if not email:
            return {
                'success': False,
                'message': '邮箱地址不能为空',
                'code': 'EMPTY_EMAIL'
            }
        
        return self.email_service.send_verification_code(email)
    
    def verify_email_code(self, email: str, code: str) -> Dict[str, Any]:
        """
        验证邮箱验证码
        
        Args:
            email: 邮箱地址
            code: 验证码
            
        Returns:
            Dict: 包含验证结果的字典
        """
        if not email:
            return {
                'success': False,
                'message': '邮箱地址不能为空',
                'code': 'EMPTY_EMAIL'
            }
        
        if not code:
            return {
                'success': False,
                'message': '验证码不能为空',
                'code': 'EMPTY_CODE'
            }
        
        return self.email_service.verify_code(email, code)
    
    def get_verification_status(self, email: str) -> Dict[str, Any]:
        """
        获取验证码状态
        
        Args:
            email: 邮箱地址
            
        Returns:
            Dict: 验证码状态信息
        """
        if not email:
            return {
                'success': False,
                'message': '邮箱地址不能为空',
                'code': 'EMPTY_EMAIL'
            }
        
        return self.email_service.get_verification_status(email)


# 全局服务实例
_verification_check_service = None

def get_verification_check_service() -> VerificationCheckService:
    """获取验证码检查服务实例（单例模式）"""
    global _verification_check_service
    if _verification_check_service is None:
        _verification_check_service = VerificationCheckService()
    return _verification_check_service


# 便捷接口函数
def send_verification_code(email: str) -> Dict[str, Any]:
    """
    发送验证码的便捷接口
    
    Args:
        email: 邮箱地址
        
    Returns:
        Dict: 发送结果
    """
    service = get_verification_check_service()
    return service.send_email_verification_code(email)


def check_verification_code(email: str, code: str) -> Dict[str, Any]:
    """
    验证验证码的便捷接口
    
    Args:
        email: 邮箱地址
        code: 验证码
        
    Returns:
        Dict: 验证结果
    """
    service = get_verification_check_service()
    return service.verify_email_code(email, code)


def get_code_status(email: str) -> Dict[str, Any]:
    """
    获取验证码状态的便捷接口
    
    Args:
        email: 邮箱地址
        
    Returns:
        Dict: 状态信息
    """
    service = get_verification_check_service()
    return service.get_verification_status(email)