# -*- coding: utf-8 -*-
"""
短信验证码服务
SMS Verification Service
"""

import json
import os
import random
import string
import time
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import traceback
from .sms_template import SMSTemplate

from infrastructure.redis import getRedisRepository


class SMSVerificationService:
    """短信验证码服务类"""
    
    def __init__(self):
        self.redis_repo = getRedisRepository()
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', '..', 'config', 'config.json')
        config_path = os.path.normpath(config_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _generate_verification_code(self) -> str:
        """生成验证码"""
        code_length = self.config.get('verification', {}).get('code_length', 6)
        return ''.join(random.choices(string.digits, k=code_length))
    
    def _get_redis_key(self, phone: str, key_type: str) -> str:
        """获取Redis键名"""
        return f"sms_verification:{key_type}:{phone}"
    
    def _validate_phone_number(self, phone: str) -> bool:
        """验证手机号格式"""
        import re
        # 简单的中国手机号验证
        pattern = r'^1[3-9]\d{9}$'
        return bool(re.match(pattern, phone))
    
    def _send_sms(self, phone: str, message: str) -> bool:
        """发送短信"""
        try:
            # 这里需要集成具体的短信服务商API
            # 以下是示例代码，需要根据实际使用的短信服务商进行修改
            
            sms_config = self.config.get('sms', {})
            
            # 示例：使用阿里云短信服务
            if sms_config.get('provider') == 'aliyun':
                return self._send_aliyun_sms(phone, message, sms_config)
            # 示例：使用腾讯云短信服务
            elif sms_config.get('provider') == 'tencent':
                return self._send_tencent_sms(phone, message, sms_config)
            # 开发环境模拟发送
            elif sms_config.get('provider') == 'mock':
                print(f"[模拟短信] 发送到 {phone}: {message}")
                return True
            else:
                print(f"[短信服务] 未配置短信服务商，消息: {message}")
                return False
                
        except Exception as e:
            print(f"发送短信失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def _send_aliyun_sms(self, phone: str, message: str, config: Dict[str, Any]) -> bool:
        """发送阿里云短信（示例实现）"""
        try:
            # 这里需要安装并使用阿里云SDK
            # pip install alibabacloud_dysmsapi20170525
            
            # 示例代码（需要根据实际情况调整）
            """
            from alibabacloud_dysmsapi20170525.client import Client as Dysmsapi20170525Client
            from alibabacloud_tea_openapi import models as open_api_models
            from alibabacloud_dysmsapi20170525 import models as dysmsapi_20170525_models
            
            config_ali = open_api_models.Config(
                access_key_id=config.get('access_key_id'),
                access_key_secret=config.get('access_key_secret'),
            )
            config_ali.endpoint = 'dysmsapi.aliyuncs.com'
            
            client = Dysmsapi20170525Client(config_ali)
            send_sms_request = dysmsapi_20170525_models.SendSmsRequest(
                phone_numbers=phone,
                sign_name=config.get('sign_name'),
                template_code=config.get('template_code'),
                template_param=json.dumps({'code': message})
            )
            
            response = client.send_sms(send_sms_request)
            return response.body.code == 'OK'
            """
            
            # 临时模拟实现
            print(f"[阿里云短信模拟] 发送到 {phone}: {message}")
            return True
            
        except Exception as e:
            print(f"阿里云短信发送失败: {str(e)}")
            return False
    
    def _send_tencent_sms(self, phone: str, message: str, config: Dict[str, Any]) -> bool:
        """发送腾讯云短信（示例实现）"""
        try:
            # 这里需要安装并使用腾讯云SDK
            # pip install tencentcloud-sdk-python
            
            # 临时模拟实现
            print(f"[腾讯云短信模拟] 发送到 {phone}: {message}")
            return True
            
        except Exception as e:
            print(f"腾讯云短信发送失败: {str(e)}")
            return False
    
    def send_verification_code(self, phone: str) -> Dict[str, Any]:
        """发送验证码短信"""
        try:
            # 验证手机号格式
            if not self._validate_phone_number(phone):
                return {
                    'success': False,
                    'message': '手机号格式不正确',
                    'error_code': 'INVALID_PHONE_FORMAT'
                }
            
            # 检查发送频率限制
            rate_limit_key = self._get_redis_key(phone, 'rate_limit')
            if self.redis_repo.exists(rate_limit_key):
                remaining_time = self.redis_repo.ttl(rate_limit_key)
                return {
                    'success': False,
                    'message': f'发送过于频繁，请{remaining_time}秒后再试',
                    'error_code': 'RATE_LIMIT_EXCEEDED',
                    'remaining_time': remaining_time
                }
            
            # 生成验证码
            verification_code = self._generate_verification_code()
            
            # 获取配置
            expire_minutes = self.config.get('verification', {}).get('expire_minutes', 5)
            rate_limit_seconds = self.config.get('verification', {}).get('rate_limit_seconds', 60)
            
            # 生成短信内容
            message = SMSTemplate.verification_code_template(verification_code, expire_minutes)
            
            # 发送短信
            if self._send_sms(phone, message):
                # 存储验证码到Redis
                code_key = self._get_redis_key(phone, 'code')
                self.redis_repo.set(code_key, verification_code, ex=expire_minutes * 60)
                
                # 设置发送频率限制
                self.redis_repo.set(rate_limit_key, '1', ex=rate_limit_seconds)
                
                # 记录发送时间
                sent_time_key = self._get_redis_key(phone, 'sent_time')
                current_time = str(int(time.time()))
                self.redis_repo.set(sent_time_key, current_time, ex=expire_minutes * 60)
                
                return {
                    'success': True,
                    'message': '验证码发送成功',
                    'expire_minutes': expire_minutes
                }
            else:
                return {
                    'success': False,
                    'message': '短信发送失败，请稍后重试',
                    'error_code': 'SMS_SEND_FAILED'
                }
                
        except Exception as e:
            print(f"发送验证码异常: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'message': '系统异常，请稍后重试',
                'error_code': 'SYSTEM_ERROR'
            }
    
    def verify_code(self, phone: str, code: str) -> Dict[str, Any]:
        """验证短信验证码"""
        try:
            # 验证手机号格式
            if not self._validate_phone_number(phone):
                return {
                    'success': False,
                    'message': '手机号格式不正确',
                    'error_code': 'INVALID_PHONE_FORMAT'
                }
            
            # 获取存储的验证码
            code_key = self._get_redis_key(phone, 'code')
            stored_code = self.redis_repo.get(code_key)
            
            if not stored_code:
                return {
                    'success': False,
                    'message': '验证码已过期或不存在',
                    'error_code': 'CODE_EXPIRED'
                }
            
            # 验证码比对
            stored_code_str = str(stored_code) if isinstance(stored_code, int) else stored_code
            if stored_code_str != code:
                return {
                    'success': False,
                    'message': '验证码错误',
                    'error_code': 'CODE_INCORRECT'
                }
            
            # 验证成功，删除验证码
            self.redis_repo.delete(code_key)
            
            # 验证成功，标记手机号为已验证
            verified_key = self._get_redis_key(phone, 'verified')
            self.redis_repo.set(verified_key, "1", ex=30 * 24 * 60 * 60)  # 30天有效期
            
            return {
                'success': True,
                'message': '验证成功'
            }
            
        except Exception as e:
            print(f"验证码验证异常: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'message': '系统异常，请稍后重试',
                'error_code': 'SYSTEM_ERROR'
            }
    
    def get_verification_status(self, phone: str) -> Dict[str, Any]:
        """获取验证状态"""
        try:
            # 验证手机号格式
            if not self._validate_phone_number(phone):
                return {
                    'success': False,
                    'message': '手机号格式不正确',
                    'error_code': 'INVALID_PHONE_FORMAT'
                }
            
            # 检查是否已验证
            verified_key = self._get_redis_key(phone, 'verified')
            is_verified = self.redis_repo.exists(verified_key)
            
            # 检查验证码是否存在
            code_key = self._get_redis_key(phone, 'code')
            code_exists = self.redis_repo.exists(code_key)
            code_ttl = self.redis_repo.ttl(code_key) if code_exists else 0
            
            # 检查发送频率限制
            rate_limit_key = self._get_redis_key(phone, 'rate_limit')
            rate_limited = self.redis_repo.exists(rate_limit_key)
            rate_limit_ttl = self.redis_repo.ttl(rate_limit_key) if rate_limited else 0
            
            return {
                'success': True,
                'data': {
                    'phone': phone,
                    'is_verified': is_verified,
                    'code_exists': code_exists,
                    'code_remaining_time': max(0, code_ttl),
                    'rate_limited': rate_limited,
                    'rate_limit_remaining_time': max(0, rate_limit_ttl)
                }
            }
            
        except Exception as e:
            print(f"获取验证状态异常: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'message': '系统异常，请稍后重试',
                'error_code': 'SYSTEM_ERROR'
            }
    
    def is_phone_verified(self, phone: str) -> bool:
        """检查手机号是否已验证"""
        try:
            if not self._validate_phone_number(phone):
                return False
                
            verified_key = self._get_redis_key(phone, 'verified')
            return self.redis_repo.exists(verified_key)
        except:
            return False


# 全局实例
_sms_verification_service = None

def get_sms_verification_service() -> SMSVerificationService:
    """获取短信验证服务实例（单例模式）"""
    global _sms_verification_service
    if _sms_verification_service is None:
        _sms_verification_service = SMSVerificationService()
    return _sms_verification_service