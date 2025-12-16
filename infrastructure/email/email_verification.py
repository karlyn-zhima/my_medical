# -*- coding: utf-8 -*-
"""
邮箱验证码服务
Email Verification Service
"""

import json
import os
import random
import smtplib
import string
import time
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any
import traceback
from .email_template import EmailTemplate

from infrastructure.redis import getRedisRepository


class EmailVerificationService:
    """邮箱验证码服务类"""
    
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
        code_length = self.config['verification']['code_length']
        return ''.join(random.choices(string.digits, k=code_length))
    
    def _get_redis_key(self, email: str, key_type: str) -> str:
        """获取Redis键名"""
        return f"verification:{key_type}:{email}"
    
    def _send_email(self, to_email: str, verification_code: str) -> bool:
        """发送验证码邮件"""
        try:
            email_config = self.config['email']
            print(f"正在连接SMTP服务器: {email_config['smtp_server']}:{email_config['smtp_port']}")
            print(f"发送邮箱: {email_config['sender_email']}")
            print(f"接收邮箱: {to_email}")
            
            # 创建邮件内容
            msg = MIMEMultipart()
            msg['From'] = email_config['sender_email']
            msg['To'] = to_email
            msg['Subject'] = "邮箱验证码"
            
            # 邮件正文
            # 使用邮件模板生成HTML内容
            body = EmailTemplate.verification_code_template(
                verification_code=verification_code,
                expire_minutes=self.config['verification']['expire_minutes']
            )
            
            msg.attach(MIMEText(body, 'html', 'utf-8'))
            
            # 连接SMTP服务器并发送邮件
            print("正在连接SMTP服务器...")
            
            if email_config.get('use_ssl', False):
                # 使用SSL连接
                server = smtplib.SMTP_SSL(email_config['smtp_server'], email_config['smtp_port'])
            else:
                # 使用普通连接
                server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
                
            server.set_debuglevel(1)  # 启用调试模式
            
            if email_config.get('use_tls', False) and not email_config.get('use_ssl', False):
                print("正在启动TLS...")
                server.starttls()
            
            print("正在登录...")
            server.login(email_config['sender_email'], email_config['sender_password'])
            
            print("正在发送邮件...")
            server.send_message(msg)
            
            print("正在关闭连接...")
            server.quit()
            
            print("邮件发送成功！")
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            print(f"SMTP认证失败: {e}")
            print("请检查邮箱地址和授权码是否正确")
            return False
        except smtplib.SMTPConnectError as e:
            print(f"SMTP连接失败: {e}")
            print("请检查SMTP服务器地址和端口是否正确")
            return False
        except smtplib.SMTPException as e:
            print(f"SMTP错误: {e}")
            return False
        except Exception as e:
            print(f"发送邮件失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def send_verification_code(self, email: str) -> Dict[str, Any]:
        """
        发送验证码
        
        Args:
            email: 邮箱地址
            
        Returns:
            Dict: 包含状态和消息的字典
        """
        try:
            # 检查邮箱格式
            if '@' not in email or '.' not in email.split('@')[1]:
                return {
                    'success': False,
                    'message': '邮箱格式不正确',
                    'code': 'INVALID_EMAIL'
                }
            
            # 检查发送间隔
            last_send_key = self._get_redis_key(email, 'last_send')
            last_send_time = self.redis_repo.get(last_send_key)
            
            if last_send_time:
                last_send_timestamp = float(last_send_time)
                current_time = time.time()
                interval = self.config['verification']['resend_interval_seconds']
                
                if current_time - last_send_timestamp < interval:
                    remaining = int(interval - (current_time - last_send_timestamp))
                    return {
                        'success': False,
                        'message': f'请等待 {remaining} 秒后再次发送',
                        'code': 'TOO_FREQUENT'
                    }
            
            # 生成验证码
            verification_code = self._generate_verification_code()
            
            # 发送邮件
            if not self._send_email(email, verification_code):
                return {
                    'success': False,
                    'message': '邮件发送失败，请检查邮箱地址或稍后重试',
                    'code': 'SEND_FAILED'
                }
            
            # 存储验证码到Redis
            code_key = self._get_redis_key(email, 'code')
            attempts_key = self._get_redis_key(email, 'attempts')
            
            expire_seconds = self.config['verification']['expire_minutes'] * 60
            
            # 存储验证码和过期时间
            self.redis_repo.set(code_key, verification_code, ex=expire_seconds)
            # 重置尝试次数
            self.redis_repo.set(attempts_key, 0, ex=expire_seconds)
            # 记录发送时间
            self.redis_repo.set(
                last_send_key, 
                str(time.time()),
                ex=self.config['verification']['resend_interval_seconds']
            )
            
            return {
                'success': True,
                'message': '验证码已发送，请查收邮件',
                'code': 'SUCCESS',
                'expire_minutes': self.config['verification']['expire_minutes']
            }
            
        except Exception as e:
            print(f"发送验证码异常: {e}")
            return {
                'success': False,
                'message': '系统异常，请稍后重试',
                'code': 'SYSTEM_ERROR'
            }
    
    def verify_code(self, email: str, code: str) -> Dict[str, Any]:
        """
        验证验证码
        
        Args:
            email: 邮箱地址
            code: 验证码
            
        Returns:
            Dict: 包含验证结果的字典
        """
        try:
            code_key = self._get_redis_key(email, 'code')
            attempts_key = self._get_redis_key(email, 'attempts')
            
            # 获取存储的验证码
            stored_code = self.redis_repo.get(code_key)
            
            if not stored_code:
                return {
                    'success': False,
                    'message': '验证码已过期或不存在，请重新获取',
                    'code': 'CODE_EXPIRED'
                }
            
            # 检查尝试次数
            attempts = self.redis_repo.get(attempts_key)
            current_attempts = int(attempts) if attempts else 0
            max_attempts = self.config['verification']['max_attempts']
            
            if current_attempts >= max_attempts:
                # 删除验证码
                self.redis_repo.delete(code_key)
                self.redis_repo.delete(attempts_key)
                return {
                    'success': False,
                    'message': f'验证失败次数过多，验证码已失效，请重新获取',
                    'code': 'TOO_MANY_ATTEMPTS'
                }
            
            # 验证码比较
            if str(stored_code) == code:
                # 验证成功，删除相关键
                self.redis_repo.delete(code_key)
                self.redis_repo.delete(attempts_key)
                
                return {
                    'success': True,
                    'message': '验证码验证成功',
                    'code': 'SUCCESS'
                }
            else:
                # 验证失败，增加尝试次数
                new_attempts = current_attempts + 1
                remaining_attempts = max_attempts - new_attempts
                
                # 更新尝试次数，保持与验证码相同的过期时间
                ttl = self.redis_repo.ttl(code_key)
                if ttl > 0:
                    self.redis_repo.set(attempts_key, new_attempts, ex=ttl)
                
                if remaining_attempts > 0:
                    return {
                        'success': False,
                        'message': f'验证码错误，还可尝试 {remaining_attempts} 次',
                        'code': 'CODE_INCORRECT',
                        'remaining_attempts': remaining_attempts
                    }
                else:
                    # 达到最大尝试次数，删除验证码
                    self.redis_repo.delete(code_key)
                    self.redis_repo.delete(attempts_key)
                    return {
                        'success': False,
                        'message': '验证失败次数过多，验证码已失效，请重新获取',
                        'code': 'TOO_MANY_ATTEMPTS'
                    }
                    
        except Exception as e:
            print(f"验证验证码异常: {e}")
            return {
                'success': False,
                'message': '系统异常，请稍后重试',
                'code': 'SYSTEM_ERROR'
            }
    
    def get_verification_status(self, email: str) -> Dict[str, Any]:
        """
        获取验证码状态
        
        Args:
            email: 邮箱地址
            
        Returns:
            Dict: 验证码状态信息
        """
        try:
            code_key = self._get_redis_key(email, 'code')
            attempts_key = self._get_redis_key(email, 'attempts')
            last_send_key = self._get_redis_key(email, 'last_send')
            
            # 检查验证码是否存在
            code_exists = self.redis_repo.exists(code_key)
            
            if not code_exists:
                return {
                    'success': True,
                    'has_code': False,
                    'message': '当前无有效验证码'
                }
            
            # 获取剩余时间
            ttl = self.redis_repo.ttl(code_key)
            attempts = self.redis_repo.get(attempts_key)
            current_attempts = int(attempts) if attempts else 0
            max_attempts = self.config['verification']['max_attempts']
            
            # 检查重发间隔
            can_resend = True
            resend_remaining = 0
            
            last_send_time = self.redis_repo.get(last_send_key)
            if last_send_time:
                last_send_timestamp = float(last_send_time)
                current_time = time.time()
                interval = self.config['verification']['resend_interval_seconds']
                
                if current_time - last_send_timestamp < interval:
                    can_resend = False
                    resend_remaining = int(interval - (current_time - last_send_timestamp))
            
            return {
                'success': True,
                'has_code': True,
                'remaining_seconds': ttl,
                'remaining_minutes': round(ttl / 60, 1),
                'current_attempts': current_attempts,
                'max_attempts': max_attempts,
                'remaining_attempts': max_attempts - current_attempts,
                'can_resend': can_resend,
                'resend_remaining_seconds': resend_remaining
            }
            
        except Exception as e:
            print(f"获取验证码状态异常: {e}")
            return {
                'success': False,
                'has_code': False,
                'message': '获取状态失败'
            }


# 全局服务实例
_email_verification_service = None

def get_email_verification_service() -> EmailVerificationService:
    """获取邮箱验证码服务实例（单例模式）"""
    global _email_verification_service
    if _email_verification_service is None:
        _email_verification_service = EmailVerificationService()
    return _email_verification_service