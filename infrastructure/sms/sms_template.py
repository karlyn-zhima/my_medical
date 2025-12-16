# -*- coding: utf-8 -*-
"""
短信模板管理模块
用于管理各种短信模板的内容
"""

class SMSTemplate:
    """短信模板类，用于生成各种类型的短信内容"""
    
    @staticmethod
    def verification_code_template(verification_code: str, expire_minutes: int) -> str:
        """
        生成验证码短信模板
        
        Args:
            verification_code (str): 验证码
            expire_minutes (int): 过期时间（分钟）
            
        Returns:
            str: 短信内容
        """
        return f"【您的应用】您的验证码是：{verification_code}，有效期{expire_minutes}分钟，请及时使用。如非本人操作，请忽略此短信。"
    
    @staticmethod
    def welcome_template(username: str) -> str:
        """
        生成欢迎短信模板
        
        Args:
            username (str): 用户名
            
        Returns:
            str: 短信内容
        """
        return f"【您的应用】亲爱的{username}，欢迎您注册我们的服务！您的账户已经成功创建。"
    
    @staticmethod
    def password_reset_template(verification_code: str, expire_minutes: int) -> str:
        """
        生成密码重置短信模板
        
        Args:
            verification_code (str): 验证码
            expire_minutes (int): 过期时间（分钟）
            
        Returns:
            str: 短信内容
        """
        return f"【您的应用】您正在重置密码，验证码：{verification_code}，有效期{expire_minutes}分钟。如非本人操作，请忽略此短信。"
    
    @staticmethod
    def login_notification_template(location: str = "未知位置") -> str:
        """
        生成登录通知短信模板
        
        Args:
            location (str): 登录位置
            
        Returns:
            str: 短信内容
        """
        return f"【您的应用】您的账户在{location}登录，如非本人操作，请及时修改密码。"