# -*- coding: utf-8 -*-
"""
邮箱基础设施模块
Email Infrastructure Module
"""

from .email_verification import EmailVerificationService, get_email_verification_service

__all__ = [
    'EmailVerificationService',
    'get_email_verification_service'
]