# -*- coding: utf-8 -*-
"""
短信验证模块
SMS Verification Module
"""

from .sms_verification import SMSVerificationService, get_sms_verification_service
from .sms_template import SMSTemplate

__all__ = [
    'SMSVerificationService',
    'get_sms_verification_service', 
    'SMSTemplate'
]