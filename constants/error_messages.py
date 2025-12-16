# -*- coding: utf-8 -*-
"""
Error Messages
错误消息映射，根据错误码获取对应的错误消息
"""

from .error_codes import ErrorCode

# 错误码与错误消息的映射关系
ERROR_MESSAGES = {
    # 成功码 (2xxxx)
    ErrorCode.SUCCESS: "操作成功",
    ErrorCode.REGISTER_SUCCESS: "注册成功",
    ErrorCode.LOGIN_SUCCESS: "登录成功",
    ErrorCode.PASSWORD_CHANGE_SUCCESS: "密码修改成功",
    ErrorCode.VERIFICATION_SEND_SUCCESS: "验证码发送成功",
    ErrorCode.VERIFICATION_SUCCESS: "验证成功",
    ErrorCode.IMAGE_PROCESS_SUCCESS: "图片处理成功",
    ErrorCode.MODEL_LOAD_SUCCESS: "模型加载成功",
    ErrorCode.MODEL_RESET_SUCCESS: "模型重置成功",
    
    # 通用错误 (400xx)
    ErrorCode.SYSTEM_ERROR: "系统错误",
    ErrorCode.UNKNOWN: "未知错误",
    
    # 用户相关错误 (401xx)
    ErrorCode.USER_NOT_FOUND: "用户不存在",
    ErrorCode.USER_ALREADY_EXISTS: "用户已存在",
    ErrorCode.INVALID_CREDENTIALS: "用户名或密码错误",
    ErrorCode.PASSWORD_TOO_WEAK: "密码强度不够",
    
    # 验证码相关错误 (402xx)
    ErrorCode.EMPTY_EMAIL: "邮箱不能为空",
    ErrorCode.INVALID_EMAIL: "邮箱格式无效",
    ErrorCode.TOO_FREQUENT: "请求过于频繁，请稍后再试",
    ErrorCode.SEND_FAILED: "验证码发送失败",
    ErrorCode.VERIFICATION_FAILED: "验证码验证失败",
    
    # 图片相关错误 (403xx)
    ErrorCode.IMAGE_TOO_LARGE: "图片文件过大",
    ErrorCode.UNSUPPORTED_FORMAT: "不支持的图片格式",
    
    # 模型相关错误 (404xx)
    ErrorCode.MODEL_LOAD_FAILED: "模型加载失败",
    ErrorCode.INFERENCE_FAILED: "推理失败",
    ErrorCode.RESET_FAILED: "模型重置失败",
}

def get_error_message(error_code: ErrorCode) -> str:
    """
    根据错误码获取错误消息
    
    Args:
        error_code: 错误码
        
    Returns:
        str: 对应的错误消息
    """
    return ERROR_MESSAGES.get(error_code, "未知错误")

def get_error_message_by_code(code: int) -> str:
    """
    根据错误码数值获取错误消息
    
    Args:
        code: 错误码数值
        
    Returns:
        str: 对应的错误消息
    """
    try:
        error_code = ErrorCode(code)
        return get_error_message(error_code)
    except ValueError:
        return "未知错误"