"""
认证装饰器
提供token验证装饰器，用于保护需要认证的API接口
"""

from functools import wraps
from flask import request, jsonify, g
from typing import Callable, Any
from .jwt_token import verify_token


def token_required(f: Callable) -> Callable:
    """
    Token验证装饰器
    用于保护需要认证的API接口
    
    使用方法:
    @token_required
    def protected_api():
        # 可以通过 g.current_user_id 获取当前用户ID
        # 可以通过 g.token_payload 获取完整的token payload
        pass
    """
    @wraps(f)
    def decorated(*args, **kwargs) -> Any:
        token = None
        
        # 从请求头获取token
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                # 支持 "Bearer <token>" 格式
                if auth_header.startswith('Bearer '):
                    token = auth_header.split(' ')[1]
                else:
                    token = auth_header
            except IndexError:
                return jsonify({
                    'success': False,
                    'message': 'Token格式错误',
                    'code': 401
                }), 401
        
        # 从请求参数获取token（备选方案）
        if not token:
            token = request.args.get('token')
        
        # 从请求体获取token（备选方案）
        if not token and request.is_json:
            token = request.json.get('token')
        
        if not token:
            return jsonify({
                'success': False,
                'message': '缺少认证token',
                'code': 401
            }), 401
        
        # 验证token
        payload = verify_token(token)
        if not payload:
            return jsonify({
                'success': False,
                'message': 'Token无效或已过期',
                'code': 401
            }), 401
        
        # 将用户信息存储到g对象中，供后续使用
        g.current_user_id = payload.get('user_id')
        g.token_payload = payload
        g.current_token = token
        
        return f(*args, **kwargs)
    
    return decorated


def optional_token(f: Callable) -> Callable:
    """
    可选Token验证装饰器
    如果提供了token则验证，如果没有提供则继续执行
    
    使用方法:
    @optional_token
    def api_with_optional_auth():
        if hasattr(g, 'current_user_id'):
            # 用户已认证
            pass
        else:
            # 用户未认证
            pass
    """
    @wraps(f)
    def decorated(*args, **kwargs) -> Any:
        token = None
        
        # 从请求头获取token
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                if auth_header.startswith('Bearer '):
                    token = auth_header.split(' ')[1]
                else:
                    token = auth_header
            except IndexError:
                pass
        
        # 从请求参数获取token
        if not token:
            token = request.args.get('token')
        
        # 从请求体获取token
        if not token and request.is_json:
            token = request.json.get('token')
        
        # 如果有token，尝试验证
        if token:
            payload = verify_token(token)
            if payload:
                g.current_user_id = payload.get('user_id')
                g.token_payload = payload
                g.current_token = token
        
        return f(*args, **kwargs)
    
    return decorated


def admin_required(f: Callable) -> Callable:
    """
    管理员权限验证装饰器
    需要先使用 @token_required 装饰器
    
    使用方法:
    @token_required
    @admin_required
    def admin_only_api():
        pass
    """
    @wraps(f)
    def decorated(*args, **kwargs) -> Any:
        # 检查是否已经通过token验证
        if not hasattr(g, 'token_payload'):
            return jsonify({
                'success': False,
                'message': '需要先进行身份认证',
                'code': 401
            }), 401
        
        # 检查是否为管理员（这里可以根据实际业务逻辑调整）
        payload = g.token_payload
        is_admin = payload.get('is_admin', False)
        
        if not is_admin:
            return jsonify({
                'success': False,
                'message': '需要管理员权限',
                'code': 403
            }), 403
        
        return f(*args, **kwargs)
    
    return decorated