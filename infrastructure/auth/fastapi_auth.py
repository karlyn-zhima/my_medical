"""
FastAPI认证依赖
提供token验证依赖，用于保护需要认证的API接口
"""

from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
from .jwt_token import verify_token


# HTTP Bearer token scheme - 设置auto_error=False以返回401而不是403
security = HTTPBearer(auto_error=False)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    获取当前用户信息（必需token）
    用于需要认证的API接口
    
    Args:
        credentials: HTTP Bearer认证凭据
        
    Returns:
        用户信息字典
        
    Raises:
        HTTPException: token无效或过期时抛出401错误
    """
    # 检查是否提供了credentials
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail={
                'success': False,
                'message': '需要提供认证token',
                'code': 401
            }
        )
    
    token = credentials.credentials
    
    # 验证token
    payload = verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=401,
            detail={
                'success': False,
                'message': 'Token无效或已过期',
                'code': 401
            }
        )
    
    return payload


async def get_current_user_optional(request: Request) -> Optional[Dict[str, Any]]:
    """
    获取当前用户信息（可选token）
    如果提供了token则验证，如果没有提供则返回None
    
    Args:
        request: FastAPI请求对象
        
    Returns:
        用户信息字典或None
    """
    token = None
    
    # 从Authorization header获取token
    authorization = request.headers.get('Authorization')
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ')[1]
    
    # 从查询参数获取token（备选方案）
    if not token:
        token = request.query_params.get('token')
    
    # 如果有token，尝试验证
    if token:
        payload = verify_token(token)
        return payload
    
    return None


async def get_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    获取管理员用户信息
    需要先通过token验证，然后检查管理员权限
    
    Args:
        current_user: 当前用户信息
        
    Returns:
        管理员用户信息字典
        
    Raises:
        HTTPException: 非管理员用户时抛出403错误
    """
    # 检查是否为管理员（这里可以根据实际业务逻辑调整）
    is_admin = current_user.get('is_admin', False)
    
    if not is_admin:
        raise HTTPException(
            status_code=403,
            detail={
                'success': False,
                'message': '需要管理员权限',
                'code': 403
            }
        )
    
    return current_user


def get_user_id_from_payload(payload: Dict[str, Any]) -> str:
    """
    从payload中获取用户ID
    
    Args:
        payload: token解析后的payload
        
    Returns:
        用户ID字符串
    """
    return str(payload.get('user_id', ''))


def get_username_from_payload(payload: Dict[str, Any]) -> str:
    """
    从payload中获取用户名
    
    Args:
        payload: token解析后的payload
        
    Returns:
        用户名字符串
    """
    return payload.get('username', '')