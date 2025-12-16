# -*- coding: utf-8 -*-
"""
User Service Layer - Application Service Implementation
用户应用服务层 - 实现用户相关的业务逻辑
"""

import sys
import os
from typing import Optional, Dict, Any, List
from decimal import Decimal
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain.user import (
    User, Email, Username, Password, TokenCount, PhoneNumber,
    UserDomainService, UserRepository, IUserRepository
)
from infrastructure.mysql import getMedicalDBWithTableName
from infrastructure.auth.jwt_token import generate_token
from constants.error_codes import ErrorCode
from constants.error_messages import get_error_message


# ================================
# 自定义异常类
# ================================

class UserServiceException(Exception):
    """用户服务异常基类"""
    pass


class UserAlreadyExistsException(UserServiceException):
    """用户已存在异常"""
    pass


class UserNotFoundException(UserServiceException):
    """用户未找到异常"""
    pass


class InvalidCredentialsException(UserServiceException):
    """无效凭据异常"""
    pass


class ValidationException(UserServiceException):
    """验证异常"""
    pass


# ================================
# 用户服务类
# ================================

class UserService:
    """用户应用服务"""
    
    def __init__(self, user_repository: Optional[IUserRepository] = None):
        """初始化用户服务"""
        if user_repository is None:
            # 使用默认的数据库仓储
            base_repository = getMedicalDBWithTableName("user")
            self._user_repository = UserRepository(base_repository)
        else:
            self._user_repository = user_repository
    
    # ================================
    # 用户注册相关方法
    # ================================
    
    def register_user(
        self,
        email: str,
        username: str,
        password: str,
        phone_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        用户注册
        
        Args:
            email: 邮箱地址
            username: 用户名
            password: 密码
            phone_number: 手机号（可选）
            
        Returns:
            Dict[str, Any]: 注册结果，包含用户信息
            
        Raises:
            UserAlreadyExistsException: 用户已存在
            ValidationException: 验证失败
        """
        try:
            # 1. 验证输入参数
            self._validate_registration_input(email, username, password, phone_number)
            
            # 2. 检查用户名是否已存在
            if not UserDomainService.is_username_available(username, self._user_repository):
                raise UserAlreadyExistsException(f"用户名 '{username}' 已存在")
            
            # 3. 检查邮箱是否已存在
            if not UserDomainService.is_email_available(email, self._user_repository):
                raise UserAlreadyExistsException(f"邮箱 '{email}' 已被注册")
            
            # 4. 创建新用户
            new_user = UserDomainService.create_new_user(
                email=email,
                username=username,
                plain_password=password,
                phone_number=phone_number,
                is_verified=True  # 注册时已验证邮箱验证码，设置为已验证
            )
            
            # 5. 保存用户到数据库
            saved_user = self._user_repository.save(new_user)
            
            # 6. 返回注册结果
            return {
                "success": True,
                "message": get_error_message(ErrorCode.REGISTER_SUCCESS),
                "code": ErrorCode.REGISTER_SUCCESS,
                "user": self._user_to_dict(saved_user, include_sensitive=False)
            }
            
        except (UserAlreadyExistsException, ValidationException):
            raise
        except Exception as e:
            raise UserServiceException(f"用户注册失败: {str(e)}")
    
    def _validate_registration_input(
        self,
        email: str,
        username: str,
        password: str,
        phone_number: Optional[int]
    ) -> None:
        """验证注册输入参数"""
        if not email or not email.strip():
            raise ValidationException("邮箱不能为空")
        
        if not username or not username.strip():
            raise ValidationException("用户名不能为空")
        
        if not password or not password.strip():
            raise ValidationException("密码不能为空")
        
        # 验证邮箱格式（通过Email值对象验证）
        try:
            Email(email.strip())
        except ValueError as e:
            raise ValidationException(f"邮箱格式错误: {str(e)}")
        
        # 验证用户名格式（通过Username值对象验证）
        try:
            Username(username.strip())
        except ValueError as e:
            raise ValidationException(f"用户名格式错误: {str(e)}")
        
        # 验证密码格式（通过Password值对象验证）
        try:
            Password.create_from_plain(password)
        except ValueError as e:
            raise ValidationException(f"密码格式错误: {str(e)}")
        
        # 验证手机号格式（如果提供）
        if phone_number is not None:
            try:
                PhoneNumber(phone_number)
            except ValueError as e:
                raise ValidationException(f"手机号格式错误: {str(e)}")
    
    # ================================
    # 用户登录相关方法
    # ================================
    
    def login(self, identifier: str, password: str) -> Dict[str, Any]:
        """
        用户登录
        
        Args:
            identifier: 登录标识符（用户名或邮箱）
            password: 密码
            
        Returns:
            Dict[str, Any]: 登录结果，包含用户信息和错误码
        """
        try:
            # 1. 验证输入参数
            if not identifier or not identifier.strip():
                return {
                    "success": False,
                    "message": get_error_message(ErrorCode.INVALID_CREDENTIALS),
                    "code": ErrorCode.INVALID_CREDENTIALS
                }
            
            if not password or not password.strip():
                return {
                    "success": False,
                    "message": get_error_message(ErrorCode.INVALID_CREDENTIALS),
                    "code": ErrorCode.INVALID_CREDENTIALS
                }
            
            # 2. 查找用户（尝试用户名和邮箱）
            user = self._find_user_by_identifier(identifier.strip())
            
            if user is None:
                return {
                    "success": False,
                    "message": get_error_message(ErrorCode.USER_NOT_FOUND),
                    "code": ErrorCode.USER_NOT_FOUND
                }
            
            # 3. 验证密码
            if not user.verify_password(password):
                return {
                    "success": False,
                    "message": get_error_message(ErrorCode.INVALID_CREDENTIALS),
                    "code": ErrorCode.INVALID_CREDENTIALS
                }
            
            # 4. 生成JWT token
            user_data = self._user_to_dict(user, include_sensitive=False)
            tokens = generate_token(user_data)
            
            # 5. 返回登录结果
            return {
                "success": True,
                "message": get_error_message(ErrorCode.LOGIN_SUCCESS),
                "code": ErrorCode.LOGIN_SUCCESS,
                "data": {
                    "user": user_data,
                    "token": tokens['access_token'],
                    "refresh_token": tokens['refresh_token']
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": get_error_message(ErrorCode.SYSTEM_ERROR),
                "code": ErrorCode.SYSTEM_ERROR
            }
    
    def _find_user_by_identifier(self, identifier: str) -> Optional[User]:
        """根据标识符查找用户（用户名或邮箱）"""
        # 首先尝试按用户名查找
        user = self._user_repository.find_by_username(identifier)
        if user:
            return user
        
        # 然后尝试按邮箱查找
        user = self._user_repository.find_by_email(identifier)
        return user
    
    # ================================
    # 密码管理相关方法
    # ================================
    
    def change_password(
        self,
        user_id: int,
        old_password: Optional[str],
        new_password: str
    ) -> Dict[str, Any]:
        """
        修改密码
        
        Args:
            user_id: 用户ID
            old_password: 旧密码（忘记密码场景下可为None）
            new_password: 新密码
            
        Returns:
            Dict[str, Any]: 修改结果
            
        Raises:
            UserNotFoundException: 用户未找到
            InvalidCredentialsException: 旧密码错误
            ValidationException: 验证失败
        """
        try:
            # 1. 验证输入参数
            if not new_password or not new_password.strip():
                raise ValidationException("新密码不能为空")
            
            # 2. 查找用户
            user = self._user_repository.find_by_id(user_id)
            if user is None:
                raise UserNotFoundException(f"用户ID {user_id} 不存在")
            
            # 3. 验证旧密码（仅在提供旧密码时验证）
            if old_password is not None:
                if not old_password.strip():
                    raise ValidationException("旧密码不能为空")
                if not user.verify_password(old_password):
                    raise InvalidCredentialsException("旧密码错误")
            
            # 5. 验证新密码格式
            try:
                new_password_obj = Password.create_from_plain(new_password)
            except ValueError as e:
                raise ValidationException(f"新密码格式错误: {str(e)}")
            
            # 6. 检查新密码是否与当前密码相同（仅在提供旧密码时检查）
            if old_password is not None and user.verify_password(new_password):
                raise ValidationException("新密码不能与当前密码相同")
            
            # 7. 更新密码
            user.change_password(new_password_obj)
            updated_user = self._user_repository.update(user)
            
            # 8. 返回结果
            return {
                "success": True,
                "message": "密码修改成功",
                "code": ErrorCode.SUCCESS,
                "user": self._user_to_dict(updated_user, include_sensitive=False)
            }
            
        except (UserNotFoundException, InvalidCredentialsException, ValidationException):
            raise
        except Exception as e:
            raise UserServiceException(f"密码修改失败: {str(e)}")
    
    def reset_password_by_email(
        self,
        email: str,
        new_password: str
    ) -> Dict[str, Any]:
        """
        通过邮箱重置密码（忘记密码场景）
        
        Args:
            email: 邮箱地址
            new_password: 新密码
            
        Returns:
            Dict[str, Any]: 重置结果
            
        Raises:
            UserNotFoundException: 用户未找到
            ValidationException: 验证失败
        """
        try:
            # 1. 验证输入参数
            if not email or not email.strip():
                raise ValidationException("邮箱不能为空")
            
            if not new_password or not new_password.strip():
                raise ValidationException("新密码不能为空")
            
            # 2. 根据邮箱查找用户
            user = self._user_repository.find_by_email(email.strip())
            if user is None:
                raise UserNotFoundException(f"邮箱 '{email}' 对应的用户不存在")
            
            # 3. 验证新密码格式
            try:
                new_password_obj = Password.create_from_plain(new_password)
            except ValueError as e:
                raise ValidationException(f"新密码格式错误: {str(e)}")
            
            # 4. 检查新密码是否与当前密码相同
            if user.verify_password(new_password):
                raise ValidationException("新密码不能与当前密码相同")
            
            # 5. 更新密码
            user.change_password(new_password_obj)
            updated_user = self._user_repository.update(user)
            
            # 6. 返回结果
            return {
                "success": True,
                "message": "密码重置成功",
                "code": ErrorCode.SUCCESS,
                "user": self._user_to_dict(updated_user, include_sensitive=False)
            }
            
        except (UserNotFoundException, ValidationException):
            raise
        except Exception as e:
            raise UserServiceException(f"密码重置失败: {str(e)}")
    
    # ================================
    # 用户信息管理相关方法
    # ================================
    
    def get_user_by_id(self, user_id: int) -> Dict[str, Any]:
        """
        根据ID获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict[str, Any]: 用户信息
            
        Raises:
            UserNotFoundException: 用户未找到
        """
        try:
            user = self._user_repository.find_by_id(user_id)
            if user is None:
                raise UserNotFoundException(f"用户ID {user_id} 不存在")
            
            return {
                "success": True,
                "user": self._user_to_dict(user, include_sensitive=False)
            }
            
        except UserNotFoundException:
            raise
        except Exception as e:
            raise UserServiceException(f"获取用户信息失败: {str(e)}")
    
    def get_user_by_username(self, username: str) -> Dict[str, Any]:
        """
        根据用户名获取用户信息
        
        Args:
            username: 用户名
            
        Returns:
            Dict[str, Any]: 用户信息
            
        Raises:
            UserNotFoundException: 用户未找到
        """
        try:
            user = self._user_repository.find_by_username(username)
            if user is None:
                raise UserNotFoundException(f"用户名 '{username}' 不存在")
            
            return {
                "success": True,
                "user": self._user_to_dict(user, include_sensitive=False)
            }
            
        except UserNotFoundException:
            raise
        except Exception as e:
            raise UserServiceException(f"获取用户信息失败: {str(e)}")
    
    def update_user_profile(
        self,
        user_id: int,
        email: Optional[str] = None,
        phone_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        更新用户资料
        
        Args:
            user_id: 用户ID
            email: 新邮箱（可选）
            phone_number: 新手机号（可选）
            
        Returns:
            Dict[str, Any]: 更新结果
            
        Raises:
            UserNotFoundException: 用户未找到
            UserAlreadyExistsException: 邮箱已被使用
            ValidationException: 验证失败
        """
        try:
            # 1. 查找用户
            user = self._user_repository.find_by_id(user_id)
            if user is None:
                raise UserNotFoundException(f"用户ID {user_id} 不存在")
            
            # 2. 更新邮箱
            if email is not None:
                email = email.strip()
                if email:
                    # 验证邮箱格式
                    try:
                        new_email = Email(email)
                    except ValueError as e:
                        raise ValidationException(f"邮箱格式错误: {str(e)}")
                    
                    # 检查邮箱是否已被其他用户使用
                    existing_user = self._user_repository.find_by_email(email)
                    if existing_user and existing_user.uid != user_id:
                        raise UserAlreadyExistsException(f"邮箱 '{email}' 已被其他用户使用")
                    
                    user.update_email(new_email)
            
            # 3. 更新手机号
            if phone_number is not None:
                try:
                    new_phone = PhoneNumber(phone_number)
                    user.update_phone_number(new_phone)
                except ValueError as e:
                    raise ValidationException(f"手机号格式错误: {str(e)}")
            
            # 4. 保存更新
            updated_user = self._user_repository.update(user)
            
            # 5. 返回结果
            return {
                "success": True,
                "message": "用户资料更新成功",
                "user": self._user_to_dict(updated_user, include_sensitive=False)
            }
            
        except (UserNotFoundException, UserAlreadyExistsException, ValidationException):
            raise
        except Exception as e:
            raise UserServiceException(f"用户资料更新失败: {str(e)}")
    
    # ================================
    # Token管理相关方法
    # ================================
    
    def add_tokens(self, user_id: int, amount: Decimal) -> Dict[str, Any]:
        """
        为用户添加Token
        
        Args:
            user_id: 用户ID
            amount: 添加的Token数量
            
        Returns:
            Dict[str, Any]: 操作结果
            
        Raises:
            UserNotFoundException: 用户未找到
            ValidationException: 验证失败
        """
        try:
            # 1. 验证参数
            if amount <= 0:
                raise ValidationException("Token数量必须大于0")
            
            # 2. 查找用户
            user = self._user_repository.find_by_id(user_id)
            if user is None:
                raise UserNotFoundException(f"用户ID {user_id} 不存在")
            
            # 3. 添加Token
            user.add_tokens(amount)
            updated_user = self._user_repository.update(user)
            
            # 4. 返回结果
            return {
                "success": True,
                "message": f"成功添加 {amount} Token",
                "user": self._user_to_dict(updated_user, include_sensitive=False)
            }
            
        except (UserNotFoundException, ValidationException):
            raise
        except Exception as e:
            raise UserServiceException(f"添加Token失败: {str(e)}")
    
    def consume_tokens(self, user_id: int, amount: Decimal) -> Dict[str, Any]:
        """
        消费用户Token
        
        Args:
            user_id: 用户ID
            amount: 消费的Token数量
            
        Returns:
            Dict[str, Any]: 操作结果
            
        Raises:
            UserNotFoundException: 用户未找到
            ValidationException: 验证失败或余额不足
        """
        try:
            # 1. 验证参数
            if amount <= 0:
                raise ValidationException("Token数量必须大于0")
            
            # 2. 查找用户
            user = self._user_repository.find_by_id(user_id)
            if user is None:
                raise UserNotFoundException(f"用户ID {user_id} 不存在")
            
            # 3. 消费Token（允许余额变为负数）
            user.consume_tokens(amount)
            updated_user = self._user_repository.update(user)
            
            # 5. 返回结果
            return {
                "success": True,
                "message": f"成功消费 {amount} Token",
                "user": self._user_to_dict(updated_user, include_sensitive=False)
            }
            
        except (UserNotFoundException, ValidationException):
            raise
        except Exception as e:
            raise UserServiceException(f"消费Token失败: {str(e)}")
    
    # ================================
    # 辅助方法
    # ================================
    
    def _user_to_dict(self, user: User, include_sensitive: bool = False) -> Dict[str, Any]:
        """将用户对象转换为字典"""
        result = {
            "id": user.uid,  # 前端期望的是id字段
            "uid": user.uid,  # 保留uid字段以保持兼容性
            "email": user.email.value if user.email else None,
            "username": user.username.value if user.username else None,
            "rest_token_count": float(user.rest_token_count.value),
            "phone_number": user.phone_number.value if user.phone_number else None,
            "is_verified": user.is_verified,
            "is_complete": user.is_complete()
        }
        
        if include_sensitive and user.password:
            result["password_hash"] = user.password.hashed_value
        
        return result
    
    def list_users(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        获取用户列表
        
        Args:
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            Dict[str, Any]: 用户列表
        """
        try:
            users = self._user_repository.find_all(limit=limit, offset=offset)
            
            return {
                "success": True,
                "users": [self._user_to_dict(user, include_sensitive=False) for user in users],
                "count": len(users)
            }
            
        except Exception as e:
            raise UserServiceException(f"获取用户列表失败: {str(e)}")