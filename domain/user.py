# -*- coding: utf-8 -*-
"""
User Domain Layer - DDD Architecture Implementation
用户领域层 - 基于领域驱动设计(DDD)的实现
"""

import re
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Dict, Any, List
from datetime import datetime


# ================================
# 值对象 (Value Objects)
# ================================

@dataclass(frozen=True)
class Email:
    """邮箱值对象"""
    value: str
    
    def __post_init__(self):
        if not self._is_valid_email(self.value):
            raise ValueError(f"无效的邮箱格式: {self.value}")
    
    @staticmethod
    def _is_valid_email(email: str) -> bool:
        """验证邮箱格式"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None


@dataclass(frozen=True)
class Username:
    """用户名值对象"""
    value: str
    
    def __post_init__(self):
        if not self._is_valid_username(self.value):
            raise ValueError(f"无效的用户名: {self.value}")
    
    @staticmethod
    def _is_valid_username(username: str) -> bool:
        """验证用户名格式"""
        if not username or len(username) < 3 or len(username) > 50:
            return False
        # 只允许字母、数字、下划线
        pattern = r'^[a-zA-Z0-9_]+$'
        return re.match(pattern, username) is not None


@dataclass(frozen=True)
class Password:
    """密码值对象"""
    hashed_value: str
    
    @classmethod
    def create_from_plain(cls, plain_password: str) -> 'Password':
        """从明文密码创建密码对象"""
        if not cls._is_valid_password(plain_password):
            raise ValueError("密码不符合安全要求")
        
        hashed = cls._hash_password(plain_password)
        return cls(hashed_value=hashed)
    
    @classmethod
    def create_from_hash(cls, hashed_password: str) -> 'Password':
        """从已哈希的密码创建密码对象"""
        return cls(hashed_value=hashed_password)
    
    def verify(self, plain_password: str) -> bool:
        """验证密码"""
        return self._hash_password(plain_password) == self.hashed_value
    
    @staticmethod
    def _is_valid_password(password: str) -> bool:
        """验证密码强度"""
        if len(password) < 8:
            return False
        # 至少包含一个字母和一个数字
        has_letter = any(c.isalpha() for c in password)
        has_digit = any(c.isdigit() for c in password)
        return has_letter and has_digit
    
    @staticmethod
    def _hash_password(password: str) -> str:
        """哈希密码"""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class TokenCount:
    """Token数量值对象"""
    value: Decimal
    
    def __post_init__(self):
        if self.value < Decimal('-5000'):  # 允许一定程度的欠费
            raise ValueError("Token数量不能低于-5000")
    
    def add(self, amount: Decimal) -> 'TokenCount':
        """增加Token"""
        return TokenCount(self.value + amount)
    
    def subtract(self, amount: Decimal) -> 'TokenCount':
        """减少Token（允许余额变为负数）"""
        new_value = self.value - amount
        # 允许余额变为负数，但不能低于-5000
        if new_value < Decimal('-5000'):
            raise ValueError("Token余额不能低于-5000")
        return TokenCount(new_value)
    
    def is_sufficient(self, required: Decimal) -> bool:
        """检查Token是否充足"""
        return self.value >= required


@dataclass(frozen=True)
class PhoneNumber:
    """手机号值对象"""
    value: Optional[int]
    
    def __post_init__(self):
        if self.value is not None and not self._is_valid_phone(self.value):
            raise ValueError(f"无效的手机号: {self.value}")
    
    @staticmethod
    def _is_valid_phone(phone: int) -> bool:
        """验证手机号格式"""
        phone_str = str(phone)
        # 简单验证：11位数字，以1开头
        return len(phone_str) == 11 and phone_str.startswith('1')


# ================================
# 实体 (Entity)
# ================================

class User:
    """用户实体"""
    
    def __init__(
        self,
        uid: Optional[int] = None,
        email: Optional[Email] = None,
        username: Optional[Username] = None,
        password: Optional[Password] = None,
        rest_token_count: Optional[TokenCount] = None,
        phone_number: Optional[PhoneNumber] = None,
        is_verified: bool = False
    ):
        self._uid = uid
        self._email = email
        self._username = username
        self._password = password
        self._rest_token_count = rest_token_count or TokenCount(Decimal('0'))
        self._phone_number = phone_number
        self._is_verified = is_verified
    
    # 属性访问器
    @property
    def uid(self) -> Optional[int]:
        return self._uid
    
    @property
    def email(self) -> Optional[Email]:
        return self._email
    
    @property
    def username(self) -> Optional[Username]:
        return self._username
    
    @property
    def password(self) -> Optional[Password]:
        return self._password
    
    @property
    def rest_token_count(self) -> TokenCount:
        return self._rest_token_count
    
    @property
    def phone_number(self) -> Optional[PhoneNumber]:
        return self._phone_number
    
    @property
    def is_verified(self) -> bool:
        return self._is_verified
    
    # 业务方法
    def update_email(self, new_email: Email):
        """更新邮箱"""
        self._email = new_email
    
    def update_username(self, new_username: Username):
        """更新用户名"""
        self._username = new_username
    
    def change_password(self, new_password: Password):
        """修改密码"""
        self._password = new_password
    
    def update_phone_number(self, new_phone: PhoneNumber):
        """更新手机号"""
        self._phone_number = new_phone
    
    def set_verified(self, verified: bool = True):
        """设置邮箱验证状态"""
        self._is_verified = verified
    
    def add_tokens(self, amount: Decimal):
        """充值Token"""
        if amount <= 0:
            raise ValueError("充值金额必须大于0")
        self._rest_token_count = self._rest_token_count.add(amount)
    
    def consume_tokens(self, amount: Decimal):
        """消费Token"""
        if amount <= 0:
            raise ValueError("消费金额必须大于0")
        self._rest_token_count = self._rest_token_count.subtract(amount)
    
    def verify_password(self, plain_password: str) -> bool:
        """验证密码"""
        if not self._password:
            return False
        return self._password.verify(plain_password)
    
    def is_complete(self) -> bool:
        """检查用户信息是否完整"""
        return all([
            self._email,
            self._username,
            self._password
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于持久化）"""
        return {
            'uid': self._uid,
            'email': self._email.value if self._email else None,
            'username': self._username.value if self._username else None,
            'password': self._password.hashed_value if self._password else None,
            'rest_token_count': self._rest_token_count.value,
            'phone_number': self._phone_number.value if self._phone_number else None,
            'is_verified': self._is_verified
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """从字典创建用户实体"""
        # 处理phone_number的类型转换（数据库中存储为字符串）
        phone_number = None
        if data.get('phone_number'):
            try:
                phone_number = PhoneNumber(int(data['phone_number']))
            except (ValueError, TypeError):
                phone_number = None
        
        return cls(
            uid=data.get('uid'),
            email=Email(data['email']) if data.get('email') else None,
            username=Username(data['username']) if data.get('username') else None,
            password=Password.create_from_hash(data['password']) if data.get('password') else None,
            rest_token_count=TokenCount(Decimal(str(data.get('rest_token_count', 0)))),
            phone_number=phone_number,
            is_verified=bool(data.get('is_verified', False))
        )
    
    def __eq__(self, other):
        """实体相等性比较（基于ID）"""
        if not isinstance(other, User):
            return False
        return self._uid is not None and self._uid == other._uid
    
    def __hash__(self):
        """实体哈希（基于ID）"""
        return hash(self._uid) if self._uid else hash(id(self))


# ================================
# 领域服务 (Domain Services)
# ================================

class UserDomainService:
    """用户领域服务"""
    
    @staticmethod
    def create_new_user(
        email: str,
        username: str,
        plain_password: str,
        phone_number: Optional[int] = None,
        is_verified: bool = True
    ) -> User:
        """创建新用户"""
        return User(
            email=Email(email),
            username=Username(username),
            password=Password.create_from_plain(plain_password),
            rest_token_count=TokenCount(Decimal('0')),
            phone_number=PhoneNumber(phone_number) if phone_number else None,
            is_verified=is_verified
        )
    
    @staticmethod
    def is_username_available(username: str, user_repository: 'IUserRepository') -> bool:
        """检查用户名是否可用"""
        existing_user = user_repository.find_by_username(username)
        return existing_user is None
    
    @staticmethod
    def is_email_available(email: str, user_repository: 'IUserRepository') -> bool:
        """检查邮箱是否可用"""
        existing_user = user_repository.find_by_email(email)
        return existing_user is None


# ================================
# 仓储接口 (Repository Interface)
# ================================

class IUserRepository(ABC):
    """用户仓储接口"""
    
    @abstractmethod
    def find_by_id(self, uid: int) -> Optional[User]:
        """根据ID查找用户"""
        pass
    
    @abstractmethod
    def find_by_email(self, email: str) -> Optional[User]:
        """根据邮箱查找用户"""
        pass
    
    @abstractmethod
    def find_by_username(self, username: str) -> Optional[User]:
        """根据用户名查找用户"""
        pass
    
    @abstractmethod
    def save(self, user: User) -> User:
        """保存用户"""
        pass
    
    @abstractmethod
    def update(self, user: User) -> User:
        """更新用户"""
        pass
    
    @abstractmethod
    def delete(self, uid: int) -> bool:
        """删除用户"""
        pass
    
    @abstractmethod
    def find_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[User]:
        """查找所有用户"""
        pass


# ================================
# 仓储实现 (Repository Implementation)
# ================================

class UserRepository(IUserRepository):
    """用户仓储实现"""
    
    def __init__(self, base_repository):
        """
        初始化用户仓储
        :param base_repository: 基础设施层的BaseRepository实例
        """
        self._base_repository = base_repository
    
    def find_by_id(self, uid: int) -> Optional[User]:
        """根据ID查找用户"""
        # 注意：基础设施层的find_by_id使用'id'字段，但用户表使用'uid'
        sql = f"SELECT * FROM {self._base_repository._table_name} WHERE uid = %s"
        results = self._base_repository._database.execute_query(sql, (uid,))
        
        if results:
            return User.from_dict(results[0])
        return None
    
    def find_by_email(self, email: str) -> Optional[User]:
        """根据邮箱查找用户"""
        sql = f"SELECT * FROM {self._base_repository._table_name} WHERE email = %s"
        results = self._base_repository._database.execute_query(sql, (email,))
        
        if results:
            return User.from_dict(results[0])
        return None
    
    def find_by_username(self, username: str) -> Optional[User]:
        """根据用户名查找用户"""
        sql = f"SELECT * FROM {self._base_repository._table_name} WHERE username = %s"
        results = self._base_repository._database.execute_query(sql, (username,))
        
        if results:
            return User.from_dict(results[0])
        return None
    
    def save(self, user: User) -> User:
        """保存用户"""
        user_dict = user.to_dict()
        # 移除uid，因为它是自增的
        user_dict.pop('uid', None)
        
        # 处理手机号格式 - 转换为字符串以避免整数范围问题
        if user_dict.get('phone_number'):
            user_dict['phone_number'] = str(user_dict['phone_number'])
        
        success = self._base_repository.save(user_dict)
        if success:
            # 查询刚插入的用户以获取自增ID
            # 假设用户名是唯一的，通过用户名查找
            return self.find_by_username(user.username.value)
        else:
            raise Exception("用户保存失败")
    
    def update(self, user: User) -> User:
        """更新用户"""
        if not user.uid:
            raise ValueError("无法更新没有ID的用户")
        
        user_dict = user.to_dict()
        uid = user_dict.pop('uid')  # 移除uid，因为它用作WHERE条件
        
        # 处理手机号格式 - 转换为字符串以避免整数范围问题
        if user_dict.get('phone_number'):
            user_dict['phone_number'] = str(user_dict['phone_number'])
        
        # 使用自定义SQL，因为基础仓储使用'id'字段
        set_clause = ', '.join([f"{key} = %s" for key in user_dict.keys()])
        sql = f"UPDATE {self._base_repository._table_name} SET {set_clause} WHERE uid = %s"
        
        params = list(user_dict.values()) + [uid]
        affected_rows = self._base_repository._database.execute_non_query(sql, tuple(params))
        
        if affected_rows > 0:
            return self.find_by_id(uid)
        else:
            raise Exception("用户更新失败")
    
    def delete(self, uid: int) -> bool:
        """删除用户"""
        sql = f"DELETE FROM {self._base_repository._table_name} WHERE uid = %s"
        affected_rows = self._base_repository._database.execute_non_query(sql, (uid,))
        return affected_rows > 0
    
    def find_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[User]:
        """查找所有用户"""
        sql = f"SELECT * FROM {self._base_repository._table_name}"
        params = []
        
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
            
        if offset is not None:
            sql += " OFFSET %s"
            params.append(offset)
        
        results = self._base_repository._database.execute_query(sql, tuple(params) if params else None)
        return [User.from_dict(result) for result in results]