# -*- coding: utf-8 -*-
"""
User Controller
用户管理控制器，处理用户相关的API请求
"""

from fastapi import APIRouter, HTTPException, Query, Form, Depends
from pydantic import BaseModel
from typing import Optional, Union

from service.user import UserService, UserNotFoundException, ValidationException, UserServiceException, UserAlreadyExistsException
from service.check import VerificationCheckService
from infrastructure.auth.fastapi_auth import get_current_user, get_current_user_optional, get_user_id_from_payload
from infrastructure.sms import get_sms_verification_service
from constants.error_codes import ErrorCode
from constants.error_messages import get_error_message

# 创建路由器
user_router = APIRouter(prefix="/user", tags=["用户管理"])

# 请求模型
class SendVerificationCodeRequest(BaseModel):
    email: str

class SendSMSVerificationCodeRequest(BaseModel):
    phone: str

class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str
    verification_code: str
    phone_number: Optional[int] = None

class LoginRequest(BaseModel):
    identifier: str  # 邮箱或用户名
    password: str

class ChangePasswordByCodeRequest(BaseModel):
    email: str
    new_password: str
    verification_code: str

class ChangePasswordByOldRequest(BaseModel):
    user_id: int
    old_password: str
    new_password: str

class VerifySMSCodeRequest(BaseModel):
    phone: str
    code: str

class SMSVerificationStatusRequest(BaseModel):
    phone: str

# 响应模型
class ApiResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    code: Optional[int] = None
    data: Optional[dict] = None

# 服务实例
user_service = UserService()
verification_service = VerificationCheckService()
sms_verification_service = get_sms_verification_service()

@user_router.post("/send-verification-code", response_model=ApiResponse)
async def send_verification_code(email: str = Form(...)):
    """发送邮箱验证码"""
    try:
        # 调用服务层发送验证码
        result = verification_service.send_email_verification_code(email)
        
        # 将字符串code转换为数值code
        code_mapping = {
            'SUCCESS': ErrorCode.VERIFICATION_SEND_SUCCESS,
            'EMPTY_EMAIL': ErrorCode.EMPTY_EMAIL,
            'INVALID_EMAIL': ErrorCode.INVALID_EMAIL,
            'TOO_FREQUENT': ErrorCode.TOO_FREQUENT,
            'SEND_FAILED': ErrorCode.SEND_FAILED,
            'SYSTEM_ERROR': ErrorCode.SYSTEM_ERROR,
            'UNKNOWN': ErrorCode.UNKNOWN
        }
        
        code_str = result.get('code', 'UNKNOWN')
        numeric_code = code_mapping.get(code_str, ErrorCode.UNKNOWN)
        
        return ApiResponse(
            success=result['success'],
            message=result['message'],
            code=numeric_code,
            data={
                'expire_minutes': result.get('expire_minutes')
            } if result['success'] else None
        )
            
    except ValidationException as e:
        raise HTTPException(
            status_code=400,
            detail={
                'success': False,
                'message': str(e),
                'code': ErrorCode.BAD_REQUEST,
                'data': None
            }
        )
    except UserServiceException as e:
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'message': str(e),
                'code': ErrorCode.SYSTEM_ERROR,
                'data': None
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'message': get_error_message(ErrorCode.SYSTEM_ERROR),
                'code': ErrorCode.SYSTEM_ERROR,
                'data': None
            }
        )

@user_router.post("/register", response_model=ApiResponse)
async def register(
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    verification_code: str = Form(..., alias="verfication_code"),  # 支持拼写错误的字段名
    phone_number: Optional[int] = Form(None)
):
    """用户注册"""
    try:
        # 验证验证码
        verify_result = verification_service.verify_email_code(email, verification_code)
        if not verify_result['success']:
            raise HTTPException(
                status_code=400,
                detail={
                    'success': False,
                    'message': f'验证码验证失败: {verify_result["message"]}',
                    'code': ErrorCode.VERIFICATION_FAILED,
                    'data': None
                }
            )
        
        # 注册用户
        register_result = user_service.register_user(
            email=email,
            username=username,
            password=password,
            phone_number=phone_number
        )
        
        return ApiResponse(
            success=register_result['success'],
            message=register_result['message'],
            code=register_result.get('code', 'UNKNOWN'),
            data=register_result.get('data')
        )
            
    except HTTPException:
        raise
    except ValidationException as e:
        raise HTTPException(
            status_code=400,
            detail={
                'success': False,
                'message': str(e),
                'code': ErrorCode.BAD_REQUEST,
                'data': None
            }
        )
    except UserAlreadyExistsException as e:
        raise HTTPException(
            status_code=400,
            detail={
                'success': False,
                'message': str(e),
                'code': ErrorCode.USER_ALREADY_EXISTS,
                'data': None
            }
        )
    except UserServiceException as e:
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'message': str(e),
                'code': ErrorCode.SYSTEM_ERROR,
                'data': None
            }
        )
    except Exception as e:
        # 记录详细的错误信息用于调试
        import traceback
        print(f"注册用户时发生错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误堆栈: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'message': get_error_message(ErrorCode.SYSTEM_ERROR),
                'code': ErrorCode.SYSTEM_ERROR,
                'data': None
            }
        )

@user_router.post("/login", response_model=ApiResponse)
async def login(
    identifier: str = Form(...),
    password: str = Form(...)
):
    """用户登录"""
    try:
        # 用户登录
        login_result = user_service.login(identifier, password)
        
        return ApiResponse(
            success=login_result['success'],
            message=login_result['message'],
            code=login_result.get('code'),
            data=login_result.get('data')
        )
            
    except ValidationException as e:
        return ApiResponse(
            success=False,
            message=str(e),
            code=ErrorCode.BAD_REQUEST,
            data=None
        )
    except InvalidCredentialsException as e:
        return ApiResponse(
            success=False,
            message=str(e),
            code=ErrorCode.INVALID_CREDENTIALS,
            data=None
        )
    except UserNotFoundException as e:
        return ApiResponse(
            success=False,
            message=str(e),
            code=ErrorCode.USER_NOT_FOUND,
            data=None
        )
    except UserServiceException as e:
        return ApiResponse(
            success=False,
            message=str(e),
            code=ErrorCode.SYSTEM_ERROR,
            data=None
        )
    except Exception as e:
        return ApiResponse(
            success=False,
            message=get_error_message(ErrorCode.SYSTEM_ERROR),
            code=ErrorCode.SYSTEM_ERROR,
            data=None
        )

@user_router.post("/change-password-by-code", response_model=ApiResponse)
async def change_password_by_code(request: ChangePasswordByCodeRequest):
    """通过验证码修改密码"""
    try:
        # 验证验证码
        verify_result = verification_service.verify_email_code(request.email, request.verification_code)
        if not verify_result['success']:
            raise HTTPException(
                status_code=400,
                detail={
                    'success': False,
                    'message': f'验证码验证失败: {verify_result["message"]}',
                    'code': ErrorCode.VERIFICATION_FAILED,
                    'data': None
                }
            )
        
        # 使用新的重置密码方法（通过邮箱）
        reset_result = user_service.reset_password_by_email(request.email, request.new_password)
        
        return ApiResponse(
            success=reset_result['success'],
            message=reset_result['message'],
            code=reset_result.get('code', ErrorCode.SUCCESS),
            data=reset_result.get('data')
        )
            
    except HTTPException:
        raise
    except UserNotFoundException as e:
        raise HTTPException(
            status_code=404,
            detail={
                'success': False,
                'message': str(e),
                'code': ErrorCode.USER_NOT_FOUND,
                'data': None
            }
        )
    except ValidationException as e:
        raise HTTPException(
            status_code=400,
            detail={
                'success': False,
                'message': str(e),
                'code': ErrorCode.BAD_REQUEST,
                'data': None
            }
        )
    except UserServiceException as e:
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'message': str(e),
                'code': ErrorCode.SYSTEM_ERROR,
                'data': None
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'message': get_error_message(ErrorCode.SYSTEM_ERROR),
                'code': ErrorCode.SYSTEM_ERROR,
                'data': None
            }
        )

@user_router.post("/change-password-by-old", response_model=ApiResponse)
async def change_password_by_old(
    request: ChangePasswordByOldRequest,
    current_user: dict = Depends(get_current_user)
):
    """通过旧密码修改密码（需要认证）"""
    try:
        # 验证用户只能修改自己的密码
        current_user_id = int(get_user_id_from_payload(current_user))
        if current_user_id != request.user_id:
            raise HTTPException(
                status_code=403,
                detail={
                    'success': False,
                    'message': '只能修改自己的密码',
                    'code': 403,
                    'data': None
                }
            )
        
        # 修改密码
        change_result = user_service.change_password(request.user_id, request.old_password, request.new_password)
        
        return ApiResponse(
            success=change_result['success'],
            message=change_result['message'],
            code=change_result.get('code', ErrorCode.SUCCESS),
            data=change_result.get('user')
        )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'message': get_error_message(ErrorCode.SYSTEM_ERROR),
                'code': ErrorCode.SYSTEM_ERROR,
                'data': None
            }
        )

@user_router.get("/info/{user_id}", response_model=ApiResponse)
async def get_user_info(
    user_id: int,
    current_user: dict = Depends(get_current_user)
):
    """获取用户信息（需要认证）"""
    try:
        # 验证用户只能获取自己的信息
        current_user_id = int(get_user_id_from_payload(current_user))
        if current_user_id != user_id:
            raise HTTPException(
                status_code=403,
                detail={
                    'success': False,
                    'message': '只能获取自己的用户信息',
                    'code': 403,
                    'data': None
                }
            )
        
        # 获取用户信息
        user_result = user_service.get_user_by_id(user_id)
        
        return {
            'success': user_result['success'],
            'message': '获取用户信息成功',
            'code': ErrorCode.SUCCESS,
            'data': user_result.get('user')
        }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"获取用户信息时发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'message': get_error_message(ErrorCode.SYSTEM_ERROR),
                'code': ErrorCode.SYSTEM_ERROR,
                'data': None
            }
        )

@user_router.post("/verification-status", response_model=ApiResponse)
async def get_verification_status(request: SendVerificationCodeRequest):
    """获取验证码状态"""
    try:
        # 获取验证码状态
        status_result = verification_service.get_verification_status(request.email)
        
        # 处理code字段，确保是数值类型
        code_value = status_result.get('code', 'UNKNOWN')
        if isinstance(code_value, str):
            # 如果是字符串，转换为对应的ErrorCode
            code_mapping = {
                'SUCCESS': ErrorCode.SUCCESS,
                'EMPTY_EMAIL': ErrorCode.EMPTY_EMAIL,
                'INVALID_EMAIL': ErrorCode.INVALID_EMAIL,
                'TOO_FREQUENT': ErrorCode.TOO_FREQUENT,
                'SEND_FAILED': ErrorCode.SEND_FAILED,
                'SYSTEM_ERROR': ErrorCode.SYSTEM_ERROR,
                'UNKNOWN': ErrorCode.UNKNOWN
            }
            code_value = code_mapping.get(code_value, ErrorCode.UNKNOWN)
        
        return ApiResponse(
            success=status_result['success'],
            message=status_result.get('message', ''),
            code=code_value,
            data={
                'has_code': status_result.get('has_code', False),
                'remaining_seconds': status_result.get('remaining_seconds', 0),
                'remaining_minutes': status_result.get('remaining_minutes', 0),
                'current_attempts': status_result.get('current_attempts', 0),
                'max_attempts': status_result.get('max_attempts', 0),
                'remaining_attempts': status_result.get('remaining_attempts', 0),
                'can_resend': status_result.get('can_resend', True),
                'resend_remaining_seconds': status_result.get('resend_remaining_seconds', 0)
            } if status_result['success'] else None
        )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'message': get_error_message(ErrorCode.SYSTEM_ERROR),
                'code': ErrorCode.SYSTEM_ERROR,
                'data': None
            }
        )

# 新增手机验证相关接口

@user_router.post("/send-sms-verification-code", response_model=ApiResponse)
async def send_sms_verification_code(phone: str = Form(...)):
    """发送短信验证码"""
    try:
        result = sms_verification_service.send_verification_code(phone)
        
        if result['success']:
            return ApiResponse(
                success=True,
                message=result['message'],
                data={
                    'expire_minutes': result.get('expire_minutes', 5)
                }
            )
        else:
            return ApiResponse(
                success=False,
                message=result['message'],
                code=400,
                data={
                    'error_code': result.get('error_code'),
                    'remaining_time': result.get('remaining_time')
                }
            )
    except Exception as e:
        return ApiResponse(
            success=False,
            message="系统异常，请稍后重试",
            code=500
        )

@user_router.post("/verify-sms-code", response_model=ApiResponse)
async def verify_sms_code(request: VerifySMSCodeRequest):
    """验证短信验证码"""
    try:
        result = sms_verification_service.verify_code(request.phone, request.code)
        
        if result['success']:
            return ApiResponse(
                success=True,
                message=result['message']
            )
        else:
            return ApiResponse(
                success=False,
                message=result['message'],
                code=400,
                data={
                    'error_code': result.get('error_code')
                }
            )
    except Exception as e:
        return ApiResponse(
            success=False,
            message="系统异常，请稍后重试",
            code=500
        )

@user_router.post("/sms-verification-status", response_model=ApiResponse)
async def get_sms_verification_status(request: SMSVerificationStatusRequest):
    """获取短信验证状态"""
    try:
        result = sms_verification_service.get_verification_status(request.phone)
        
        if result['success']:
            return ApiResponse(
                success=True,
                message="获取状态成功",
                data=result['data']
            )
        else:
            return ApiResponse(
                success=False,
                message=result['message'],
                code=400,
                data={
                    'error_code': result.get('error_code')
                }
            )
    except Exception as e:
        return ApiResponse(
            success=False,
            message="系统异常，请稍后重试",
            code=500
        )

@user_router.get("/check-phone-verified/{phone}", response_model=ApiResponse)
async def check_phone_verified(phone: str):
    """检查手机号是否已验证"""
    try:
        is_verified = sms_verification_service.is_phone_verified(phone)
        
        return ApiResponse(
            success=True,
            message="查询成功",
            data={
                'phone': phone,
                'is_verified': is_verified
            }
        )
    except Exception as e:
        return ApiResponse(
            success=False,
            message="系统异常，请稍后重试",
            code=500
        )