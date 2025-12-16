from enum import IntEnum

class ErrorCode(IntEnum):
    """
    定义系统统一的错误码常量。
    使用 IntEnum 方便在返回 JSON 时转换为整型值。
    
    编码规范：
    - 成功码：2xxxx (200开头)
    - 失败码：4xxxx (400开头)
    - 第四位：功能模块类型
    - 第五位：具体错误类型
    
    功能模块分类：
    - 0x：通用
    - 1x：用户相关
    - 2x：验证码相关
    - 3x：图像相关
    - 4x：模型相关
    """
    
    # ========== 成功码 (2xxxx) ==========
    # 通用成功
    SUCCESS = 20000                    # 通用成功
    
    # 用户相关成功 (201xx)
    REGISTER_SUCCESS = 20101           # 用户注册成功
    LOGIN_SUCCESS = 20102              # 用户登录成功
    PASSWORD_CHANGE_SUCCESS = 20103    # 密码修改成功
    
    # 验证码相关成功 (202xx)
    VERIFICATION_SEND_SUCCESS = 20201  # 验证码发送成功
    VERIFICATION_SUCCESS = 20202       # 验证码验证成功
    
    # 图像相关成功 (203xx)
    IMAGE_PROCESS_SUCCESS = 20301      # 图像处理成功
    
    # 模型相关成功 (204xx)
    MODEL_LOAD_SUCCESS = 20401         # 模型加载成功
    MODEL_RESET_SUCCESS = 20402        # 模型重置成功
    
    # ========== 失败码 (4xxxx) ==========
    # 通用错误 (400xx)
    BAD_REQUEST = 40001                # 请求参数错误
    SYSTEM_ERROR = 40002               # 系统内部错误
    SERVER_ERROR = 40003               # 服务器错误
    UNKNOWN = 40099                    # 未知错误
    
    # 用户相关错误 (401xx)
    USER_NOT_FOUND = 40101             # 用户不存在
    USER_ALREADY_EXISTS = 40102        # 用户已存在
    INVALID_CREDENTIALS = 40103        # 用户名或密码错误
    PASSWORD_TOO_WEAK = 40104          # 密码强度不够
    
    # 验证码相关错误 (402xx)
    EMPTY_EMAIL = 40201                # 邮箱为空
    INVALID_EMAIL = 40202              # 邮箱格式无效
    TOO_FREQUENT = 40203               # 发送过于频繁
    SEND_FAILED = 40204                # 发送失败
    CODE_EXPIRED = 40205               # 验证码已过期
    TOO_MANY_ATTEMPTS = 40206          # 尝试次数过多
    CODE_INCORRECT = 40207             # 验证码错误
    VERIFICATION_FAILED = 40208        # 验证失败
    
    # 图像相关错误 (403xx)
    INVALID_IMAGE = 40301              # 无效的图像
    ENCODING_FAILED = 40302            # 图像编码失败
    IMAGE_TOO_LARGE = 40303            # 图像文件过大
    UNSUPPORTED_FORMAT = 40304         # 不支持的图像格式
    
    # 模型相关错误 (404xx)
    MODEL_NOT_LOADED = 40401           # 模型未加载
    MODEL_LOAD_FAILED = 40402          # 模型加载失败
    RESET_FAILED = 40403               # 模型重置失败
    INFERENCE_FAILED = 40404           # 推理失败