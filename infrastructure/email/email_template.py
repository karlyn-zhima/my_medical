"""
邮件模板管理模块
用于管理各种邮件模板的HTML内容
"""

class EmailTemplate:
    """邮件模板类，用于生成各种类型的邮件内容"""
    
    @staticmethod
    def verification_code_template(verification_code: str, expire_minutes: int) -> str:
        """
        生成验证码邮件的HTML模板
        
        Args:
            verification_code (str): 验证码
            expire_minutes (int): 过期时间（分钟）
            
        Returns:
            str: HTML格式的邮件内容
        """
        return f"""
        <html>
        <body>
            <h2>邮箱验证码</h2>
            <p>您的验证码是：<strong style="color: #007bff; font-size: 18px;">{verification_code}</strong></p>
            <p>验证码有效期为 {expire_minutes} 分钟，请及时使用。</p>
            <p>如果您没有请求此验证码，请忽略此邮件。</p>
        </body>
        </html>
        """
    
    @staticmethod
    def welcome_template(username: str) -> str:
        """
        生成欢迎邮件的HTML模板
        
        Args:
            username (str): 用户名
            
        Returns:
            str: HTML格式的邮件内容
        """
        return f"""
        <html>
        <body>
            <h2>欢迎注册！</h2>
            <p>亲爱的 {username}，</p>
            <p>欢迎您注册我们的服务！您的账户已经成功创建。</p>
            <p>如有任何问题，请随时联系我们。</p>
        </body>
        </html>
        """
    
    @staticmethod
    def password_reset_template(reset_link: str, expire_minutes: int) -> str:
        """
        生成密码重置邮件的HTML模板
        
        Args:
            reset_link (str): 重置链接
            expire_minutes (int): 过期时间（分钟）
            
        Returns:
            str: HTML格式的邮件内容
        """
        return f"""
        <html>
        <body>
            <h2>密码重置</h2>
            <p>您请求重置密码，请点击下面的链接：</p>
            <p><a href="{reset_link}" style="color: #007bff;">重置密码</a></p>
            <p>此链接将在 {expire_minutes} 分钟后过期。</p>
            <p>如果您没有请求重置密码，请忽略此邮件。</p>
        </body>
        </html>
        """