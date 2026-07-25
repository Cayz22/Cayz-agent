"""
sanitizers 模块单元测试：验证敏感信息脱敏与有害内容检测
"""
import pytest
from cayz_agent.sanitizers import sanitize_text, sanitize_exception, contains_harmful_content


class TestSanitizeText:
    """测试 sanitize_text 函数"""

    @pytest.mark.parametrize("text", ["", None])
    def test_empty_input(self, text):
        """空字符串或 None 应原样返回"""
        assert sanitize_text(text) == text

    def test_normal_text_unchanged(self):
        """普通文本不应被修改"""
        text = "今天天气很好，适合户外运动。"
        assert sanitize_text(text) == text

    def test_openai_key_masked(self):
        """OpenAI sk- 开头的密钥应被脱敏"""
        text = "我的密钥是 sk-abcdefghij1234567890xyz"
        result = sanitize_text(text)
        assert "sk-abcdefghij1234567890xyz" not in result
        assert "敏感信息已隐藏" in result

    def test_tavily_key_masked(self):
        """Tavily tvly- 开头的密钥应被脱敏"""
        text = "tavily key: tvly-abcdefghij"
        result = sanitize_text(text)
        assert "tvly-abcdefghij" not in result
        assert "敏感信息已隐藏" in result

    def test_bearer_token_masked(self):
        """Bearer 令牌应被脱敏"""
        text = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz12"
        result = sanitize_text(text)
        assert "Bearer abcdefghijklmnopqrstuvwxyz12" not in result
        assert "敏感信息已隐藏" in result

    def test_password_masked(self):
        """password= 后的值应被脱敏"""
        text = "password=mysecretpass123"
        result = sanitize_text(text)
        assert "mysecretpass123" not in result

    def test_secret_masked(self):
        """secret= 后的值应被脱敏"""
        text = "secret=supersecretvalue"
        result = sanitize_text(text)
        assert "supersecretvalue" not in result

    def test_api_key_masked(self):
        """api_key= 后的值应被脱敏"""
        text = "api_key=ak1234567890abcdef"
        result = sanitize_text(text)
        assert "ak1234567890abcdef" not in result

    def test_aws_key_masked(self):
        """AWS AccessKey 应被脱敏"""
        text = "AWS key: AKIA1234567890ABCDEF"
        result = sanitize_text(text)
        assert "AKIA1234567890ABCDEF" not in result
        assert "敏感信息已隐藏" in result

    def test_aliyun_key_masked(self):
        """阿里云 LTAI 开头的 Key 应被脱敏"""
        text = "LTAI1234567890ab"
        result = sanitize_text(text)
        assert "LTAI1234567890ab" not in result

    def test_multiple_secrets_in_one_text(self):
        """同一段文本中多个敏感信息都应被脱敏"""
        text = "key1=sk-abcdefghijklmnopqrst, key2=tvly-abcdefghij"
        result = sanitize_text(text)
        assert "sk-abcdefghijklmnopqrst" not in result
        assert "tvly-abcdefghij" not in result

    def test_mixed_normal_and_sensitive(self):
        """正常文本与敏感信息混合时，仅脱敏敏感部分"""
        text = "配置信息：sk-abcdefghijklmnopqrst，端口是 8080"
        result = sanitize_text(text)
        assert "sk-abcdefghijklmnopqrst" not in result
        assert "8080" in result
        assert "配置信息" in result


class TestSanitizeException:
    """测试 sanitize_exception 函数"""

    def test_masks_windows_path(self):
        """应隐藏 Windows 路径"""
        exc = FileNotFoundError("无法找到 D:\\secrets\\config.json")
        result = sanitize_exception(exc)
        assert "D:\\secrets\\config.json" not in result
        assert "本地路径已隐藏" in result

    def test_masks_api_key_in_exception(self):
        """应隐藏异常中的 API Key"""
        exc = RuntimeError("failed with key=sk-abcdefghijklmnopqrst")
        result = sanitize_exception(exc)
        assert "sk-abcdefghijklmnopqrst" not in result


class TestContainsHarmfulContent:
    """测试 contains_harmful_content 函数"""

    def test_normal_text_not_harmful(self):
        """正常文本不应被标记为有害"""
        assert contains_harmful_content("今天天气很好") is False
        assert contains_harmful_content("") is False
        assert contains_harmful_content(None) is False

    def test_rm_rf_detected(self):
        """rm -rf / 应被检测"""
        assert contains_harmful_content("执行 rm -rf / 删除所有文件") is True

    def test_format_detected(self):
        """format c: 应被检测"""
        assert contains_harmful_content("请运行 format c: 来修复") is True

    def test_mkfs_detected(self):
        """mkfs 应被检测"""
        assert contains_harmful_content("用 mkfs.ext4 /dev/sda1 格式化") is True

    def test_fork_bomb_detected(self):
        """fork bomb 应被检测"""
        assert contains_harmful_content(":(){ :|:& };:") is True

    def test_safe_command_not_flagged(self):
        """安全命令不应被误判"""
        assert contains_harmful_content("ls -la /home/user") is False
        assert contains_harmful_content("rm -rf /tmp/test_dir") is False


class TestP2ExtendedSanitizers:
    """P2 扩展脱敏：测试新增的 JWT/私钥/连接串/手机号/身份证/邮箱/中文密钥等模式"""

    def test_jwt_masked(self):
        """JWT 令牌应被脱敏"""
        from cayz_agent.sanitizers import sanitize_text
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456ghi789"
        result = sanitize_text(jwt)
        assert jwt not in result
        assert "敏感信息已隐藏" in result

    def test_private_key_masked(self):
        """PEM 私钥块应被脱敏"""
        from cayz_agent.sanitizers import sanitize_text
        pem = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA1234567890abcdefghijklmnopqrstuvwxyz\n"
            "-----END RSA PRIVATE KEY-----"
        )
        result = sanitize_text(pem)
        assert "MIIEowIBAAKCAQEA" not in result
        assert "敏感信息已隐藏" in result

    def test_connection_string_masked(self):
        """数据库连接串密码应被脱敏"""
        from cayz_agent.sanitizers import sanitize_text
        conn = "postgres://user:secretpass@localhost:5432/db"
        result = sanitize_text(conn)
        assert "secretpass" not in result
        assert "postgres://user" in result  # 用户名保留
        assert "@localhost" in result  # 主机保留

    def test_phone_masked(self):
        """中国手机号应被脱敏"""
        from cayz_agent.sanitizers import sanitize_text
        text = "联系电话：13812345678"
        result = sanitize_text(text)
        assert "13812345678" not in result
        assert "敏感信息已隐藏" in result

    def test_id_card_masked(self):
        """身份证号应被脱敏"""
        from cayz_agent.sanitizers import sanitize_text
        text = "身份证号：110101199001011234"
        result = sanitize_text(text)
        assert "110101199001011234" not in result

    def test_email_partial_masked(self):
        """邮箱应部分脱敏（保留前3字符与域名）"""
        from cayz_agent.sanitizers import sanitize_text
        text = "联系邮箱：john.doe@example.com"
        result = sanitize_text(text)
        assert "john.doe@" not in result
        assert "@example.com" in result

    def test_chinese_credential_masked(self):
        """中文密钥描述应被脱敏"""
        from cayz_agent.sanitizers import sanitize_text
        text = "密码: mySecretPass123"
        result = sanitize_text(text)
        assert "mySecretPass123" not in result
        assert "敏感信息已隐藏" in result

    def test_aws_key_masked(self):
        """AWS AccessKey 应被脱敏"""
        from cayz_agent.sanitizers import sanitize_text
        text = "AWS Key: AKIAIOSFODNN7EXAMPLE"
        result = sanitize_text(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result

    def test_detect_sensitive_info_empty(self):
        """空文本不检测到敏感信息"""
        from cayz_agent.sanitizers import detect_sensitive_info
        assert detect_sensitive_info("") == []
        assert detect_sensitive_info("普通文本无敏感信息") == []

    def test_detect_sensitive_info_phone(self):
        """检测手机号类型"""
        from cayz_agent.sanitizers import detect_sensitive_info
        assert "phone" in detect_sensitive_info("手机：13812345678")

    def test_detect_sensitive_info_multiple(self):
        """检测多种敏感信息类型"""
        from cayz_agent.sanitizers import detect_sensitive_info
        text = "电话 13812345678，邮箱 test@example.com，key=sk-abcdefghij1234567890"
        found = detect_sensitive_info(text)
        assert "phone" in found
        assert "email" in found
        assert "api_key" in found


class TestP2SanitizingLogFilter:
    """P2 日志过滤器：测试日志自动脱敏"""

    def test_filter_masks_msg(self):
        """日志 record.msg 中的敏感信息应被脱敏"""
        import logging
        from cayz_agent.sanitizers import SanitizingLogFilter

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="API Key: sk-abcdefghij1234567890xyz",
            args=None, exc_info=None,
        )
        f = SanitizingLogFilter()
        assert f.filter(record) is True
        assert "sk-abcdefghij1234567890xyz" not in record.msg
        assert "敏感信息已隐藏" in record.msg

    def test_filter_masks_args(self):
        """日志 args 中的敏感信息应被脱敏"""
        import logging
        from cayz_agent.sanitizers import SanitizingLogFilter

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="请求头 %s",
            args=("Bearer abcdefghijklmnopqrstuvwxyz12",),
            exc_info=None,
        )
        f = SanitizingLogFilter()
        f.filter(record)
        formatted = record.msg % record.args
        assert "Bearer abcdefghijklmnopqrstuvwxyz12" not in formatted
