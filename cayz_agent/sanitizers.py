"""
安全脱敏模块：统一处理敏感信息过滤

P2 安全修复：
- 扩展正则覆盖：JWT、私钥、数据库连接串、手机号、邮箱、身份证、中文 Key 描述
- 新增 sanitize_log：日志专用脱敏（保留可读性，隐藏完整值）
- 新增 SanitizingLogFilter：日志过滤器，自动对 record.msg/args 脱敏

- sanitize_text: 替换 API Key、密码、令牌等敏感信息
- sanitize_exception: 对异常信息脱敏（含本地路径）
- contains_harmful_content: 检测有害内容特征
"""

import logging
import re

# ============================================================
# 敏感信息检测正则（P2 扩展覆盖）
# ============================================================

# 通用密钥/令牌格式（原有 + 扩展）
SENSITIVE_PATTERN = re.compile(
    r"(sk-[a-zA-Z0-9_\-]{20,}"
    r"|tvly-[a-zA-Z0-9_\-]{10,}"
    r'|key=[\'"]?[a-zA-Z0-9]{10,}'
    r"|Bearer\s+[a-zA-Z0-9_\-\.]{20,}"
    r'|password=[\'"]?[^\s\'"]{8,}'
    r'|token=[\'"]?[a-zA-Z0-9_\-\.]{20,}'
    r'|secret=[\'"]?[^\s\'"]{8,}'
    r'|api[_-]?key=[\'"]?[a-zA-Z0-9]{10,})',
    re.IGNORECASE,
)

# AWS / 阿里云 AccessKey 格式
AWS_KEY_PATTERN = re.compile(r"(AKIA[0-9A-Z]{16}|LTAI[0-9A-Za-z]{12,})")

# P2 新增：JWT（三段式 base64.urlsafe，以 ey 开头）
JWT_PATTERN = re.compile(r"eyJ[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}")

# P2 新增：PEM 私钥块
PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
)

# P2 新增：数据库连接串中的密码（postgres://user:pass@host, mongodb://user:pass@host 等）
CONN_STRING_PATTERN = re.compile(r"((?:postgres|mysql|mongodb|redis|amqp)://[^\s:]+:)[^\s@]+(@[^\s/]+)")

# P2 新增：中国手机号（1开头11位，前后非数字边界）
PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")

# P2 新增：中国身份证号（18位，最后一位可能是X）
ID_CARD_PATTERN = re.compile(
    r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
)

# P2 新增：邮箱地址（用于日志脱敏，保留域名隐藏用户名）
EMAIL_PATTERN = re.compile(r"\b([a-zA-Z0-9._%+-]{1,3})[a-zA-Z0-9._%+-]*@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b")

# P2 新增：中文密钥描述（"密码: xxx"、"密钥: xxx"、"口令: xxx"）
CN_SENSITIVE_PATTERN = re.compile(
    r"((?:密码|密钥|口令|令牌|凭证|授权码|验证码)\s*[:：]\s*)[^\s，。,.\n]{4,}",
    re.IGNORECASE,
)

# P2 新增：腾讯云 / 华为云 SecretKey 格式
CLOUD_KEY_PATTERN = re.compile(
    r"(AKID[a-zA-Z0-9]{32,}"  # 腾讯云 SecretId
    r"|AKIDAPP[a-zA-Z0-9]{13}"  # 腾讯云 AppId
    r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"  # UUID 格式密钥
)

REPLACEMENT = "***[敏感信息已隐藏]***"


def sanitize_text(text: str) -> str:
    """
    对文本进行脱敏处理，替换其中的敏感信息（API Key、密码、JWT、私钥、手机号等）。

    P2 扩展：覆盖 JWT、PEM 私钥、数据库连接串、手机号、身份证、邮箱、中文密钥描述、
    腾讯云/华为云 Key 格式。
    """
    if not text:
        return text
    text = SENSITIVE_PATTERN.sub(REPLACEMENT, text)
    text = AWS_KEY_PATTERN.sub(REPLACEMENT, text)
    text = JWT_PATTERN.sub(REPLACEMENT, text)
    text = PRIVATE_KEY_PATTERN.sub(REPLACEMENT, text)
    text = CONN_STRING_PATTERN.sub(r"\1***\2", text)
    text = PHONE_PATTERN.sub(REPLACEMENT, text)
    text = ID_CARD_PATTERN.sub(REPLACEMENT, text)
    text = EMAIL_PATTERN.sub(r"\1***@\2", text)
    text = CN_SENSITIVE_PATTERN.sub(r"\1" + REPLACEMENT, text)
    text = CLOUD_KEY_PATTERN.sub(REPLACEMENT, text)
    return text


def sanitize_exception(exc: Exception) -> str:
    """
    对异常信息进行脱敏，防止内部 URL、Key 等泄露
    """
    msg = str(exc)
    # 过滤可能出现在异常中的内部路径
    msg = re.sub(r"[A-Za-z]:\\[^\s]+", "[本地路径已隐藏]", msg)  # Windows 路径
    msg = re.sub(r"/[A-Za-z0-9_\-./]+", "[路径已隐藏]", msg)  # Unix 路径
    return sanitize_text(msg)


def sanitize_log(msg: str) -> str:
    """日志专用脱敏（与 sanitize_text 一致，语义分离便于独立调整策略）"""
    return sanitize_text(msg)


class SanitizingLogFilter(logging.Filter):
    """日志过滤器：自动对 log record 的 msg 与 args 脱敏。

    P2 日志泄露修复：在 logger 输出前拦截，确保 API Key / 密码 / 手机号等
    不会以明文写入日志文件或 ELK 等日志聚合系统。

    用法：
        logging.getLogger().addFilter(SanitizingLogFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # 脱敏格式化模板
        if isinstance(record.msg, str):
            record.msg = sanitize_log(record.msg)
        # 脱敏位置参数
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: sanitize_log(str(v)) if isinstance(v, str) else v for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(sanitize_log(arg) if isinstance(arg, str) else arg for arg in record.args)
        return True


# 有害内容特征（启发式检测，用于输出审查）
_HARMFUL_PATTERNS = [
    re.compile(r"rm\s+-rf\s+/(?:\s|$)", re.IGNORECASE),  # 危险删除命令（仅根目录）
    re.compile(r"format\s+[c-z]:", re.IGNORECASE),  # 格式化磁盘
    re.compile(r"mkfs\.\w+\s+/dev/", re.IGNORECASE),  # Linux 格式化
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:", re.IGNORECASE),  # fork bomb
]


def contains_harmful_content(text: str) -> bool:
    """
    检测文本中是否包含有害/危险内容特征。

    Returns:
        True 如果检测到有害内容
    """
    if not text:
        return False
    return any(pattern.search(text) for pattern in _HARMFUL_PATTERNS)


def detect_sensitive_info(text: str) -> list[str]:
    """检测文本中包含的敏感信息类型（P2 知识库敏感检测）。

    Returns:
        检测到的敏感信息类型列表（如 ["phone", "id_card", "api_key"]）。
        空列表表示未检测到敏感信息。
    """
    if not text:
        return []
    found = []
    if SENSITIVE_PATTERN.search(text) or AWS_KEY_PATTERN.search(text) or CLOUD_KEY_PATTERN.search(text):
        found.append("api_key")
    if PHONE_PATTERN.search(text):
        found.append("phone")
    if ID_CARD_PATTERN.search(text):
        found.append("id_card")
    if EMAIL_PATTERN.search(text):
        found.append("email")
    if JWT_PATTERN.search(text):
        found.append("jwt")
    if PRIVATE_KEY_PATTERN.search(text):
        found.append("private_key")
    if CONN_STRING_PATTERN.search(text):
        found.append("connection_string")
    if CN_SENSITIVE_PATTERN.search(text):
        found.append("credential")
    return found
