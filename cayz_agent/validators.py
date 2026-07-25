"""
输入验证模块

- validate_user_input: 验证用户输入（空、长度、注入防护）
- validate_search_query: 验证搜索查询参数
"""

import logging
import re

logger = logging.getLogger(__name__)

# 限制参数
MAX_INPUT_LENGTH = 2000  # 单次用户输入最大字符数
MAX_SEARCH_QUERY_LENGTH = 500  # 搜索查询最大字符数
MAX_KNOWLEDGE_TEXT_LENGTH = 100000  # 知识库文档最大字符数（比对话输入宽松）
MAX_BATCH_ITEMS = 50  # 批量上传单次最大文档数
# M1 会话 ID 格式约束：仅允许字母/数字/连字符/下划线，长度 8-128
# 防止换行符（日志注入）、超长字符串（DoS）、特殊字符（路径/SQL 元字符）
MAX_THREAD_ID_LENGTH = 128
MIN_THREAD_ID_LENGTH = 8
_THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{8,128}$")

# 常见 prompt injection 模式（启发式，非穷举）
# P2 增强：新增中文注入模式检测，覆盖"忽略上述指令"、"你现在是DAN"等中文变体
_INJECTION_PATTERNS = [
    # 英文模式
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+(?:dan|developer|admin)", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"\[.*?system.*?\]", re.IGNORECASE),
    # P2 中文模式：指令覆盖类
    re.compile(r"(忽略|无视|跳过|不要遵守|放弃)(以上|上述|之前|前面|所有)?的?(所有)?(指令|指示|规则|限制|约束|提示)"),
    re.compile(r"(忘记|清除|重置|清空)(之前|先前|上面|所有)?的?(所有)?(指令|指示|对话|上下文|设定|记忆|历史)"),
    re.compile(r"(从现在起|从现在开始|从这一刻起)(你|请)?(是|扮演|充当|变成|作为)"),
    re.compile(r"(你现在|你现在是|从现在起你是)(DAN|开发者|管理员|root|admin|无限制)", re.IGNORECASE),
    # P2 中文模式：角色劫持类
    re.compile(r"(请|你)?(扮演|充当|模拟|假装)(开发者|管理员|root|DAN|无限制AI)", re.IGNORECASE),
    re.compile(r"(解除|取消|关闭|移除)(安全|内容|伦理|道德)?(限制|限制器|过滤|护栏|约束)"),
    # P2 中文模式：系统提示注入
    re.compile(r"【.*?(系统|指令|规则).*?】"),
    re.compile(r"(系统提示|系统指令|系统消息)\s*[:：]"),
    # P2 中文模式：越狱相关
    re.compile(r"(越狱|jailbreak|break.?out)", re.IGNORECASE),
    re.compile(r"(开发者模式|developer.?mode|god.?mode|上帝模式)", re.IGNORECASE),
]


class InputValidationError(ValueError):
    """输入验证失败"""


def validate_user_input(text: str) -> str:
    """
    验证用户输入文本。

    Returns:
        清理后的文本（strip 后）
    Raises:
        InputValidationError: 输入为空 / 过长 / 含注入特征
    """
    if not text or not text.strip():
        raise InputValidationError("输入不能为空")

    text = text.strip()

    if len(text) > MAX_INPUT_LENGTH:
        raise InputValidationError(f"输入过长（{len(text)} 字符），最大允许 {MAX_INPUT_LENGTH} 字符")

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning("检测到疑似 prompt injection，已拦截: %s", text[:80])
            raise InputValidationError("输入包含不被允许的指令注入特征")

    return text


def validate_search_query(query: str) -> str:
    """
    验证 web_search 工具的查询参数。

    Returns:
        清理后的查询
    Raises:
        InputValidationError: 查询为空 / 过长
    """
    if not query or not query.strip():
        raise InputValidationError("搜索查询不能为空")

    query = query.strip()

    if len(query) > MAX_SEARCH_QUERY_LENGTH:
        raise InputValidationError(f"搜索查询过长（{len(query)} 字符），最大允许 {MAX_SEARCH_QUERY_LENGTH} 字符")

    return query


def validate_knowledge_text(text: str) -> str:
    """
    验证知识库文档文本。

    与 validate_user_input 的区别：
    - 字符上限更大（MAX_KNOWLEDGE_TEXT_LENGTH），适合长文档
    - 不做 prompt injection 检测（知识库文档可能合法包含类似文本）

    Returns:
        清理后的文本
    Raises:
        InputValidationError: 文本为空 / 过长
    """
    if not text or not text.strip():
        raise InputValidationError("文档内容不能为空")

    text = text.strip()

    if len(text) > MAX_KNOWLEDGE_TEXT_LENGTH:
        raise InputValidationError(f"文档过长（{len(text)} 字符），最大允许 {MAX_KNOWLEDGE_TEXT_LENGTH} 字符")

    return text


def validate_thread_id(thread_id: str) -> str:
    """
    M1 会话 ID 格式校验：仅允许字母/数字/连字符/下划线，长度 8-128。

    防护目标：
    - 换行符 / 控制字符：防止日志注入（伪造日志条目）
    - 超长字符串：防止 DoS / 日志膨胀
    - 特殊字符（../、SQL 元字符等）：防止路径穿越 / 下游 metadata 污染

    Returns:
        校验通过的 thread_id
    Raises:
        InputValidationError: thread_id 格式非法
    """
    if not thread_id:
        raise InputValidationError("thread_id 不能为空")

    if len(thread_id) < MIN_THREAD_ID_LENGTH:
        raise InputValidationError(f"thread_id 过短（{len(thread_id)} 字符），最小 {MIN_THREAD_ID_LENGTH} 字符")

    if len(thread_id) > MAX_THREAD_ID_LENGTH:
        raise InputValidationError(f"thread_id 过长（{len(thread_id)} 字符），最大 {MAX_THREAD_ID_LENGTH} 字符")

    if not _THREAD_ID_PATTERN.match(thread_id):
        raise InputValidationError("thread_id 格式非法：仅允许字母、数字、连字符、下划线")

    return thread_id
