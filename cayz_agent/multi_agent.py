"""
多 Agent 协作架构

基于 LangGraph StateGraph 实现"路由 Agent + 专业子 Agent"的协作模式：

┌─────────┐     ┌──────────────┐
│  User   │────▶│  Router Agent │ (意图识别)
└─────────┘     └──────┬───────┘
                       │
   ┌──────────────┬────┴────────────┬──────────────┐
   ▼              ▼                 ▼              ▼
┌─────────────┐┌───────────┐ ┌───────────┐ ┌─────────────┐
│Knowledge Agt││Search Agt │ │ Chat Agent │ │Business Agt │
│  (RAG 检索)  ││(联网搜索)  │ │ (通用对话)  │ │(CRM/通知/邮件)│
└─────────────┘└───────────┘ └───────────┘ └─────────────┘
   │              │              │              │
   └──────────────┴──────────────┴──────────────┘
                       │
                       ▼
                   ┌────────┐
                   │  User  │
                   └────────┘

路由 Agent 分析用户最新消息，输出路由标签：
- "knowledge"：私有知识/项目文档/内部资料 → 知识库 Agent
- "search"：天气/新闻/股票/实时信息 → 搜索 Agent
- "chat"：闲聊/通用问答/时间 → 通用 Agent
- "business"：CRM 客户/订单查询、企业微信通知、邮件发送 → 业务集成 Agent
"""
import logging
from typing import Annotated, TypedDict, Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .llm import create_llm
from .tools import AGENT_TOOLS
from .config import get_settings
from .monitor import record_route
from .cache import get_llm_cache
from .graph import build_checkpointer, log_token_usage

logger = logging.getLogger(__name__)

# 路由标签
RouteType = Literal["knowledge", "search", "chat", "business"]

# 合法路由集合（用于校验 route_after_tools 返回值，防止拼接出非法节点名）
_VALID_ROUTES = ("knowledge", "search", "chat", "business")


def _make_router_cache_key(user_message: str) -> str:
    """构造 router LLM 调用的缓存键。

    路由判定只依赖用户消息内容（ROUTER_PROMPT 固定），因此可直接用 prompt+message 哈希。
    """
    import hashlib
    payload = f"{ROUTER_PROMPT}|{user_message}"
    return "router:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MultiAgentState(TypedDict):
    """多 Agent 共享状态"""
    messages: Annotated[list, add_messages]
    # 路由 Agent 输出的意图标签
    route: str


# ===== 创建共享 LLM（延迟初始化，避免模块导入时产生副作用）=====
# 模块级名字保留为 None，首次使用时通过 _get_llm*() 创建；
# 测试可通过 patch("cayz_agent.multi_agent._llm", ...) 直接覆盖。
_llm = None
_llm_with_tools = None


def _get_llm():
    """获取 LLM 单例（延迟初始化）"""
    global _llm
    if _llm is None:
        _llm = create_llm()
    return _llm


def _get_llm_with_tools():
    """获取绑定了工具的 LLM 单例（延迟初始化）"""
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = _get_llm().bind_tools(AGENT_TOOLS)
    return _llm_with_tools


# ===== 路由 Agent =====
ROUTER_PROMPT = """你是一个意图识别路由器。根据用户最新消息判断应交给哪个专业 Agent 处理。

只输出以下四个标签之一，不要输出任何其他内容：
- knowledge：用户询问项目文档、产品手册、内部知识、已上传的资料等私有知识
- search：用户询问天气、新闻、股票、体育、实时信息等需要联网搜索的问题
- chat：用户闲聊、问候、询问时间、通用问答等
- business：用户要求查询 CRM 客户/订单信息、发送企业微信通知、发送邮件等业务系统集成操作

判断规则：
1. 如果用户提到"知识库""文档""手册""资料""产品说明"等词，路由到 knowledge
2. 如果用户询问"今天""最新""现在""实时"等需要外部数据的问题，路由到 search
3. 如果用户提到"客户""订单""CRM""企业微信""通知""邮件""发送邮件""SMTP"等词，路由到 business
4. 其余情况路由到 chat
"""


def router_node(state: MultiAgentState):
    """路由 Agent：分析用户意图，输出路由标签"""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if not isinstance(last_message, HumanMessage):
        # 非用户消息，默认走 chat
        return {"route": "chat"}

    user_text = last_message.content if isinstance(last_message.content, str) else str(last_message.content)
    cache_key = _make_router_cache_key(user_text)

    try:
        # 尝试从 LLM 缓存命中（路由判定是纯文本响应，无副作用，适合短期复用）
        cache = get_llm_cache()
        cached_route = cache.get(cache_key)
        if cached_route is not None:
            logger.info("路由 Agent 缓存命中: %s", cached_route)
            record_route(cached_route)
            return {"route": cached_route}

        route_prompt = SystemMessage(content=ROUTER_PROMPT)
        response = _get_llm().invoke([route_prompt, last_message])
        route_text = response.content.strip().lower()

        # 提取有效路由标签（business 优先于 chat 检查，避免被 chat 兜底吞掉）
        route = "chat"
        for valid in ("knowledge", "search", "business", "chat"):
            if valid in route_text:
                route = valid
                break

        logger.info("路由 Agent 判定意图: %s", route)
        # 写入缓存（无论识别成功还是回退 chat，都缓存以减少下次相同请求的 LLM 调用）
        cache.set(cache_key, route)
        record_route(route)
        return {"route": route}

    except Exception as e:
        logger.warning("路由 Agent 异常，默认走 chat: %s", e)
        record_route("chat")
        return {"route": "chat"}


def route_after_router(state: MultiAgentState) -> str:
    """条件边：根据路由标签分流到对应子 Agent"""
    return state.get("route", "chat")


# ===== 知识库 Agent =====
KNOWLEDGE_PROMPT = """你是一个知识库问答专家。当用户询问私有知识、项目文档、内部资料时：
1. 必须先调用 knowledge_search 工具从知识库检索相关文档
2. 如果知识库无相关信息，提示用户使用 knowledge_upload 工具上传文档
3. 基于检索到的内容，用简洁专业的中文回答

如果用户希望上传知识，调用 knowledge_upload 工具。
"""


def knowledge_agent_node(state: MultiAgentState):
    """知识库 Agent：处理私有知识问答

    P2-4 修复：LLM 调用失败时返回降级 AIMessage，避免异常冒泡导致整个图崩溃。
    """
    messages = state["messages"]
    system = SystemMessage(content=KNOWLEDGE_PROMPT)
    try:
        response = _get_llm_with_tools().invoke([system] + messages)
        log_token_usage(response)
        logger.info("知识库 Agent 响应")
        return {"messages": [response]}
    except Exception:
        logger.exception("知识库 Agent 调用失败，返回降级响应")
        return {"messages": [AIMessage(content="抱歉，知识库服务暂时不可用，请稍后重试。")]}


# ===== 搜索 Agent =====
SEARCH_PROMPT = """你是一个联网搜索专家。当用户询问天气、新闻、股票、实时信息时：
1. 必须调用 web_search 工具联网搜索，不能编造答案
2. 基于搜索结果，用简洁专业的中文回答

如果用户询问时间，调用 get_current_time 工具。
"""


def search_agent_node(state: MultiAgentState):
    """搜索 Agent：处理实时信息查询

    P2-4 修复：LLM 调用失败时返回降级 AIMessage。
    """
    messages = state["messages"]
    system = SystemMessage(content=SEARCH_PROMPT)
    try:
        response = _get_llm_with_tools().invoke([system] + messages)
        log_token_usage(response)
        logger.info("搜索 Agent 响应")
        return {"messages": [response]}
    except Exception:
        logger.exception("搜索 Agent 调用失败，返回降级响应")
        return {"messages": [AIMessage(content="抱歉，搜索服务暂时不可用，请稍后重试。")]}


# ===== 通用 Agent =====
CHAT_PROMPT = """你是一个友好的中文智能助手。处理用户的闲聊、问候、通用问答。
你可以使用 get_current_time 工具查询时间。
请用简洁、自然的中文回答用户。
"""


def chat_agent_node(state: MultiAgentState):
    """通用 Agent：处理闲聊和通用问答

    P2-4 修复：LLM 调用失败时返回降级 AIMessage。
    """
    messages = state["messages"]
    system = SystemMessage(content=CHAT_PROMPT)
    try:
        response = _get_llm_with_tools().invoke([system] + messages)
        log_token_usage(response)
        logger.info("通用 Agent 响应")
        return {"messages": [response]}
    except Exception:
        logger.exception("通用 Agent 调用失败，返回降级响应")
        return {"messages": [AIMessage(content="抱歉，AI 服务暂时不可用，请稍后重试。")]}


# ===== 业务集成 Agent =====
BUSINESS_PROMPT = """你是一个业务系统集成专家。处理用户涉及 CRM、企业微信通知、邮件发送等业务操作：

可用工具及使用场景：
1. crm_query_customer(customer_id)：查询客户基本信息和订单汇总（customer_id 形如 C001）
2. crm_search_customers(keyword)：按姓名/公司/邮箱模糊搜索客户
3. crm_query_order(order_id)：查询订单详情（order_id 形如 ORD-2024-001）
4. send_wecom_notification(message, msg_type="text")：通过企业微信群机器人发送通知
5. send_email(to, subject, body, html=False)：通过 SMTP 发送邮件（to 多地址用逗号分隔）

执行规则：
1. 根据用户请求调用对应工具，不要编造客户/订单/通知结果
2. 工具返回错误时，如实告知用户失败原因（例如未配置 SMTP/Webhook）
3. 涉及多步操作时（如查客户后发通知），逐步执行工具并基于上一步结果决策
4. 用简洁专业的中文回复
"""


def business_agent_node(state: MultiAgentState):
    """业务集成 Agent：处理 CRM/通知/邮件类业务操作

    P2-4 修复：LLM 调用失败时返回降级 AIMessage。
    """
    messages = state["messages"]
    system = SystemMessage(content=BUSINESS_PROMPT)
    try:
        response = _get_llm_with_tools().invoke([system] + messages)
        log_token_usage(response)
        logger.info("业务集成 Agent 响应")
        return {"messages": [response]}
    except Exception:
        logger.exception("业务集成 Agent 调用失败，返回降级响应")
        return {"messages": [AIMessage(content="抱歉，业务集成服务暂时不可用，请稍后重试。")]}


# ===== 子 Agent 后续路由 =====
def should_continue_from_sub_agent(state: MultiAgentState):
    """子 Agent 响应后，如果需要调用工具则进入 tools 节点，否则结束"""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ===== 构建多 Agent 图 =====
# P2-6 修复：缓存编译后的多 Agent 图实例，避免每次调用都新建 SQLite 连接导致泄漏
_compiled_multi_agent_graph = None
_multi_agent_lock = __import__("threading").Lock()


def create_multi_agent_graph():
    """
    构建并编译多 Agent 协作图

    图结构：
        START → router → (knowledge | search | chat | business)
                         → tools → 对应子 agent → END

    持久化与单 Agent 图保持一致：通过 build_checkpointer() 读取 checkpoint_backend 配置，
    避免多 Agent 模式下重启会话丢失。

    P2-6 修复：缓存编译结果（与 graph.py 的 create_graph 一致），避免每次调用
    都新建 SQLite 连接导致连接泄漏与「database is locked」。
    """
    global _compiled_multi_agent_graph
    if _compiled_multi_agent_graph is not None:
        return _compiled_multi_agent_graph
    with _multi_agent_lock:
        if _compiled_multi_agent_graph is not None:
            return _compiled_multi_agent_graph
        _compiled_multi_agent_graph = _build_multi_agent_graph()
        return _compiled_multi_agent_graph


def _build_multi_agent_graph():
    """实际构建多 Agent 图（仅由 create_multi_agent_graph 在锁内调用）"""
    from langgraph.prebuilt import ToolNode

    workflow = StateGraph(MultiAgentState)

    # 添加节点
    workflow.add_node("router", router_node)
    workflow.add_node("knowledge_agent", knowledge_agent_node)
    workflow.add_node("search_agent", search_agent_node)
    workflow.add_node("chat_agent", chat_agent_node)
    workflow.add_node("business_agent", business_agent_node)
    workflow.add_node("tools", ToolNode(AGENT_TOOLS, handle_tool_errors=True))

    # 入口
    workflow.set_entry_point("router")

    # 路由 Agent → 四个子 Agent
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "knowledge": "knowledge_agent",
            "search": "search_agent",
            "chat": "chat_agent",
            "business": "business_agent",
        },
    )

    # 每个子 Agent → tools 或 END
    for sub_agent in (
        "knowledge_agent",
        "search_agent",
        "chat_agent",
        "business_agent",
    ):
        workflow.add_conditional_edges(
            sub_agent,
            should_continue_from_sub_agent,
            {"tools": "tools", END: END},
        )

    # tools → 回到对应的子 Agent（通过 route 标签判断）
    def route_after_tools(state: MultiAgentState) -> str:
        """工具执行后返回到对应的子 Agent（带合法性校验，防止拼接出非法节点名）"""
        route = state.get("route", "chat")
        if route not in _VALID_ROUTES:
            logger.warning("route_after_tools 收到非法 route=%s，回退到 chat_agent", route)
            route = "chat"
        return f"{route}_agent"

    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "knowledge_agent": "knowledge_agent",
            "search_agent": "search_agent",
            "chat_agent": "chat_agent",
            "business_agent": "business_agent",
        },
    )

    return workflow.compile(checkpointer=build_checkpointer())
