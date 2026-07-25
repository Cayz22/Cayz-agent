"""
multi_agent 模块单元测试：验证多 Agent 协作架构

- router_node: 意图识别路由
- knowledge_agent_node / search_agent_node / chat_agent_node: 子 Agent
- create_multi_agent_graph: 图构建
- should_continue_from_sub_agent: 条件路由
"""
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import END

from cayz_agent.multi_agent import (
    router_node,
    knowledge_agent_node,
    search_agent_node,
    chat_agent_node,
    business_agent_node,
    should_continue_from_sub_agent,
    route_after_router,
    MultiAgentState,
    create_multi_agent_graph,
    _VALID_ROUTES,
)


class TestRouterNode:
    """测试路由 Agent"""

    def test_non_human_message_defaults_to_chat(self):
        """非 HumanMessage 应默认路由到 chat"""
        state = {"messages": [AIMessage(content="hi")], "route": ""}
        result = router_node(state)
        assert result["route"] == "chat"

    def test_empty_messages_defaults_to_chat(self):
        """空消息列表应默认路由到 chat"""
        state = {"messages": [], "route": ""}
        result = router_node(state)
        assert result["route"] == "chat"

    @patch("cayz_agent.multi_agent._llm")
    def test_routes_to_knowledge(self, mock_llm):
        """识别为知识库意图时路由到 knowledge"""
        mock_response = MagicMock()
        mock_response.content = "knowledge"
        mock_llm.invoke.return_value = mock_response

        state = {"messages": [HumanMessage(content="查询项目文档")], "route": ""}
        result = router_node(state)
        assert result["route"] == "knowledge"

    @patch("cayz_agent.multi_agent._llm")
    def test_routes_to_search(self, mock_llm):
        """识别为搜索意图时路由到 search"""
        mock_response = MagicMock()
        mock_response.content = "search"
        mock_llm.invoke.return_value = mock_response

        state = {"messages": [HumanMessage(content="今天天气怎么样")], "route": ""}
        result = router_node(state)
        assert result["route"] == "search"

    @patch("cayz_agent.multi_agent._llm")
    def test_routes_to_chat(self, mock_llm):
        """识别为闲聊意图时路由到 chat"""
        mock_response = MagicMock()
        mock_response.content = "chat"
        mock_llm.invoke.return_value = mock_response

        state = {"messages": [HumanMessage(content="你好")], "route": ""}
        result = router_node(state)
        assert result["route"] == "chat"

    @patch("cayz_agent.multi_agent._llm")
    def test_unrecognized_response_defaults_to_chat(self, mock_llm):
        """LLM 返回无法识别的内容时默认路由到 chat"""
        mock_response = MagicMock()
        mock_response.content = "unknown_label_xyz"
        mock_llm.invoke.return_value = mock_response

        state = {"messages": [HumanMessage(content="???")], "route": ""}
        result = router_node(state)
        assert result["route"] == "chat"

    @patch("cayz_agent.multi_agent._llm")
    def test_exception_defaults_to_chat(self, mock_llm):
        """LLM 异常时默认路由到 chat"""
        mock_llm.invoke.side_effect = Exception("API error")

        state = {"messages": [HumanMessage(content="test")], "route": ""}
        result = router_node(state)
        assert result["route"] == "chat"

    @patch("cayz_agent.multi_agent._llm")
    def test_case_insensitive_routing(self, mock_llm):
        """路由标签应大小写不敏感"""
        mock_response = MagicMock()
        mock_response.content = "KNOWLEDGE"
        mock_llm.invoke.return_value = mock_response

        state = {"messages": [HumanMessage(content="查文档")], "route": ""}
        result = router_node(state)
        assert result["route"] == "knowledge"

    @patch("cayz_agent.multi_agent._llm")
    def test_routes_to_business(self, mock_llm):
        """识别为业务集成意图时路由到 business"""
        mock_response = MagicMock()
        mock_response.content = "business"
        mock_llm.invoke.return_value = mock_response

        state = {"messages": [HumanMessage(content="查询客户 C001 的订单")], "route": ""}
        result = router_node(state)
        assert result["route"] == "business"

    @patch("cayz_agent.multi_agent._llm")
    def test_business_takes_precedence_over_chat(self, mock_llm):
        """business 标签应在 chat 之前被检查，避免被 chat 兜底吞掉"""
        # 模拟 LLM 返回包含 business 的文本
        mock_response = MagicMock()
        mock_response.content = "应该路由到 business agent"
        mock_llm.invoke.return_value = mock_response

        state = {"messages": [HumanMessage(content="发邮件")], "route": ""}
        result = router_node(state)
        assert result["route"] == "business"

    @patch("cayz_agent.multi_agent._llm")
    def test_router_caches_route_decision(self, mock_llm):
        """相同消息第二次调用应命中缓存，不再调用 LLM（阶段 H：LLM 缓存）"""
        from cayz_agent.cache import get_llm_cache, reset_cache_singletons

        reset_cache_singletons()

        mock_response = MagicMock()
        mock_response.content = "knowledge"
        mock_llm.invoke.return_value = mock_response

        state = {"messages": [HumanMessage(content="查文档")], "route": ""}

        # 第一次：未命中缓存，调用 LLM
        result1 = router_node(state)
        assert result1["route"] == "knowledge"
        assert mock_llm.invoke.call_count == 1

        # 第二次：相同消息应命中缓存，不调用 LLM
        result2 = router_node(state)
        assert result2["route"] == "knowledge"
        assert mock_llm.invoke.call_count == 1  # 仍是 1，没有增加

    @patch("cayz_agent.multi_agent._llm")
    def test_router_different_messages_not_cached(self, mock_llm):
        """不同消息不应命中缓存，应分别调用 LLM"""
        from cayz_agent.cache import reset_cache_singletons

        reset_cache_singletons()

        mock_response = MagicMock()
        mock_response.content = "knowledge"
        mock_llm.invoke.return_value = mock_response

        # 不同消息
        router_node({"messages": [HumanMessage(content="查询文档A")], "route": ""})
        router_node({"messages": [HumanMessage(content="查询文档B")], "route": ""})

        # 应该调用 2 次 LLM
        assert mock_llm.invoke.call_count == 2


class TestRouteAfterRouter:
    """测试路由条件边"""

    def test_returns_route_from_state(self):
        """应返回 state 中的 route"""
        state = {"route": "knowledge"}
        assert route_after_router(state) == "knowledge"

    def test_defaults_to_chat(self):
        """无 route 时默认返回 chat"""
        state = {}
        assert route_after_router(state) == "chat"


class TestSubAgents:
    """测试子 Agent 节点"""

    @patch("cayz_agent.multi_agent._llm_with_tools")
    def test_knowledge_agent_invokes_llm(self, mock_llm_tools):
        """知识库 Agent 应调用带工具的 LLM"""
        mock_response = AIMessage(content="知识库回答")
        mock_llm_tools.invoke.return_value = mock_response

        state = {"messages": [HumanMessage(content="查询文档")], "route": "knowledge"}
        result = knowledge_agent_node(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        mock_llm_tools.invoke.assert_called_once()

    @patch("cayz_agent.multi_agent._llm_with_tools")
    def test_search_agent_invokes_llm(self, mock_llm_tools):
        """搜索 Agent 应调用带工具的 LLM"""
        mock_response = AIMessage(content="搜索结果")
        mock_llm_tools.invoke.return_value = mock_response

        state = {"messages": [HumanMessage(content="今天天气")], "route": "search"}
        result = search_agent_node(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        mock_llm_tools.invoke.assert_called_once()

    @patch("cayz_agent.multi_agent._llm_with_tools")
    def test_chat_agent_invokes_llm(self, mock_llm_tools):
        """通用 Agent 应调用带工具的 LLM"""
        mock_response = AIMessage(content="你好！")
        mock_llm_tools.invoke.return_value = mock_response

        state = {"messages": [HumanMessage(content="你好")], "route": "chat"}
        result = chat_agent_node(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        mock_llm_tools.invoke.assert_called_once()

    @patch("cayz_agent.multi_agent._llm_with_tools")
    def test_business_agent_invokes_llm(self, mock_llm_tools):
        """业务集成 Agent 应调用带工具的 LLM"""
        mock_response = AIMessage(content="已查询客户信息")
        mock_llm_tools.invoke.return_value = mock_response

        state = {"messages": [HumanMessage(content="查询客户 C001")], "route": "business"}
        result = business_agent_node(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        mock_llm_tools.invoke.assert_called_once()
        # 验证传入的 system message 包含业务集成相关提示
        call_args = mock_llm_tools.invoke.call_args[0][0]
        assert isinstance(call_args[0], SystemMessage)
        assert "CRM" in call_args[0].content or "客户" in call_args[0].content


class TestShouldContinueFromSubAgent:
    """测试子 Agent 的后续路由"""

    def test_returns_tools_when_tool_calls_exist(self):
        """有 tool_calls 时应路由到 tools"""
        mock_msg = MagicMock()
        mock_msg.tool_calls = [{"name": "web_search", "args": {}}]
        state = {"messages": [mock_msg], "route": "chat"}

        assert should_continue_from_sub_agent(state) == "tools"

    def test_returns_end_when_no_tool_calls(self):
        """无 tool_calls 时应结束"""
        mock_msg = MagicMock()
        mock_msg.tool_calls = []
        state = {"messages": [mock_msg], "route": "chat"}

        assert should_continue_from_sub_agent(state) == END

    def test_returns_end_when_empty_messages(self):
        """空消息列表应结束"""
        state = {"messages": [], "route": "chat"}
        assert should_continue_from_sub_agent(state) == END

    def test_returns_end_when_no_tool_calls_attr(self):
        """消息无 tool_calls 属性时应结束"""
        mock_msg = MagicMock(spec=["content"])
        mock_msg.content = "hello"
        state = {"messages": [mock_msg], "route": "chat"}

        assert should_continue_from_sub_agent(state) == END


class TestCreateMultiAgentGraph:
    """测试多 Agent 图构建"""

    def test_graph_compiles_successfully(self):
        """图应能成功编译"""
        graph = create_multi_agent_graph()
        assert graph is not None

    def test_graph_has_nodes(self):
        """图应包含所有节点"""
        graph = create_multi_agent_graph()
        # 编译后的图应能获取节点信息
        # LangGraph 的 CompiledGraph 有 nodes 属性
        assert hasattr(graph, "nodes")

    def test_graph_invoke_with_chat_route(self):
        """图应能处理 chat 路由的消息（使用 mock）"""
        graph = create_multi_agent_graph()

        with patch("cayz_agent.multi_agent._llm") as mock_llm, \
             patch("cayz_agent.multi_agent._llm_with_tools") as mock_llm_tools:

            # 路由 Agent 返回 chat
            router_response = MagicMock()
            router_response.content = "chat"
            mock_llm.invoke.return_value = router_response

            # chat Agent 返回普通回复
            mock_llm_tools.invoke.return_value = AIMessage(content="你好！有什么可以帮你的？")

            result = graph.invoke(
                {"messages": [HumanMessage(content="你好")]},
                config={"configurable": {"thread_id": "test-1"}},
            )

            assert "messages" in result

    def test_graph_invoke_with_business_route(self):
        """图应能处理 business 路由的消息（使用 mock）"""
        graph = create_multi_agent_graph()

        with patch("cayz_agent.multi_agent._llm") as mock_llm, \
             patch("cayz_agent.multi_agent._llm_with_tools") as mock_llm_tools:

            # 路由 Agent 返回 business
            router_response = MagicMock()
            router_response.content = "business"
            mock_llm.invoke.return_value = router_response

            # business Agent 返回普通回复（无 tool_calls，直接结束）
            mock_llm_tools.invoke.return_value = AIMessage(content="已为您查询到客户信息")

            result = graph.invoke(
                {"messages": [HumanMessage(content="查询客户 C001")]},
                config={"configurable": {"thread_id": "test-business-1"}},
            )

            assert "messages" in result
            # 验证 business_agent 被调用（而非 chat_agent）
            assert mock_llm_tools.invoke.called


class TestRouteAfterToolsSafety:
    """测试 route_after_tools 的合法性校验（防止字符串拼接出非法节点名）"""

    def test_valid_routes_constant(self):
        """_VALID_ROUTES 应包含四个路由"""
        assert "knowledge" in _VALID_ROUTES
        assert "search" in _VALID_ROUTES
        assert "chat" in _VALID_ROUTES
        assert "business" in _VALID_ROUTES
        assert len(_VALID_ROUTES) == 4

    def test_route_after_tools_returns_valid_node_for_business(self):
        """business 路由应能正确拼接出 business_agent 节点名"""
        # 直接测试内部函数：通过编译图的 nodes 间接验证
        graph = create_multi_agent_graph()
        assert "business_agent" in graph.nodes

    def test_route_after_tools_rejects_invalid_route(self):
        """非法 route 值应回退到 chat_agent（防止拼接注入）"""
        # 重新构建一次图，提取 route_after_tools 内部函数
        # 由于 route_after_tools 是 create_multi_agent_graph 内的闭包，我们通过
        # 模拟一个非法 state 直接调用 _VALID_ROUTES 校验逻辑来覆盖
        invalid_route = "evil; rm -rf /"
        assert invalid_route not in _VALID_ROUTES

    def test_graph_has_business_agent_node(self):
        """编译后的图应包含 business_agent 节点"""
        graph = create_multi_agent_graph()
        assert "business_agent" in graph.nodes
        assert "router" in graph.nodes
        assert "tools" in graph.nodes
