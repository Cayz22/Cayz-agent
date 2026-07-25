"""
__main__ 模块单元测试：验证 CLI 交互入口

通过 mock input / print / create_graph 验证 main() 的控制流：
- exit/quit 命令应终止循环
- 正常输入应调用 graph.invoke 并打印脱敏后的回复
- 异常应被捕获并打印脱敏错误信息
"""

from unittest.mock import MagicMock, patch

import pytest

from cayz_agent import __version__


class TestMain:
    """测试 __main__.main()"""

    def _import_main(self):
        """动态导入 main 函数，避免模块级副作用"""
        from cayz_agent.__main__ import main

        return main

    def test_exit_command_terminates_loop(self):
        """输入 'exit' 应立即退出循环并打印再见消息"""
        main = self._import_main()

        mock_app = MagicMock()
        with (
            patch("cayz_agent.__main__.get_settings"),
            patch("cayz_agent.__main__.setup_logging"),
            patch("cayz_agent.__main__.create_graph", return_value=mock_app),
            patch("builtins.input", return_value="exit"),
            patch("builtins.print") as mock_print,
        ):
            main()

        # graph.invoke 不应被调用
        mock_app.invoke.assert_not_called()
        # 应打印再见消息
        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        assert "再见" in printed

    def test_quit_command_terminates_loop(self):
        """输入 'quit' 也应退出循环"""
        main = self._import_main()

        with (
            patch("cayz_agent.__main__.get_settings"),
            patch("cayz_agent.__main__.setup_logging"),
            patch("cayz_agent.__main__.create_graph", return_value=MagicMock()),
            patch("builtins.input", return_value="quit"),
            patch("builtins.print"),
        ):
            main()

    def test_exit_case_insensitive(self):
        """'EXIT' 大写也应退出"""
        main = self._import_main()

        mock_app = MagicMock()
        with (
            patch("cayz_agent.__main__.get_settings"),
            patch("cayz_agent.__main__.setup_logging"),
            patch("cayz_agent.__main__.create_graph", return_value=mock_app),
            patch("builtins.input", return_value="EXIT"),
            patch("builtins.print"),
        ):
            main()

        mock_app.invoke.assert_not_called()

    def test_normal_input_invokes_graph_and_prints_response(self):
        """正常输入应调用 graph.invoke 并打印脱敏后的回复"""
        main = self._import_main()

        # 构造 mock app，返回包含 AI 回复的 result
        mock_app = MagicMock()
        mock_ai_message = MagicMock()
        mock_ai_message.content = "你好，我是 cayz-agent"
        mock_app.invoke.return_value = {"messages": [mock_ai_message]}

        # 第一次输入正常消息，第二次输入 exit 退出
        inputs = iter(["你好", "exit"])

        with (
            patch("cayz_agent.__main__.get_settings"),
            patch("cayz_agent.__main__.setup_logging"),
            patch("cayz_agent.__main__.create_graph", return_value=mock_app),
            patch("builtins.input", side_effect=inputs),
            patch("builtins.print") as mock_print,
            patch("cayz_agent.__main__.sanitize_text", side_effect=lambda x: x) as mock_sanitize,
        ):
            main()

        # graph.invoke 应被调用一次
        mock_app.invoke.assert_called_once()
        # sanitize_text 应被调用对 AI 回复脱敏
        mock_sanitize.assert_called_with("你好，我是 cayz-agent")
        # 应打印回复
        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        assert "你好，我是 cayz-agent" in printed

    def test_exception_is_caught_and_sanitized(self):
        """graph.invoke 抛异常时应被捕获，打印脱敏后的错误信息后继续循环"""
        main = self._import_main()

        mock_app = MagicMock()
        mock_app.invoke.side_effect = RuntimeError("Internal error with sk-secret123")

        inputs = iter(["触发错误", "exit"])

        with (
            patch("cayz_agent.__main__.get_settings"),
            patch("cayz_agent.__main__.setup_logging"),
            patch("cayz_agent.__main__.create_graph", return_value=mock_app),
            patch("builtins.input", side_effect=inputs),
            patch("builtins.print") as mock_print,
            patch("cayz_agent.__main__.sanitize_exception", return_value="敏感信息已隐藏") as mock_sanitize_exc,
        ):
            main()

        # sanitize_exception 应被调用
        mock_sanitize_exc.assert_called()
        # 应打印错误信息
        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        assert "敏感信息已隐藏" in printed or "出错" in printed

    def test_startup_banner_printed(self):
        """启动时应打印版本号横幅"""
        main = self._import_main()

        with (
            patch("cayz_agent.__main__.get_settings"),
            patch("cayz_agent.__main__.setup_logging"),
            patch("cayz_agent.__main__.create_graph", return_value=MagicMock()),
            patch("builtins.input", return_value="exit"),
            patch("builtins.print") as mock_print,
        ):
            main()

        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        assert __version__ in printed
        assert "cayz-agent" in printed

    def test_uses_fixed_thread_id(self):
        """CLI 应使用固定的 thread_id 以保持会话记忆"""
        main = self._import_main()

        mock_app = MagicMock()
        mock_ai_message = MagicMock()
        mock_ai_message.content = "ok"
        mock_app.invoke.return_value = {"messages": [mock_ai_message]}

        inputs = iter(["hi", "exit"])

        with (
            patch("cayz_agent.__main__.get_settings"),
            patch("cayz_agent.__main__.setup_logging"),
            patch("cayz_agent.__main__.create_graph", return_value=mock_app),
            patch("builtins.input", side_effect=inputs),
            patch("builtins.print"),
        ):
            main()

        # 验证 invoke 被调用时传入了固定 thread_id
        call_kwargs = mock_app.invoke.call_args.kwargs
        assert call_kwargs["config"]["configurable"]["thread_id"] == "cayz-user-session-001"
