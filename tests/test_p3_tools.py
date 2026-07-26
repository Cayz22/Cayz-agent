"""
P3 新增工具测试：calculate / fetch_url / read_file / write_file / python_repl

覆盖：
1. calculate：基本运算、常量、安全防护（拒绝函数调用/属性访问）
2. fetch_url：URL 校验、HTTP 错误处理、内容提取
3. read_file/write_file：路径白名单、越权防护、编码检测
4. python_repl：基本执行、危险关键字拦截、内置模块访问
5. 权限分级：scope 与工具分配正确
"""

import os
import socket
from unittest.mock import MagicMock, patch

import pytest

from cayz_agent.tools import (
    _validate_workspace_path,
    calculate,
    fetch_url,
    get_tools_for_scope,
    python_repl,
    read_file,
    write_file,
)

# ============================================================
# 1. calculate 工具测试
# ============================================================


class TestCalculate:
    """安全数学表达式求值"""

    def test_basic_arithmetic(self):
        """基本四则运算"""
        assert calculate.invoke({"expression": "2 + 3"}) == "5"
        assert calculate.invoke({"expression": "10 - 4"}) == "6"
        assert calculate.invoke({"expression": "6 * 7"}) == "42"
        assert calculate.invoke({"expression": "15 / 4"}) == "3.75"

    def test_operator_precedence(self):
        """运算符优先级"""
        assert calculate.invoke({"expression": "2 + 3 * 4"}) == "14"
        assert calculate.invoke({"expression": "(2 + 3) * 4"}) == "20"  # 括号
        assert calculate.invoke({"expression": "2 ** 10"}) == "1024"

    def test_floor_div_and_mod(self):
        """整除和取模"""
        assert calculate.invoke({"expression": "17 // 5"}) == "3"
        assert calculate.invoke({"expression": "17 % 5"}) == "2"

    def test_unary_operators(self):
        """一元正负号"""
        assert calculate.invoke({"expression": "-5"}) == "-5"
        assert calculate.invoke({"expression": "-(-5)"}) == "5"
        assert calculate.invoke({"expression": "+5"}) == "5"

    def test_constants_pi_e_tau(self):
        """数学常量 pi/e/tau"""
        import math

        assert calculate.invoke({"expression": "pi"}) == str(math.pi)
        assert calculate.invoke({"expression": "e"}) == str(math.e)
        assert calculate.invoke({"expression": "tau"}) == str(math.tau)

    def test_float_result_integer_formatted(self):
        """整数结果的浮点表达式应去掉 .0"""
        # 4.0 应返回 "4" 而非 "4.0"
        result = calculate.invoke({"expression": "8 / 2"})
        assert result == "4"

    def test_empty_expression_rejected(self):
        """空表达式应被拒绝"""
        assert "为空" in calculate.invoke({"expression": ""})
        assert "为空" in calculate.invoke({"expression": "   "})

    def test_oversized_expression_rejected(self):
        """超长表达式应被拒绝（防 DoS）"""
        long_expr = "1 + " * 100
        result = calculate.invoke({"expression": long_expr})
        assert "过长" in result

    def test_unknown_variable_rejected(self):
        """未知变量应被拒绝"""
        result = calculate.invoke({"expression": "foo + 1"})
        assert "未知变量" in result or "禁止" in result

    def test_function_call_rejected(self):
        """函数调用应被拒绝（防代码执行）"""
        result = calculate.invoke({"expression": "__import__('os')"})
        assert "错误" in result or "禁止" in result

    def test_attribute_access_rejected(self):
        """属性访问应被拒绝"""
        result = calculate.invoke({"expression": "(1).__class__"})
        assert "错误" in result or "禁止" in result

    def test_division_by_zero(self):
        """除零应返回错误"""
        result = calculate.invoke({"expression": "1 / 0"})
        assert "计算错误" in result

    def test_invalid_syntax(self):
        """非法语法应返回错误"""
        result = calculate.invoke({"expression": "2 + + ) 3"})
        assert "错误" in result


# ============================================================
# 2. fetch_url 工具测试
# ============================================================


class TestFetchUrl:
    """网页内容抓取"""

    def test_empty_url_rejected(self):
        """空 URL 应被拒绝"""
        assert "为空" in fetch_url.invoke({"url": ""})

    def test_non_http_url_rejected(self):
        """非 http(s) URL 应被拒绝"""
        result = fetch_url.invoke({"url": "ftp://example.com"})
        assert "http://" in result or "https://" in result

    def test_oversized_url_rejected(self):
        """超长 URL 应被拒绝"""
        long_url = "http://example.com/" + "a" * 3000
        result = fetch_url.invoke({"url": long_url})
        assert "过长" in result

    def test_ssrf_loopback_rejected(self):
        """P0 SSRF 防护：回环地址应被拒绝"""
        # mock DNS 解析返回 127.0.0.1
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [(socket.AF_INET, 0, 0, "", ("127.0.0.1", 0))]
            result = fetch_url.invoke({"url": "http://localhost/admin"})
        assert "SSRF" in result or "非公网" in result

    def test_ssrf_private_address_rejected(self):
        """P0 SSRF 防护：私有地址（10.x / 192.168.x / 172.16-31.x）应被拒绝"""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [(socket.AF_INET, 0, 0, "", ("10.0.0.1", 0))]
            result = fetch_url.invoke({"url": "http://internal.example.com/"})
        assert "SSRF" in result or "非公网" in result

    def test_ssrf_metadata_service_rejected(self):
        """P0 SSRF 防护：云厂商元数据地址 169.254.169.254 应被拒绝（防止 IAM 凭证窃取）"""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [(socket.AF_INET, 0, 0, "", ("169.254.169.254", 0))]
            result = fetch_url.invoke({"url": "http://169.254.169.254/latest/meta-data/"})
        assert "SSRF" in result or "非公网" in result

    def test_successful_fetch_extracts_text(self):
        """成功抓取应提取正文文本"""
        html = "<html><head><title>Test</title></head><body><p>Hello World</p></body></html>"
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.headers = {"Content-Length": str(len(html))}
        mock_resp.raise_for_status = MagicMock()
        # P0 SSRF 修复：fetch_url 现在手动跟随重定向，mock 必须显式声明非重定向
        mock_resp.is_redirect = False

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("httpx.Client", return_value=mock_client):
            result = fetch_url.invoke({"url": "http://example.com"})

        # 应包含正文内容（去掉 HTML 标签后）
        assert "Hello World" in result

    def test_http_error_handled(self):
        """HTTP 错误应返回错误信息"""
        import httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.reason_phrase = "Not Found"
        mock_resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("Not Found", request=MagicMock(), response=mock_resp)
        )
        # P0 SSRF 修复：mock 必须显式声明非重定向
        mock_resp.is_redirect = False

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("httpx.Client", return_value=mock_client):
            result = fetch_url.invoke({"url": "http://example.com/notfound"})

        assert "404" in result

    def test_oversized_response_rejected(self):
        """超长响应体应被拒绝"""
        mock_resp = MagicMock()
        mock_resp.text = "x" * 100
        mock_resp.headers = {"Content-Length": "999999999"}  # 声明超大
        mock_resp.raise_for_status = MagicMock()
        # P0 SSRF 修复：mock 必须显式声明非重定向
        mock_resp.is_redirect = False

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("httpx.Client", return_value=mock_client):
            result = fetch_url.invoke({"url": "http://example.com"})

        assert "过大" in result


# ============================================================
# 3. read_file / write_file 工具测试
# ============================================================


class TestFileTools:
    """文件读写工具"""

    def test_workspace_not_configured(self):
        """未配置 workspace 时应拒绝"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = ""
            result = read_file.invoke({"file_path": "test.txt"})
            assert "未启用" in result or "未配置" in result

    def test_path_traversal_rejected(self, tmp_path):
        """路径穿越攻击应被拒绝"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            # 尝试 ../ 跳出 workspace
            result = read_file.invoke({"file_path": "../../../etc/passwd"})
            assert "越界" in result

    def test_path_traversal_prefix_confusion_rejected(self, tmp_path):
        """P0 安全加固：前缀混淆攻击应被拒绝

        旧实现 str(target).startswith(str(workspace_root)) 在 workspace=/data/workspace
        而 target=/data/workspace-evil/secret.txt 时返回 True，导致可读写任意文件。
        新实现用 relative_to 替代，从根本上消除前缀混淆。
        """
        # 在 tmp_path 同级创建一个 "tmp_path-evil" 目录模拟前缀混淆场景
        evil_dir = tmp_path.parent / (tmp_path.name + "-evil")
        evil_dir.mkdir(exist_ok=True)
        try:
            evil_file = evil_dir / "secret.txt"
            evil_file.write_text("stolen-content", encoding="utf-8")

            with patch("cayz_agent.tools.get_settings") as mock:
                mock.return_value.tools_workspace_dir = str(tmp_path)
                # 构造 ../<tmp_name>-evil/secret.txt，解析后路径以 workspace_root 为前缀但不在其子树内
                relative_evil = f"../{tmp_path.name}-evil/secret.txt"
                result = read_file.invoke({"file_path": relative_evil})
                assert "越界" in result
                # 确认未读取到 evil 文件内容
                assert "stolen-content" not in result
        finally:
            if evil_dir.exists():
                import shutil

                shutil.rmtree(evil_dir, ignore_errors=True)

    def test_write_and_read_roundtrip(self, tmp_path):
        """写入后读取应返回相同内容"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)

            # 写入
            write_result = write_file.invoke({"file_path": "test.txt", "content": "Hello 文件"})
            assert "已写入" in write_result

            # 读取
            read_result = read_file.invoke({"file_path": "test.txt"})
            assert read_result == "Hello 文件"

    def test_write_creates_parent_dirs(self, tmp_path):
        """写入应自动创建父目录"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)

            result = write_file.invoke({"file_path": "sub/dir/test.txt", "content": "nested"})
            assert "已写入" in result
            assert (tmp_path / "sub" / "dir" / "test.txt").exists()

    def test_read_nonexistent_file(self, tmp_path):
        """读取不存在的文件应返回错误"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = read_file.invoke({"file_path": "nonexistent.txt"})
            assert "不存在" in result or "错误" in result

    def test_write_oversized_content_rejected(self, tmp_path):
        """超大内容应被拒绝"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            huge_content = "x" * (1024 * 1024 + 1)
            result = write_file.invoke({"file_path": "big.txt", "content": huge_content})
            assert "过大" in result

    def test_read_oversized_file_rejected(self, tmp_path):
        """读取超大文件应被拒绝"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            # 创建一个超过 512KB 的文件
            big_file = tmp_path / "big.txt"
            big_file.write_text("x" * (512 * 1024 + 1), encoding="utf-8")
            result = read_file.invoke({"file_path": "big.txt"})
            assert "过大" in result

    def test_read_gbk_encoded_file(self, tmp_path):
        """应能读取 GBK 编码的文件"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            # 写入 GBK 编码文件
            gbk_file = tmp_path / "gbk.txt"
            gbk_file.write_text("中文内容", encoding="gbk")
            result = read_file.invoke({"file_path": "gbk.txt"})
            assert "中文内容" in result


# ============================================================
# 4. python_repl 工具测试
# ============================================================


class TestPythonRepl:
    """受控 Python 执行"""

    def test_basic_print(self):
        """基本 print 输出"""
        result = python_repl.invoke({"code": "print('Hello')"})
        assert "Hello" in result

    def test_arithmetic(self):
        """算术运算"""
        result = python_repl.invoke({"code": "print(2 + 3 * 4)"})
        assert "14" in result

    def test_multiline_code(self):
        """多行代码"""
        code = "x = 10\ny = 20\nprint(x + y)"
        result = python_repl.invoke({"code": code})
        assert "30" in result

    def test_math_module_available(self):
        """math 模块应可用"""
        result = python_repl.invoke({"code": "print(math.sqrt(16))"})
        assert "4.0" in result

    def test_statistics_module_available(self):
        """statistics 模块应可用"""
        result = python_repl.invoke({"code": "print(statistics.mean([1, 2, 3, 4, 5]))"})
        assert "3" in result

    def test_json_module_available(self):
        """json 模块应可用"""
        result = python_repl.invoke({"code": "print(json.dumps({'a': 1}))"})
        assert '{"a": 1}' in result

    def test_empty_code_rejected(self):
        """空代码应被拒绝"""
        result = python_repl.invoke({"code": ""})
        assert "为空" in result

    def test_oversized_code_rejected(self):
        """超长代码应被拒绝"""
        long_code = "x = 1\n" * 5000
        result = python_repl.invoke({"code": long_code})
        assert "过长" in result

    def test_import_blocked(self):
        """import 应被拦截"""
        result = python_repl.invoke({"code": "import os"})
        assert "危险" in result or "禁止" in result

    def test_open_blocked(self):
        """open() 应被拦截"""
        result = python_repl.invoke({"code": "open('/etc/passwd')"})
        assert "危险" in result or "禁止" in result

    def test_exec_blocked(self):
        """exec() 应被拦截"""
        result = python_repl.invoke({"code": "exec('print(1)')"})
        assert "危险" in result or "禁止" in result

    def test_subprocess_blocked(self):
        """subprocess 应被拦截"""
        result = python_repl.invoke({"code": "subprocess.run(['ls'])"})
        assert "危险" in result or "禁止" in result

    def test_no_output_returns_hint(self):
        """无输出时应返回提示"""
        result = python_repl.invoke({"code": "x = 1"})
        assert "无输出" in result

    def test_execution_error_handled(self):
        """执行错误应返回错误信息"""
        result = python_repl.invoke({"code": "print(undefined_var)"})
        assert "错误" in result or "Error" in result

    def test_infinite_while_loop_rejected(self):
        """P0 安全加固：while True 无 break 的死循环应被 AST 静态分析拒绝"""
        result = python_repl.invoke({"code": "while True:\n    pass"})
        assert "无限循环" in result
        assert "拒绝执行" in result

    def test_while_true_with_break_allowed(self):
        """P0：while True 含 break 应允许执行（带退出条件）"""
        result = python_repl.invoke({"code": "i = 0\nwhile True:\n    i += 1\n    if i >= 3:\n        break\nprint(i)"})
        assert "3" in result

    def test_while_nonzero_constant_rejected(self):
        """P0：while 1 无 break 也应被拒绝（非常量真值检测）"""
        result = python_repl.invoke({"code": "while 1:\n    x = 1"})
        assert "无限循环" in result

    def test_while_falsy_condition_allowed(self):
        """P0：while False 不应被拦截（条件为假不会死循环）"""
        result = python_repl.invoke({"code": "while False:\n    print('never')\nprint('done')"})
        assert "done" in result

    def test_subclasses_bypass_rejected(self):
        """P0 安全加固：__subclasses__ 沙箱逃逸应被拒绝（黑名单拦截）"""
        # 经典 Python 沙箱逃逸 PoC：().__class__.__base__.__subclasses__()
        code = "[x for x in ().__class__.__base__.__subclasses__() if x.__name__ == 'Popen'][0](['id'])"
        result = python_repl.invoke({"code": code})
        # 应被黑名单或 AST 拦截，不能实际执行命令
        assert "禁止" in result or "错误" in result
        # 确认未实际执行（无 uid 输出）
        assert "uid=" not in result

    def test_class_attribute_access_rejected(self):
        """P0：AST 层禁止访问以下划线开头的属性（拦截 __class__ 链）"""
        # 即使绕过黑名单（如用变量拼接），AST 检测 Attribute.attr.startswith("_") 仍拦截
        result = python_repl.invoke({"code": "x = 1\ny = x.__class__"})
        assert "禁止" in result or "错误" in result

    def test_getattr_reflection_rejected(self):
        """P0：反射函数 getattr 应被 AST 拦截（防止字符串拼接绕过黑名单）"""
        # 用不含黑名单关键词的 getattr 调用，验证 AST 层 Call 节点拦截
        # getattr(x, "__cla" + "ss__") 可绕过关键字黑名单，AST 层拦截 Call 节点
        result = python_repl.invoke({"code": "getattr((), 'foo')"})
        assert "禁止" in result or "错误" in result
        assert "getattr" in result

    def test_normal_attribute_access_allowed(self):
        """P0：非下划线属性访问应正常工作（不误伤合法用法）"""
        # math.sqrt / math.pi 等合法属性访问不应被 AST 拦截
        result = python_repl.invoke({"code": "print(math.sqrt(16))"})
        assert "4.0" in result


# ============================================================
# 5. 权限分级测试
# ============================================================


class TestToolScopePermissions:
    """工具权限分级"""

    def test_readonly_scope_has_readonly_tools(self):
        """readonly scope 应包含只读工具"""
        tools = get_tools_for_scope("readonly")
        tool_names = {t.name for t in tools}
        # 应包含 P3 新增只读工具
        assert "calculate" in tool_names
        assert "fetch_url" in tool_names
        assert "read_file" in tool_names
        assert "python_repl" in tool_names
        # 不应包含写工具
        assert "write_file" not in tool_names
        assert "knowledge_upload" not in tool_names
        assert "send_email" not in tool_names

    def test_write_scope_has_write_tools(self):
        """write scope 应包含 write_file 和 knowledge_upload"""
        tools = get_tools_for_scope("write")
        tool_names = {t.name for t in tools}
        assert "write_file" in tool_names
        assert "knowledge_upload" in tool_names
        # 不应包含 admin 工具
        assert "send_email" not in tool_names
        assert "send_wecom_notification" not in tool_names

    def test_admin_scope_has_all_tools(self):
        """admin scope 应包含全部工具"""
        tools = get_tools_for_scope("admin")
        tool_names = {t.name for t in tools}
        assert "calculate" in tool_names
        assert "write_file" in tool_names
        assert "send_email" in tool_names
        assert "send_wecom_notification" in tool_names

    def test_unknown_scope_degrades_to_readonly(self):
        """未知 scope 应降级为 readonly（最严格）"""
        tools = get_tools_for_scope("unknown_scope")
        readonly_tools = get_tools_for_scope("readonly")
        assert {t.name for t in tools} == {t.name for t in readonly_tools}
