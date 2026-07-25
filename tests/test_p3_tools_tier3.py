"""
P3 第三梯队工具测试：hash_encode / text_diff / regex_test / unit_convert

覆盖：
1. hash_encode：哈希计算 + 编码转换 + 错误处理
2. text_diff：差异对比 + 相同文本 + 边界
3. regex_test：匹配 + 捕获组 + flags + 语法错误
4. unit_convert：各类别换算 + 温度公式 + 错误单位
5. 权限分级：scope 与工具分配
"""

import base64
import hashlib

import pytest

from cayz_agent.tools import (
    get_tools_for_scope,
    hash_encode,
    regex_test,
    text_diff,
    unit_convert,
)

# ============================================================
# 1. hash_encode 工具测试
# ============================================================


class TestHashEncode:
    """哈希计算与编码转换"""

    def test_md5_hash(self):
        """MD5 哈希应与 hashlib 结果一致"""
        text = "hello"
        expected = hashlib.md5(text.encode("utf-8")).hexdigest()
        result = hash_encode.invoke({"text": text, "algorithm": "md5"})
        assert result == expected

    def test_sha256_hash(self):
        """SHA256 哈希应正确"""
        text = "hello"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        result = hash_encode.invoke({"text": text, "algorithm": "sha256"})
        assert result == expected

    def test_sha512_hash(self):
        """SHA512 哈希应正确"""
        text = "test"
        expected = hashlib.sha512(text.encode("utf-8")).hexdigest()
        result = hash_encode.invoke({"text": text, "algorithm": "sha512"})
        assert result == expected

    def test_default_algorithm_is_md5(self):
        """默认算法应为 md5"""
        text = "test"
        expected = hashlib.md5(text.encode("utf-8")).hexdigest()
        result = hash_encode.invoke({"text": text})
        assert result == expected

    def test_base64_encode(self):
        """Base64 编码应正确"""
        text = "hello"
        expected = base64.b64encode(text.encode("utf-8")).decode("ascii")
        result = hash_encode.invoke({"text": text, "algorithm": "base64"})
        assert result == expected

    def test_base64_decode(self):
        """Base64 解码应正确"""
        encoded = base64.b64encode(b"hello").decode("ascii")
        result = hash_encode.invoke({"text": encoded, "algorithm": "base64_decode"})
        assert result == "hello"

    def test_url_encode(self):
        """URL 编码应正确处理特殊字符"""
        result = hash_encode.invoke({"text": "hello world&foo=bar", "algorithm": "url_encode"})
        assert "hello%20world" in result
        assert "%26" in result
        assert "%3D" in result

    def test_url_decode(self):
        """URL 解码应正确"""
        result = hash_encode.invoke({"text": "hello%20world%26foo%3Dbar", "algorithm": "url_decode"})
        assert result == "hello world&foo=bar"

    def test_hex_encode(self):
        """Hex 编码应正确"""
        result = hash_encode.invoke({"text": "AB", "algorithm": "hex_encode"})
        assert result == "4142"  # 'A'=0x41, 'B'=0x42

    def test_hex_decode(self):
        """Hex 解码应正确"""
        result = hash_encode.invoke({"text": "4142", "algorithm": "hex_decode"})
        assert result == "AB"

    def test_empty_text_rejected(self):
        """空文本应被拒绝"""
        result = hash_encode.invoke({"text": "", "algorithm": "md5"})
        assert "为空" in result

    def test_oversized_text_rejected(self):
        """超长文本应被拒绝"""
        result = hash_encode.invoke({"text": "x" * (1024 * 1024 + 1), "algorithm": "md5"})
        assert "过长" in result

    def test_unsupported_algorithm(self):
        """不支持的算法应返回错误"""
        result = hash_encode.invoke({"text": "test", "algorithm": "unknown"})
        assert "不支持" in result
        assert "md5" in result  # 错误信息中应列出支持的算法

    def test_invalid_base64_decode(self):
        """无效 Base64 解码应返回错误"""
        result = hash_encode.invoke({"text": "!!!invalid base64!!!", "algorithm": "base64_decode"})
        assert "错误" in result

    def test_chinese_text_md5(self):
        """中文文本哈希应正确"""
        text = "你好世界"
        expected = hashlib.md5(text.encode("utf-8")).hexdigest()
        result = hash_encode.invoke({"text": text, "algorithm": "md5"})
        assert result == expected


# ============================================================
# 2. text_diff 工具测试
# ============================================================


class TestTextDiff:
    """文本差异对比"""

    def test_identical_texts_no_diff(self):
        """相同文本应返回无差异提示"""
        result = text_diff.invoke({"text1": "hello", "text2": "hello"})
        assert "完全相同" in result or "无差异" in result

    def test_added_line_shown(self):
        """新增行应在 diff 中显示"""
        text1 = "line1\nline2"
        text2 = "line1\nline2\nline3"
        result = text_diff.invoke({"text1": text1, "text2": text2})
        assert "line3" in result
        assert "+" in result  # unified diff 新增行前缀

    def test_removed_line_shown(self):
        """删除行应在 diff 中显示"""
        text1 = "line1\nline2\nline3"
        text2 = "line1\nline3"
        result = text_diff.invoke({"text1": text1, "text2": text2})
        assert "line2" in result
        assert "-" in result  # unified diff 删除行前缀

    def test_modified_line_shown(self):
        """修改行应同时显示删除和新增"""
        text1 = "hello world"
        text2 = "hello python"
        result = text_diff.invoke({"text1": text1, "text2": text2})
        assert "world" in result
        assert "python" in result

    def test_empty_text1(self):
        """text1 为空时所有行均为新增"""
        result = text_diff.invoke({"text1": "", "text2": "new content"})
        assert "new content" in result

    def test_empty_text2(self):
        """text2 为空时所有行均为删除"""
        result = text_diff.invoke({"text1": "old content", "text2": ""})
        assert "old content" in result

    def test_oversized_text_rejected(self):
        """超长文本应被拒绝"""
        big_text = "x" * (100 * 1024 + 1)
        result = text_diff.invoke({"text1": big_text, "text2": "test"})
        assert "过长" in result

    def test_context_lines_configurable(self):
        """上下文行数应可配置"""
        # 多行文本，修改中间一行
        text1 = "\n".join(f"line{i}" for i in range(10))
        text2 = "\n".join(f"line{i}" if i != 5 else "MODIFIED" for i in range(10))

        # 上下文 1 行
        result_1 = text_diff.invoke({"text1": text1, "text2": text2, "lines_per_context": 1})
        # 上下文 5 行
        result_5 = text_diff.invoke({"text1": text1, "text2": text2, "lines_per_context": 5})

        # 上下文 5 行的结果应更长（显示更多未变更行）
        assert len(result_5) > len(result_1)

    def test_multiline_diff_format(self):
        """多行 diff 应为 unified diff 格式"""
        text1 = "a\nb\nc"
        text2 = "a\nB\nc"
        result = text_diff.invoke({"text1": text1, "text2": text2})
        # unified diff 应包含 +++/--- 头部
        assert "---" in result or "原始" in result
        assert "+++" in result or "修改后" in result


# ============================================================
# 3. regex_test 工具测试
# ============================================================


class TestRegexTest:
    """正则表达式测试"""

    def test_basic_match(self):
        """基本匹配应返回结果"""
        result = regex_test.invoke({"pattern": r"\d+", "text": "abc123def"})
        assert "匹配数: 1" in result
        assert "123" in result

    def test_multiple_matches(self):
        """多个匹配应全部显示"""
        result = regex_test.invoke({"pattern": r"\d+", "text": "a1b2c3"})
        assert "匹配数: 3" in result

    def test_no_match(self):
        """无匹配应返回提示"""
        result = regex_test.invoke({"pattern": r"\d+", "text": "no digits here"})
        assert "无匹配" in result

    def test_capture_groups(self):
        """捕获组应显示"""
        result = regex_test.invoke({"pattern": r"(\w+)@(\w+)\.(\w+)", "text": "user@example.com"})
        assert "捕获组" in result
        assert "user" in result
        assert "example" in result
        assert "com" in result

    def test_named_groups(self):
        """命名捕获组应显示"""
        result = regex_test.invoke({"pattern": r"(?P<year>\d{4})-(?P<month>\d{2})", "text": "2026-07"})
        assert "命名组" in result or "year" in result
        assert "2026" in result
        assert "07" in result

    def test_ignore_case_flag(self):
        """i flag 应忽略大小写"""
        # 不带 flag：无匹配
        result_no_flag = regex_test.invoke({"pattern": "hello", "text": "HELLO"})
        assert "无匹配" in result_no_flag
        # 带 i flag：应匹配
        result_with_flag = regex_test.invoke({"pattern": "hello", "text": "HELLO", "flags": "i"})
        assert "匹配数: 1" in result_with_flag

    def test_multiline_flag(self):
        """m flag 应支持多行匹配"""
        text = "line1\nline2\nline3"
        # 不带 m flag：^ 只匹配字符串开头
        _result_no_m = regex_test.invoke({"pattern": "^line", "text": text})
        # 带 m flag：^ 匹配每行开头
        result_with_m = regex_test.invoke({"pattern": "^line", "text": text, "flags": "m"})
        # 多行模式应匹配更多
        assert "匹配数: 3" in result_with_m

    def test_invalid_pattern_returns_error(self):
        """非法正则应返回语法错误"""
        result = regex_test.invoke({"pattern": "[invalid", "text": "test"})
        assert "语法错误" in result or "错误" in result

    def test_empty_pattern_rejected(self):
        """空正则应被拒绝"""
        result = regex_test.invoke({"pattern": "", "text": "test"})
        assert "为空" in result

    def test_empty_text_rejected(self):
        """空文本应被拒绝"""
        result = regex_test.invoke({"pattern": "\\d+", "text": ""})
        assert "为空" in result

    def test_match_position_reported(self):
        """匹配位置应被报告"""
        result = regex_test.invoke({"pattern": "abc", "text": "xxabcxx"})
        assert "位置" in result
        assert "2-5" in result  # abc 在位置 2-5

    def test_oversized_pattern_rejected(self):
        """超长正则应被拒绝"""
        result = regex_test.invoke({"pattern": "x" * 1001, "text": "test"})
        assert "过长" in result


# ============================================================
# 4. unit_convert 工具测试
# ============================================================


class TestUnitConvert:
    """单位换算"""

    def test_length_km_to_m(self):
        """长度：km → m"""
        result = unit_convert.invoke({"value": 1, "from_unit": "km", "to_unit": "m", "category": "length"})
        assert "1000" in result

    def test_length_m_to_km(self):
        """长度：m → km"""
        result = unit_convert.invoke({"value": 1000, "from_unit": "m", "to_unit": "km", "category": "length"})
        assert "1" in result

    def test_length_mi_to_km(self):
        """长度：英里 → km"""
        result = unit_convert.invoke({"value": 1, "from_unit": "mi", "to_unit": "km", "category": "length"})
        # 1 mile ≈ 1.609344 km
        assert "1.609" in result or "1.61" in result

    def test_weight_kg_to_lb(self):
        """重量：kg → lb"""
        result = unit_convert.invoke({"value": 1, "from_unit": "kg", "to_unit": "lb", "category": "weight"})
        # 1 kg ≈ 2.20462 lb
        assert "2.20" in result or "2.2046" in result

    def test_weight_jin_to_kg(self):
        """重量：斤 → kg"""
        result = unit_convert.invoke({"value": 2, "from_unit": "斤", "to_unit": "kg", "category": "weight"})
        # 2 斤 = 1 kg
        assert "1" in result

    def test_temperature_c_to_f(self):
        """温度：摄氏 → 华氏"""
        result = unit_convert.invoke({"value": 0, "from_unit": "c", "to_unit": "f", "category": "temperature"})
        # 0°C = 32°F
        assert "32" in result

    def test_temperature_f_to_c(self):
        """温度：华氏 → 摄氏"""
        result = unit_convert.invoke({"value": 212, "from_unit": "f", "to_unit": "c", "category": "temperature"})
        # 212°F = 100°C
        assert "100" in result

    def test_temperature_c_to_k(self):
        """温度：摄氏 → 开尔文"""
        result = unit_convert.invoke({"value": 0, "from_unit": "c", "to_unit": "k", "category": "temperature"})
        # 0°C = 273.15 K
        assert "273.15" in result

    def test_time_h_to_s(self):
        """时间：小时 → 秒"""
        result = unit_convert.invoke({"value": 1, "from_unit": "h", "to_unit": "s", "category": "time"})
        assert "3600" in result

    def test_time_day_to_h(self):
        """时间：天 → 小时"""
        result = unit_convert.invoke({"value": 1, "from_unit": "day", "to_unit": "h", "category": "time"})
        assert "24" in result

    def test_data_kb_to_b(self):
        """数据量：KB → B"""
        result = unit_convert.invoke({"value": 1, "from_unit": "kb", "to_unit": "b", "category": "data"})
        assert "1024" in result

    def test_data_mb_to_gb(self):
        """数据量：MB → GB"""
        result = unit_convert.invoke({"value": 1024, "from_unit": "mb", "to_unit": "gb", "category": "data"})
        assert "1" in result

    def test_invalid_value_rejected(self):
        """无效数值应被拒绝（Pydantic 在 invoke 时拦截）"""
        # Pydantic 的 float 类型校验会拦截非法字符串，抛 ValidationError
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            unit_convert.invoke({"value": "not_a_number", "from_unit": "m", "to_unit": "km", "category": "length"})

    def test_unsupported_category(self):
        """不支持的类别应返回错误"""
        result = unit_convert.invoke({"value": 1, "from_unit": "x", "to_unit": "y", "category": "unknown"})
        assert "不支持" in result

    def test_unsupported_unit(self):
        """不支持的单位应返回错误并列出可用单位"""
        result = unit_convert.invoke({"value": 1, "from_unit": "xxx", "to_unit": "m", "category": "length"})
        assert "不支持" in result
        assert "km" in result  # 错误信息应列出可用单位

    def test_empty_unit_rejected(self):
        """空单位应被拒绝"""
        result = unit_convert.invoke({"value": 1, "from_unit": "", "to_unit": "m", "category": "length"})
        assert "为空" in result

    def test_default_category_is_length(self):
        """默认类别应为 length"""
        result = unit_convert.invoke({"value": 1, "from_unit": "km", "to_unit": "m"})
        assert "1000" in result


# ============================================================
# 5. 权限分级测试
# ============================================================


class TestToolScopePermissionsTier3:
    """第三梯队工具权限分级"""

    def test_readonly_has_tier3_tools(self):
        """readonly scope 应包含第三梯队工具"""
        tools = get_tools_for_scope("readonly")
        tool_names = {t.name for t in tools}
        assert "hash_encode" in tool_names
        assert "text_diff" in tool_names
        assert "regex_test" in tool_names
        assert "unit_convert" in tool_names

    def test_write_scope_inherits_tier3(self):
        """write scope 应包含第三梯队工具（继承自 readonly）"""
        tools = get_tools_for_scope("write")
        tool_names = {t.name for t in tools}
        assert "hash_encode" in tool_names
        assert "text_diff" in tool_names
        assert "regex_test" in tool_names
        assert "unit_convert" in tool_names

    def test_admin_scope_has_all_tools(self):
        """admin scope 应包含全部工具（含第三梯队）"""
        tools = get_tools_for_scope("admin")
        tool_names = {t.name for t in tools}
        tier3_tools = ["hash_encode", "text_diff", "regex_test", "unit_convert"]
        for tool_name in tier3_tools:
            assert tool_name in tool_names, f"{tool_name} 不在 admin scope"

    def test_total_tool_count(self):
        """验证工具总数：9 原始 + 5 一梯队 + 7 二梯队 + 4 三梯队 = 25"""
        admin_tools = get_tools_for_scope("admin")
        assert len(admin_tools) == 25
