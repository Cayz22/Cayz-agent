"""
P3 第二梯队工具测试：parse_pdf / parse_excel / parse_csv / generate_qrcode /
get_weather / get_exchange_rate / translate

覆盖：
1. 文件解析类：PDF/Excel/CSV 解析、路径白名单、错误处理
2. 二维码生成：基本生成、数据校验、路径白名单
3. 外部信息查询类：API 未配置处理、参数校验、HTTP mock
4. 权限分级：scope 与工具分配正确
"""
import os
from unittest.mock import patch, MagicMock

import pytest

from cayz_agent.tools import (
    parse_pdf,
    parse_excel,
    parse_csv,
    generate_qrcode,
    get_weather,
    get_exchange_rate,
    translate,
    get_tools_for_scope,
)


# ============================================================
# 1. parse_pdf 工具测试
# ============================================================

class TestParsePdf:
    """PDF 文本提取"""

    def test_workspace_not_configured(self):
        """未配置 workspace 时应拒绝"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = ""
            result = parse_pdf.invoke({"file_path": "test.pdf"})
            assert "未启用" in result or "未配置" in result

    def test_nonexistent_file(self, tmp_path):
        """不存在的文件应返回错误"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = parse_pdf.invoke({"file_path": "nonexistent.pdf"})
            assert "不存在" in result or "错误" in result

    def test_oversized_file_rejected(self, tmp_path):
        """超大文件应被拒绝"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            # 创建一个超过 10MB 的假 PDF
            big_file = tmp_path / "big.pdf"
            big_file.write_bytes(b"0" * (10 * 1024 * 1024 + 1))
            result = parse_pdf.invoke({"file_path": "big.pdf"})
            assert "过大" in result

    def test_valid_pdf_extracts_text(self, tmp_path):
        """有效 PDF 应提取文本"""
        # 使用 pypdf 创建一个简单的 PDF
        try:
            from pypdf import PdfWriter
        except ImportError:
            pytest.skip("pypdf 不可用")

        pdf_file = tmp_path / "test.pdf"
        writer = PdfWriter()
        # 添加一个空白页（pypdf 无法直接添加文本，需 reportlab）
        # 这里仅测试工具能调用且不崩溃
        writer.add_blank_page(width=200, height=200)
        with open(pdf_file, "wb") as f:
            writer.write(f)

        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = parse_pdf.invoke({"file_path": "test.pdf"})
            # 空白页应返回"未提取到文本"提示
            assert "未提取到文本" in result or "扫描件" in result

    def test_path_traversal_rejected(self, tmp_path):
        """路径穿越应被拒绝"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = parse_pdf.invoke({"file_path": "../../../etc/passwd"})
            assert "越界" in result


# ============================================================
# 2. parse_excel 工具测试
# ============================================================

class TestParseExcel:
    """Excel 解析"""

    def test_workspace_not_configured(self):
        """未配置 workspace 时应拒绝"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = ""
            result = parse_excel.invoke({"file_path": "test.xlsx"})
            assert "未启用" in result or "未配置" in result

    def test_nonexistent_file(self, tmp_path):
        """不存在的文件应返回错误"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = parse_excel.invoke({"file_path": "nonexistent.xlsx"})
            assert "不存在" in result or "错误" in result

    def test_valid_excel_extracts_data(self, tmp_path):
        """有效 Excel 应提取为 Markdown 表格"""
        from openpyxl import Workbook

        xlsx_file = tmp_path / "test.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(["姓名", "年龄", "城市"])
        ws.append(["张三", 25, "北京"])
        ws.append(["李四", 30, "上海"])
        wb.save(xlsx_file)
        wb.close()

        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = parse_excel.invoke({"file_path": "test.xlsx"})
            # 应包含表头与数据
            assert "姓名" in result
            assert "张三" in result
            assert "北京" in result
            # 应为 Markdown 表格格式
            assert "|" in result

    def test_specified_sheet_not_found(self, tmp_path):
        """指定不存在的工作表应返回错误"""
        from openpyxl import Workbook

        xlsx_file = tmp_path / "test.xlsx"
        wb = Workbook()
        wb.save(xlsx_file)
        wb.close()

        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = parse_excel.invoke({"file_path": "test.xlsx", "sheet_name": "NotExist"})
            assert "不存在" in result

    def test_path_traversal_rejected(self, tmp_path):
        """路径穿越应被拒绝"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = parse_excel.invoke({"file_path": "../../../etc/passwd"})
            assert "越界" in result


# ============================================================
# 3. parse_csv 工具测试
# ============================================================

class TestParseCsv:
    """CSV 解析"""

    def test_workspace_not_configured(self):
        """未配置 workspace 时应拒绝"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = ""
            result = parse_csv.invoke({"file_path": "test.csv"})
            assert "未启用" in result or "未配置" in result

    def test_valid_csv_extracts_data(self, tmp_path):
        """有效 CSV 应提取为 Markdown 表格"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("姓名,年龄,城市\n张三,25,北京\n李四,30,上海\n", encoding="utf-8")

        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = parse_csv.invoke({"file_path": "test.csv"})
            assert "姓名" in result
            assert "张三" in result
            assert "北京" in result
            assert "|" in result

    def test_csv_with_bom(self, tmp_path):
        """UTF-8 BOM 编码的 CSV 应能正确解析"""
        csv_file = tmp_path / "bom.csv"
        # UTF-8 BOM + 中文内容
        csv_file.write_bytes(b"\xef\xbb\xbf" + "姓名,年龄\n张三,25\n".encode("utf-8"))

        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = parse_csv.invoke({"file_path": "bom.csv"})
            assert "姓名" in result
            assert "张三" in result

    def test_csv_with_tsv_delimiter(self, tmp_path):
        """TSV（制表符分隔）应能解析"""
        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text("姓名\t年龄\n张三\t25\n", encoding="utf-8")

        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = parse_csv.invoke({"file_path": "test.tsv", "delimiter": "\\t"})
            assert "姓名" in result
            assert "张三" in result

    def test_empty_csv(self, tmp_path):
        """空 CSV 应返回提示"""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("", encoding="utf-8")

        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = parse_csv.invoke({"file_path": "empty.csv"})
            assert "为空" in result

    def test_csv_only_empty_lines(self, tmp_path):
        """仅空行的 CSV 应返回提示"""
        csv_file = tmp_path / "blanks.csv"
        csv_file.write_text("\n\n\n", encoding="utf-8")

        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = parse_csv.invoke({"file_path": "blanks.csv"})
            assert "无有效数据" in result


# ============================================================
# 4. generate_qrcode 工具测试
# ============================================================

class TestGenerateQrcode:
    """二维码生成"""

    def test_empty_data_rejected(self, tmp_path):
        """空数据应被拒绝"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = generate_qrcode.invoke({"data": ""})
            assert "为空" in result

    def test_oversized_data_rejected(self, tmp_path):
        """超长数据应被拒绝"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = generate_qrcode.invoke({"data": "x" * 2001})
            assert "过长" in result

    def test_valid_qrcode_generated(self, tmp_path):
        """有效数据应生成二维码图片"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = generate_qrcode.invoke({
                "data": "https://example.com",
                "file_path": "test_qr.png"
            })
            assert "已生成" in result
            assert (tmp_path / "test_qr.png").exists()
            # 文件大小应 > 0
            assert (tmp_path / "test_qr.png").stat().st_size > 0

    def test_qrcode_creates_parent_dirs(self, tmp_path):
        """应自动创建父目录"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = generate_qrcode.invoke({
                "data": "test",
                "file_path": "sub/dir/qr.png"
            })
            assert "已生成" in result
            assert (tmp_path / "sub" / "dir" / "qr.png").exists()

    def test_path_traversal_rejected(self, tmp_path):
        """路径穿越应被拒绝"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.tools_workspace_dir = str(tmp_path)
            result = generate_qrcode.invoke({
                "data": "test",
                "file_path": "../../../etc/qr.png"
            })
            assert "越界" in result


# ============================================================
# 5. get_weather 工具测试
# ============================================================

class TestGetWeather:
    """天气查询"""

    def test_empty_location_rejected(self):
        """空城市应被拒绝"""
        result = get_weather.invoke({"location": ""})
        assert "为空" in result

    def test_api_not_configured(self):
        """未配置 API Key 应返回提示"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.weather_api_key = ""
            result = get_weather.invoke({"location": "北京"})
            assert "未配置" in result

    def test_successful_weather_query(self):
        """成功查询应返回天气信息"""
        # mock 城市查询 + 天气查询两个 HTTP 调用
        geo_response = MagicMock()
        geo_response.json.return_value = {
            "code": "200",
            "location": [{"id": "101010100", "name": "北京"}],
        }
        geo_response.raise_for_status = MagicMock()

        weather_response = MagicMock()
        weather_response.json.return_value = {
            "code": "200",
            "now": {
                "temp": "25",
                "feelsLike": "26",
                "text": "晴",
                "windDir": "东南",
                "windScale": "2",
                "humidity": "45",
                "obsTime": "2026-07-25T12:00+08:00",
            },
        }
        weather_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        # 第一次调用返回 geo，第二次返回 weather
        mock_client.get = MagicMock(side_effect=[geo_response, weather_response])

        with patch("httpx.Client", return_value=mock_client), \
             patch("cayz_agent.tools.get_settings") as mock_settings:
            mock_settings.return_value.weather_api_key = "test_key"
            mock_settings.return_value.weather_api_base = "https://devapi.qweather.com/v7"
            result = get_weather.invoke({"location": "北京"})

        assert "北京" in result
        assert "25" in result
        assert "晴" in result

    def test_city_not_found(self):
        """城市不存在应返回错误"""
        geo_response = MagicMock()
        geo_response.json.return_value = {"code": "404", "location": []}
        geo_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=geo_response)

        with patch("httpx.Client", return_value=mock_client), \
             patch("cayz_agent.tools.get_settings") as mock_settings:
            mock_settings.return_value.weather_api_key = "test_key"
            result = get_weather.invoke({"location": "不存在的城市"})

        assert "未找到" in result


# ============================================================
# 6. get_exchange_rate 工具测试
# ============================================================

class TestGetExchangeRate:
    """汇率查询"""

    def test_invalid_currency_code(self):
        """无效货币代码应被拒绝"""
        result = get_exchange_rate.invoke({"base": "US", "target": "CNY"})
        assert "3 字符" in result

    def test_api_not_configured(self):
        """未配置 API Key 应返回提示"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.exchange_rate_api_key = ""
            result = get_exchange_rate.invoke({"base": "USD", "target": "CNY"})
            assert "未配置" in result

    def test_successful_rate_query(self):
        """成功查询应返回汇率"""
        response = MagicMock()
        response.json.return_value = {
            "success": True,
            "date": "2026-07-25",
            "rates": {"USD": 1.08, "CNY": 7.85},
        }
        response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=response)

        with patch("httpx.Client", return_value=mock_client), \
             patch("cayz_agent.tools.get_settings") as mock_settings:
            mock_settings.return_value.exchange_rate_api_key = "test_key"
            mock_settings.return_value.exchange_rate_api_base = "https://data.fixer.io/api"
            result = get_exchange_rate.invoke({"base": "USD", "target": "CNY"})

        # 1 USD = 7.85 / 1.08 ≈ 7.2685 CNY
        assert "USD" in result
        assert "CNY" in result
        assert "7." in result  # 约 7.x

    def test_api_error_handled(self):
        """API 返回错误应处理"""
        response = MagicMock()
        response.json.return_value = {
            "success": False,
            "error": {"code": 101, "info": "无效 access_key"},
        }
        response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=response)

        with patch("httpx.Client", return_value=mock_client), \
             patch("cayz_agent.tools.get_settings") as mock_settings:
            mock_settings.return_value.exchange_rate_api_key = "test_key"
            result = get_exchange_rate.invoke({"base": "USD", "target": "CNY"})

        assert "错误" in result


# ============================================================
# 7. translate 工具测试
# ============================================================

class TestTranslate:
    """翻译"""

    def test_empty_text_rejected(self):
        """空文本应被拒绝"""
        result = translate.invoke({"text": ""})
        assert "为空" in result

    def test_oversized_text_rejected(self):
        """超长文本应被拒绝"""
        result = translate.invoke({"text": "x" * 6001})
        assert "过长" in result

    def test_api_not_configured(self):
        """未配置 API 应返回提示"""
        with patch("cayz_agent.tools.get_settings") as mock:
            mock.return_value.baidu_translate_app_id = ""
            mock.return_value.baidu_translate_api_key = ""
            result = translate.invoke({"text": "hello"})
            assert "未配置" in result

    def test_successful_translation(self):
        """成功翻译应返回结果"""
        response = MagicMock()
        response.json.return_value = {
            "from": "en",
            "to": "zh",
            "trans_result": [
                {"src": "hello", "dst": "你好"}
            ],
        }
        response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=response)

        with patch("httpx.Client", return_value=mock_client), \
             patch("cayz_agent.tools.get_settings") as mock_settings:
            mock_settings.return_value.baidu_translate_app_id = "test_id"
            mock_settings.return_value.baidu_translate_api_key = "test_key"
            result = translate.invoke({"text": "hello", "to_lang": "zh"})

        assert "你好" in result
        assert "源语言" in result

    def test_api_error_handled(self):
        """API 错误应处理"""
        response = MagicMock()
        response.json.return_value = {
            "error_code": "54001",
            "error_msg": "签名错误",
        }
        response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=response)

        with patch("httpx.Client", return_value=mock_client), \
             patch("cayz_agent.tools.get_settings") as mock_settings:
            mock_settings.return_value.baidu_translate_app_id = "test_id"
            mock_settings.return_value.baidu_translate_api_key = "test_key"
            result = translate.invoke({"text": "hello"})

        assert "错误" in result
        assert "签名" in result


# ============================================================
# 8. 权限分级测试
# ============================================================

class TestToolScopePermissionsTier2:
    """第二梯队工具权限分级"""

    def test_readonly_has_parse_and_query_tools(self):
        """readonly scope 应包含解析类与查询类工具"""
        tools = get_tools_for_scope("readonly")
        tool_names = {t.name for t in tools}
        # 第二梯队只读工具
        assert "parse_pdf" in tool_names
        assert "parse_excel" in tool_names
        assert "parse_csv" in tool_names
        assert "get_weather" in tool_names
        assert "get_exchange_rate" in tool_names
        assert "translate" in tool_names
        # 不应包含写工具
        assert "generate_qrcode" not in tool_names
        assert "write_file" not in tool_names

    def test_write_scope_has_qrcode(self):
        """write scope 应包含 generate_qrcode"""
        tools = get_tools_for_scope("write")
        tool_names = {t.name for t in tools}
        assert "generate_qrcode" in tool_names
        assert "write_file" in tool_names
        # 不应包含 admin 工具
        assert "send_email" not in tool_names

    def test_admin_scope_has_all_tier2_tools(self):
        """admin scope 应包含全部第二梯队工具"""
        tools = get_tools_for_scope("admin")
        tool_names = {t.name for t in tools}
        tier2_tools = [
            "parse_pdf", "parse_excel", "parse_csv",
            "generate_qrcode",
            "get_weather", "get_exchange_rate", "translate",
        ]
        for tool_name in tier2_tools:
            assert tool_name in tool_names, f"{tool_name} 不在 admin scope"

    def test_unknown_scope_degrades_to_readonly(self):
        """未知 scope 应降级为 readonly"""
        tools = get_tools_for_scope("unknown")
        readonly_tools = get_tools_for_scope("readonly")
        assert {t.name for t in tools} == {t.name for t in readonly_tools}
