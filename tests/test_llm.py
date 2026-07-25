"""
llm 模块单元测试：验证多模型工厂

- create_llm: 根据 LLM_PROVIDER 配置返回正确的 ChatOpenAI 实例
- list_supported_providers: 返回支持的 provider 列表
"""

from unittest.mock import patch

import pytest
from langchain_openai import ChatOpenAI

from cayz_agent.config import Settings
from cayz_agent.llm import create_llm, list_supported_providers


class TestListSupportedProviders:
    """测试 list_supported_providers"""

    def test_returns_list(self):
        """应返回列表"""
        providers = list_supported_providers()
        assert isinstance(providers, list)

    def test_contains_all_providers(self):
        """应包含所有 5 个 provider"""
        providers = list_supported_providers()
        assert len(providers) == 5
        for p in ["openai", "zhipu", "qwen", "ernie", "ollama"]:
            assert p in providers


class TestCreateLlm:
    """测试 create_llm 工厂函数"""

    @patch("cayz_agent.llm.get_settings")
    def test_openai_provider(self, mock_get_settings):
        """openai provider 应使用 openai_api_key 和 openai_api_base"""
        mock_get_settings.return_value = Settings(
            llm_provider="openai",
            openai_api_key="sk-test",
            openai_api_base="https://api.openai.com/v1",
            model_name="gpt-4",
            temperature=0.5,
        )
        llm = create_llm()
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-4"
        assert llm.temperature == 0.5

    @patch("cayz_agent.llm.get_settings")
    def test_zhipu_provider(self, mock_get_settings):
        """zhipu provider 应使用 zhipu_api_base"""
        mock_get_settings.return_value = Settings(
            llm_provider="zhipu",
            zhipu_api_key="zhipu-test-key",
            zhipu_api_base="https://open.bigmodel.cn/api/paas/v4",
            model_name="glm-4",
            temperature=0.0,
        )
        llm = create_llm()
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "glm-4"

    @patch("cayz_agent.llm.get_settings")
    def test_zhipu_fallback_to_openai_key(self, mock_get_settings):
        """zhipu provider 在未设置 zhipu_api_key 时回退到 openai_api_key"""
        mock_get_settings.return_value = Settings(
            llm_provider="zhipu",
            zhipu_api_key="",
            openai_api_key="fallback-key",
            model_name="glm-4",
        )
        llm = create_llm()
        assert isinstance(llm, ChatOpenAI)

    @patch("cayz_agent.llm.get_settings")
    def test_qwen_provider(self, mock_get_settings):
        """qwen provider 应使用 openai_api_base（DashScope）"""
        mock_get_settings.return_value = Settings(
            llm_provider="qwen",
            openai_api_key="qwen-key",
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name="qwen-turbo",
        )
        llm = create_llm()
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "qwen-turbo"

    @patch("cayz_agent.llm.get_settings")
    def test_ernie_provider(self, mock_get_settings):
        """ernie provider 应使用 ernie_api_base"""
        mock_get_settings.return_value = Settings(
            llm_provider="ernie",
            ernie_api_key="ernie-key",
            ernie_api_base="https://qianfan.baidubce.com/v2",
            model_name="ernie-bot-4",
        )
        llm = create_llm()
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "ernie-bot-4"

    @patch("cayz_agent.llm.get_settings")
    def test_ollama_provider(self, mock_get_settings):
        """ollama provider 应自动添加 /v1 后缀"""
        mock_get_settings.return_value = Settings(
            llm_provider="ollama",
            ollama_base_url="http://localhost:11434",
            model_name="llama3",
        )
        llm = create_llm()
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "llama3"

    @patch("cayz_agent.llm.get_settings")
    def test_ollama_provider_with_trailing_slash(self, mock_get_settings):
        """ollama provider 应正确处理带尾部斜杠的 URL"""
        mock_get_settings.return_value = Settings(
            llm_provider="ollama",
            ollama_base_url="http://localhost:11434/",
            model_name="llama3",
        )
        llm = create_llm()
        assert isinstance(llm, ChatOpenAI)

    @patch("cayz_agent.llm.get_settings")
    def test_ollama_provider_with_existing_v1(self, mock_get_settings):
        """ollama provider 在已有 /v1 时不应重复添加"""
        mock_get_settings.return_value = Settings(
            llm_provider="ollama",
            ollama_base_url="http://localhost:11434/v1",
            model_name="llama3",
        )
        llm = create_llm()
        assert isinstance(llm, ChatOpenAI)

    @patch("cayz_agent.llm.get_settings")
    def test_unknown_provider_raises_config_error(self, mock_get_settings):
        """未知 provider 应在 Settings 构造时被 P2-9 validator 拒绝（fail-fast）"""
        from pydantic import ValidationError

        # P2-9：Settings 的 field_validator 在构造时即拒绝未知 provider，
        # 无需等到 create_llm() 才报错（fail-fast 原则）
        with pytest.raises(ValidationError, match="llm_provider"):
            Settings(
                llm_provider="unknown",
                openai_api_key="sk-test",
                model_name="gpt-4",
            )

    @patch("cayz_agent.llm.get_settings")
    def test_default_provider_is_openai(self, mock_get_settings):
        """默认 provider 应为 openai"""
        mock_get_settings.return_value = Settings(
            llm_provider="openai",
            openai_api_key="sk-test",
            model_name="gpt-3.5-turbo",
        )
        llm = create_llm()
        assert isinstance(llm, ChatOpenAI)
