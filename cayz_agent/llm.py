"""
多模型工厂：统一对接 OpenAI / 智谱 GLM / 通义千问 / 百度文心 / Ollama

所有 provider 都通过 langchain_openai.ChatOpenAI 接入（OpenAI 兼容协议），
少数原生不兼容的 provider 通过对应的 langchain 社区包接入。
"""

import logging

from langchain_openai import ChatOpenAI

from .config import get_settings
from .exceptions import ConfigError, LLMError

logger = logging.getLogger(__name__)


def create_llm() -> ChatOpenAI:
    """
    根据配置创建对应的 LLM 实例。

    支持的 provider：
    - openai: OpenAI 官方或任意兼容服务（含阿里 DashScope）
    - zhipu: 智谱 GLM（OpenAI 兼容协议）
    - qwen: 通义千问（通过 DashScope OpenAI 兼容接口，等价于 openai provider）
    - ernie: 百度文心（通过千帆 v2 OpenAI 兼容接口）
    - ollama: 本地 Ollama 推理服务

    Returns:
        ChatOpenAI 实例，已配置好 api_key / base_url / model

    Raises:
        ConfigError: provider 不支持或缺少必要配置
        LLMError: LLM 实例创建失败
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()

    try:
        if provider == "zhipu":
            return _create_zhipu_llm(settings)
        elif provider == "qwen":
            return _create_qwen_llm(settings)
        elif provider == "ernie":
            return _create_ernie_llm(settings)
        elif provider == "ollama":
            return _create_ollama_llm(settings)
        elif provider == "openai":
            return _create_openai_llm(settings)
        else:
            raise ConfigError(f"不支持的 LLM provider: {provider}，可选: openai/zhipu/qwen/ernie/ollama")
    except (ConfigError, LLMError):
        raise
    except Exception as e:
        raise LLMError(f"创建 LLM 实例失败 (provider={provider})", cause=e) from e


def _create_openai_llm(settings) -> ChatOpenAI:
    """OpenAI 官方或兼容服务（含阿里 DashScope）"""
    logger.info("使用 OpenAI 兼容 provider, model=%s, base=%s", settings.model_name, settings.openai_api_base)
    return ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        max_retries=3,
        request_timeout=settings.llm_request_timeout,
    )


def _create_zhipu_llm(settings) -> ChatOpenAI:
    """
    智谱 GLM：OpenAI 兼容协议
    base_url=https://open.bigmodel.cn/api/paas/v4
    """
    api_key = settings.zhipu_api_key or settings.openai_api_key
    logger.info("使用智谱 GLM provider, model=%s", settings.model_name)
    return ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=api_key,
        base_url=settings.zhipu_api_base,
        max_retries=3,
        request_timeout=settings.llm_request_timeout,
    )


def _create_qwen_llm(settings) -> ChatOpenAI:
    """
    通义千问：通过阿里云 DashScope 的 OpenAI 兼容接口
    base_url=https://dashscope.aliyuncs.com/compatible-mode/v1
    """
    logger.info("使用通义千问 provider, model=%s", settings.model_name)
    return ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        max_retries=3,
        request_timeout=settings.llm_request_timeout,
    )


def _create_ernie_llm(settings) -> ChatOpenAI:
    """
    百度文心：通过千帆 v2 OpenAI 兼容接口
    需要设置 api_key（即 API Key）和 secret_key（即 Secret Key）
    千帆 v2 接口要求 Bearer token 格式：需要在外部生成 access_token
    这里简化处理，使用 OpenAI 兼容模式
    """
    logger.info("使用百度文心 provider, model=%s", settings.model_name)
    return ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.ernie_api_key,
        base_url=settings.ernie_api_base,
        max_retries=3,
        request_timeout=settings.llm_request_timeout,
    )


def _create_ollama_llm(settings) -> ChatOpenAI:
    """
    Ollama 本地推理：通过 OpenAI 兼容接口
    base_url=http://localhost:11434/v1
    """
    base_url = settings.ollama_base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    logger.info("使用 Ollama 本地 provider, model=%s, base=%s", settings.model_name, base_url)
    return ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key="ollama",  # Ollama 不需要真实 key，但字段不能为空
        base_url=base_url,
        max_retries=3,
        request_timeout=settings.llm_request_timeout,
    )


def list_supported_providers() -> list[str]:
    """返回支持的 provider 列表"""
    return ["openai", "zhipu", "qwen", "ernie", "ollama"]
