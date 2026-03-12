from __future__ import annotations

from dataclasses import replace

from pipeline.schemas import LLMConfig, ProviderPreset


PROVIDER_DEFAULTS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "chat_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "chat_model": "deepseek-ai/DeepSeek-V3.2",
        "embedding_model": "Qwen/Qwen3-Embedding-8B",
    },
    "custom": {
        "base_url": "",
        "chat_model": "",
        "embedding_model": "",
    },
}


def provider_defaults(preset: ProviderPreset) -> dict[str, str]:
    return dict(PROVIDER_DEFAULTS[preset])


def apply_provider_defaults(config: LLMConfig, preset: ProviderPreset) -> LLMConfig:
    defaults = provider_defaults(preset)
    return replace(
        config,
        provider_preset=preset,
        base_url=defaults["base_url"],
        chat_model=defaults["chat_model"],
        embedding_model=defaults["embedding_model"],
    )
