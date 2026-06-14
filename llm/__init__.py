# llm/__init__.py
from .base import BaseLLM
from .anthropic_client import AnthropicLLM
from .openai_client import OpenAILLM


def get_llm(provider: str = "anthropic") -> BaseLLM:
    if provider == "anthropic":
        return AnthropicLLM()
    elif provider == "openai":
        return OpenAILLM()
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose 'anthropic' or 'openai'")