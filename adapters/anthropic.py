from crewai import LLM

from config import Model, Provider
from .base import BaseAdapter


class AnthropicAdapter(BaseAdapter):
    """
    Provider driver: anthropic
    Handles Zhipu (GLM) and MiniMax chat models via Anthropic-compatible API.
    embed/rerank are not supported by these providers via this interface.
    """

    def __init__(self, provider: Provider, models: list[Model]):
        self.provider = provider
        self._chat_models = [m for m in models if "chat" in m.roles]
        self._embed_models = [m for m in models if "embed" in m.roles]
        self._rerank_models = [m for m in models if "rerank" in m.roles]

    @property
    def chat_llm(self) -> LLM:
        if not self._chat_models:
            raise ValueError(
                f"No chat model configured for provider {self.provider.name}"
            )
        model = self._chat_models[0]
        return LLM(
            model=model.model,
            base_url=self.provider.base_url,
            api_key=self.provider.api_key,
            model_type="anthropic",
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            f"Provider {self.provider.name} (driver=anthropic) does not support embed"
        )

    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[dict]:
        raise NotImplementedError(
            f"Provider {self.provider.name} (driver=anthropic) does not support rerank"
        )
