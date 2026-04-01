from __future__ import annotations

import httpx
from typing import Any

from crewai import LLM
from openai import OpenAI
from typing_extensions import final

from harness.config import Model, Provider
from .base import BaseAdapter


@final
class SiliconFlowAdapter(BaseAdapter):
    """
    Provider driver: siliconflow
    chat  -> OpenAI SDK via /v1/chat/completions
    embed -> OpenAI SDK via /v1/embeddings
    rerank -> httpx via /v1/rerank
    """

    provider: Provider
    _chat_models: list[Model]
    _embed_models: list[Model]
    _rerank_models: list[Model]
    _client: OpenAI
    _chat_model_name: str | None
    _embed_model_name: str | None
    _rerank_model_name: str | None

    def __init__(self, provider: Provider, models: list[Model]) -> None:
        self.provider = provider
        self._chat_models = [m for m in models if "chat" in m.roles]
        self._embed_models = [m for m in models if "embed" in m.roles]
        self._rerank_models = [m for m in models if "rerank" in m.roles]

        self._client = OpenAI(
            base_url=provider.base_url,
            api_key=provider.api_key,
        )
        self._chat_model_name = (
            self._chat_models[0].model if self._chat_models else None
        )
        self._embed_model_name = (
            self._embed_models[0].model if self._embed_models else None
        )
        self._rerank_model_name = (
            self._rerank_models[0].model if self._rerank_models else None
        )

    @property
    def chat_llm(self) -> LLM:
        if not self._chat_model_name:
            raise ValueError(
                f"No chat model configured for provider {self.provider.name}"
            )
        return LLM(
            model=self._chat_model_name,
            base_url=self.provider.base_url,
            api_key=self.provider.api_key,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._embed_model_name:
            raise ValueError(
                f"No embed model configured for provider {self.provider.name}"
            )
        response = self._client.embeddings.create(
            model=self._embed_model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[dict[str, Any]]:
        if not self._rerank_model_name:
            raise ValueError(
                f"No rerank model configured for provider {self.provider.name}"
            )
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{self.provider.base_url}/rerank",
                headers={
                    "Authorization": f"Bearer {self.provider.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._rerank_model_name,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                },
            )
            resp.raise_for_status()
            return resp.json()["results"]
