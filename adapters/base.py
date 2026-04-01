from abc import ABC, abstractmethod

from crewai import LLM


class BaseAdapter(ABC):
    """All provider adapters implement this interface."""

    @property
    @abstractmethod
    def chat_llm(self) -> LLM:
        """Return a CrewAI LLM instance for use in Agent."""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a list of texts."""
        ...

    @abstractmethod
    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[dict]:
        """Rerank documents for a query, return list of {index, document, relevance_score}."""
        ...
