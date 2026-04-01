from crewai import Agent

from adapters import init_adapters, load_adapter
from config import load_config


def build_agent(
    role: str,
    goal: str,
    backstory: str,
    chat_model: str,
    *,
    embed_model: str | None = None,
    rerank_model: str | None = None,
) -> Agent:
    """
    Build a CrewAI Agent backed by a configured LLM provider.

    Args:
        role: Agent role
        goal: Agent goal
        backstory: Agent backstory
        chat_model: Model name for chat (e.g. "GLM-5.1")
        embed_model: Optional model name for embedding (e.g. "Qwen/Qwen3-Embedding-8B")
        rerank_model: Optional model name for reranking (e.g. "Qwen/Qwen3-Reranker-8B")
    """
    chat_adapter = load_adapter(chat_model)

    agent_kwargs: dict = {
        "role": role,
        "goal": goal,
        "backstory": backstory,
        "llm": chat_adapter.chat_llm,
        "verbose": True,
    }

    if embed_model:
        agent_kwargs["embedder"] = load_adapter(embed_model)

    if rerank_model:
        agent_kwargs["reranker"] = load_adapter(rerank_model)

    return Agent(**agent_kwargs)


def init(config_path: str = "config.yml") -> None:
    """Load config and initialize all adapters. Call once at startup."""
    config = load_config(config_path)
    init_adapters(config)
