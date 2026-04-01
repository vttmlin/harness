# Multi-LLM Provider Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified LLM provider adapter layer that lets CrewAI agents use Zhipu, MiniMax, and SiliconFlow through config-driven model registration, with chat/embed/rerank support.

**Architecture:** Provider-level `driver` field routes to `AnthropicAdapter` or `SiliconFlowAdapter`. Each adapter exposes `chat_llm` (for CrewAI Agent), `embed()`, and `rerank()`. All API keys from environment variables, resolved at load time.

**Tech Stack:** Python 3.13, Pydantic 2, CrewAI 1.12, OpenAI SDK, httpx

---

## Task 1: config.py — Pydantic models + env var resolution

**Files:**
- Create: `config.py`

- [ ] **Step 1: Write config.py with Provider/Model/Config models and env var resolution**

```python
# config.py
import os
import re
from typing import Literal

from pydantic import BaseModel


def _resolve_env(value: str) -> str:
    """Resolve ${ENV_VAR} placeholders from environment variables."""
    pattern = re.compile(r"\$\{(\w+)\}")
    matches = pattern.findall(value)
    for var in matches:
        value = value.replace(f"${{{var}}}", os.environ.get(var, ""))
    return value


class Provider(BaseModel):
    name: str
    driver: Literal["anthropic", "siliconflow"]
    base_url: str
    api_key: str

    def model_post_init(self) -> None:
        self.api_key = _resolve_env(self.api_key)
        self.base_url = _resolve_env(self.base_url)


class Model(BaseModel):
    name: str
    provider: str
    model: str
    roles: list[Literal["chat", "embed", "rerank"]]


class Config(BaseModel):
    providers: list[Provider]
    models: list[Model]

    def get_provider(self, name: str) -> Provider:
        for p in self.providers:
            if p.name == name:
                return p
        raise KeyError(f"Provider not found: {name}")

    def get_models(self, provider_name: str) -> list[Model]:
        return [m for m in self.models if m.provider == provider_name]


def load_config(path: str = "config.yml") -> Config:
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
```

- [ ] **Step 2: Add pyyaml to dependencies**

Modify `pyproject.toml`:
```toml
dependencies = [
    "crewai>=1.12.2",
    "mlflow>=3.10.1",
    "pydantic>=2.11.10",
    "pyyaml>=6.0",
    "httpx>=0.28.0",
]
```

Run: `uv sync`

- [ ] **Step 3: Commit**

```bash
git add config.py pyproject.toml uv.lock
git commit -m "feat: add Pydantic config models with env var resolution"
```

---

## Task 2: adapters/base.py — BaseAdapter abstract class

**Files:**
- Create: `adapters/base.py`
- Modify: `adapters/__init__.py`

- [ ] **Step 1: Write adapters/base.py**

```python
# adapters/base.py
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
```

- [ ] **Step 2: Commit**

```bash
git add adapters/base.py
git commit -m "feat: add BaseAdapter abstract class"
```

---

## Task 3: adapters/anthropic.py — AnthropicAdapter (Zhipu / MiniMax)

**Files:**
- Create: `adapters/anthropic.py`

- [ ] **Step 1: Write adapters/anthropic.py**

```python
# adapters/anthropic.py
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
            raise ValueError(f"No chat model configured for provider {self.provider.name}")
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

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[dict]:
        raise NotImplementedError(
            f"Provider {self.provider.name} (driver=anthropic) does not support rerank"
        )
```

- [ ] **Step 2: Commit**

```bash
git add adapters/anthropic.py
git commit -m "feat: add AnthropicAdapter for Zhipu and MiniMax"
```

---

## Task 4: adapters/siliconflow.py — SiliconFlowAdapter (chat + embed + rerank)

**Files:**
- Create: `adapters/siliconflow.py`

- [ ] **Step 1: Write adapters/siliconflow.py**

```python
# adapters/siliconflow.py
import httpx

from crewai import LLM
from openai import OpenAI

from config import Model, Provider
from .base import BaseAdapter


class SiliconFlowAdapter(BaseAdapter):
    """
    Provider driver: siliconflow
    chat  -> OpenAI SDK via /v1/chat/completions
    embed -> OpenAI SDK via /v1/embeddings
    rerank -> httpx via /v1/rerank
    """

    def __init__(self, provider: Provider, models: list[Model]):
        self.provider = provider
        self._chat_models = [m for m in models if "chat" in m.roles]
        self._embed_models = [m for m in models if "embed" in m.roles]
        self._rerank_models = [m for m in models if "rerank" in m.roles]

        self._client = OpenAI(
            base_url=provider.base_url,
            api_key=provider.api_key,
        )
        self._chat_model_name = self._chat_models[0].model if self._chat_models else None
        self._embed_model_name = self._embed_models[0].model if self._embed_models else None
        self._rerank_model_name = self._rerank_models[0].model if self._rerank_models else None

    @property
    def chat_llm(self) -> LLM:
        if not self._chat_model_name:
            raise ValueError(f"No chat model configured for provider {self.provider.name}")
        return LLM(
            model=self._chat_model_name,
            base_url=self.provider.base_url,
            api_key=self.provider.api_key,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._embed_model_name:
            raise ValueError(f"No embed model configured for provider {self.provider.name}")
        response = self._client.embeddings.create(
            model=self._embed_model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[dict]:
        if not self._rerank_model_name:
            raise ValueError(f"No rerank model configured for provider {self.provider.name}")
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
```

- [ ] **Step 2: Commit**

```bash
git add adapters/siliconflow.py
git commit -m "feat: add SiliconFlowAdapter with chat/embed/rerank support"
```

---

## Task 5: adapters/__init__.py — factory functions

**Files:**
- Create: `adapters/__init__.py`

- [ ] **Step 1: Write adapters/__init__.py**

```python
# adapters/__init__.py
from config import Config

from .anthropic import AnthropicAdapter
from .base import BaseAdapter
from .siliconflow import SiliconFlowAdapter

_registry: dict[str, BaseAdapter] = {}


def init_adapters(config: Config) -> None:
    """Initialize all adapters from config and register them by model name."""
    _registry.clear()
    for provider in config.providers:
        models = config.get_models(provider.name)
        if provider.driver == "anthropic":
            adapter: BaseAdapter = AnthropicAdapter(provider, models)
        elif provider.driver == "siliconflow":
            adapter = SiliconFlowAdapter(provider, models)
        else:
            raise ValueError(f"Unknown driver: {provider.driver}")

        for model in models:
            _registry[model.name] = adapter


def load_adapter(model_name: str) -> BaseAdapter:
    """Get the adapter for a given model name."""
    if model_name not in _registry:
        raise KeyError(f"Model not registered: {model_name}. Did you call init_adapters()?")
    return _registry[model_name]
```

- [ ] **Step 2: Commit**

```bash
git add adapters/__init__.py
git commit -m "feat: add adapter factory with init_adapters and load_adapter"
```

---

## Task 6: crewai_integration.py + agents/__init__.py

**Files:**
- Create: `crewai_integration.py`
- Create: `agents/__init__.py`

- [ ] **Step 1: Write crewai_integration.py**

```python
# crewai_integration.py
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
```

- [ ] **Step 2: Write agents/__init__.py**

```python
# agents/__init__.py
from crewai_integration import build_agent, init

__all__ = ["build_agent", "init"]
```

- [ ] **Step 3: Commit**

```bash
git add crewai_integration.py agents/__init__.py
git commit -m "feat: add crewai_integration with build_agent and init helpers"
```

---

## Task 7: config.yml — actual configuration

**Files:**
- Modify: `config.yml`

- [ ] **Step 1: Write production config.yml with all three providers**

```yaml
providers:
  - name: zhipu
    driver: anthropic
    base_url: https://open.bigmodel.cn/api/anthropic
    api_key: "${ZHIPU_API_KEY}"

  - name: minimax
    driver: anthropic
    base_url: https://api.minimaxi.com/anthropic
    api_key: "${MINIMAX_API_KEY}"

  - name: siliconflow
    driver: siliconflow
    base_url: https://api.siliconflow.cn/v1
    api_key: "${SILICONFLOW_API_KEY}"

models:
  - name: GLM-5.1
    provider: zhipu
    model: GLM-5.1
    roles: [chat]

  - name: MiniMax-M2.7
    provider: minimax
    model: MiniMax-M2.7
    roles: [chat]

  - name: Qwen/Qwen3-Embedding-8B
    provider: siliconflow
    model: Qwen/Qwen3-Embedding-8B
    roles: [embed]

  - name: Qwen/Qwen3-Reranker-8B
    provider: siliconflow
    model: Qwen/Qwen3-Reranker-8B
    roles: [rerank]
```

- [ ] **Step 2: Commit**

```bash
git add config.yml
git commit -m "feat: add config.yml with zhipu/minimax/siliconflow providers"
```

---

## Task 8: Verification — run imports and basic sanity check

**Files:**
- Modify: `config.yml` (update .env.example path)

- [ ] **Step 1: Run import check**

```bash
cd /Users/vttmlin/workspace/part-time/harness
uv run python -c "
from config import load_config
from adapters import init_adapters, load_adapter

print('imports OK')

# Test config loading
config = load_config('config.yml')
print(f'providers: {[p.name for p in config.providers]}')
print(f'models: {[m.name for m in config.models]}')

# Test adapter init
init_adapters(config)
print('adapters initialized OK')

# Test model registry
glm_adapter = load_adapter('GLM-5.1')
minimax_adapter = load_adapter('MiniMax-M2.7')
sf_embed_adapter = load_adapter('Qwen/Qwen3-Embedding-8B')
sf_rerank_adapter = load_adapter('Qwen/Qwen3-Reranker-8B')
print('all models loadable OK')

# Verify driver routing
print(f'GLM driver: {type(glm_adapter).__name__}')
print(f'MiniMax driver: {type(minimax_adapter).__name__}')
print(f'SiliconFlow embed driver: {type(sf_embed_adapter).__name__}')
"
```

Expected output:
```
imports OK
providers: ['zhipu', 'minimax', 'siliconflow']
models: ['GLM-5.1', 'MiniMax-M2.7', 'Qwen/Qwen3-Embedding-8B', 'Qwen/Qwen3-Reranker-8B']
adapters initialized OK
all models loadable OK
GLM driver: AnthropicAdapter
MiniMax driver: AnthropicAdapter
SiliconFlow embed driver: SiliconFlowAdapter
```

- [ ] **Step 2: Verify .env.example is up to date**

Update `.env.example` with all three provider keys:

```bash
cat >> .env.example << 'EOF'

# LLM Provider API Keys
ZHIPU_API_KEY=your_zhipu_key_here
MINIMAX_API_KEY=your_minimax_key_here
SILICONFLOW_API_KEY=your_siliconflow_key_here
EOF
```

- [ ] **Step 3: Commit verification**

```bash
git add .env.example
git commit -m "test: verify adapter loading and driver routing"
```

---

## Self-Review Checklist

- [ ] Spec coverage: All 3 providers (zhipu, minimax, siliconflow), all 3 roles (chat, embed, rerank), config-driven with driver routing — covered in Tasks 1-7
- [ ] Placeholder scan: No TBD/TODO, all code is complete and runnable
- [ ] Type consistency: `BaseAdapter.chat_llm` property name consistent across all adapters, `embed()` and `rerank()` signatures match across all adapters
- [ ] Commit count: 8 tasks = 8 commits (one per task, atomic)
- [ ] Dependencies: httpx added in Task 1, all others use existing deps (crewai, pydantic, openai via crewai)

---

**Plan complete.** Saved to `docs/superpowers/plans/2026-04-01-multi-llm-provider-plan.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
