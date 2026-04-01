# 多厂商 LLM 适配层设计

**日期**: 2026-04-01
**状态**: 已批准

---

## 1. 背景与目标

本项目（CrewAI + MLflow）需要接入多个 LLM 厂商（Zhipu、迷你融、MiniMax、SiliconFlow）。
目标是通过统一的配置层加载不同厂商的模型，对上屏蔽厂商差异，对下支持 CrewAI Agent、Embedding、Rerank 等多种调用方式。

---

## 2. 架构概览

```
config.yml              ← 统一配置入口（providers + models）
       ↓
config.py               ← Pydantic 模型解析，构建 provider/model 注册表
       ↓
adapters/
  __init__.py           ← 工厂函数 load_adapter(name)
  base.py               ← 抽象基类 BaseAdapter
  anthropic.py          ← driver=anthropic 适配器（Zhipu / MiniMax）
  siliconflow.py        ← driver=siliconflow 适配器（SiliconFlow）
       ↓
crewai_integration.py   ← 从 adapters 构建 CrewAI LLM 实例
```

**核心原则**：
- `driver` 是 provider 级别字段，决定使用哪个 Adapter 类
- `driver: anthropic` → `AnthropicAdapter`：适用 Zhipu / MiniMax，chat 走 Anthropic 兼容协议
- `driver: siliconflow` → `SiliconFlowAdapter`：适用 SiliconFlow，chat 走 OpenAI 兼容协议，embed 走 OpenAI SDK，rerank 单独用 httpx 调
- API Key 直传，不做 JWT 封装
- Embedding / Rerank 通过同一 adapter 实例提供，与 Chat 接口共用 provider 配置
- CrewAI Agent 通过 `llm` 属性获取 Chat LLM 实例，通过 `embedder` / `reranker` 注入 RAG 能力

---

## 3. 配置设计

### 3.1 config.yml

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

**字段说明**：
- `driver`: 决定使用哪个 Adapter 类。`anthropic` → `AnthropicAdapter`；`siliconflow` → `SiliconFlowAdapter`
- `api_key` 支持 `${ENV_VAR}` 格式的环境变量占位符，运行时展开
- `roles`: 数组，标识该模型支持的调用类型（`chat` | `embed` | `rerank`）

### 3.2 配置解析（config.py）

```python
# config.py
from pydantic import BaseModel

class Provider(BaseModel):
    name: str
    driver: Literal["anthropic", "siliconflow"]
    base_url: str
    api_key: str  # 支持 ${ENV_VAR} 格式

class Model(BaseModel):
    name: str
    provider: str
    model: str           # 厂商内部的模型名
    roles: list[Literal["chat", "embed", "rerank"]]

class Config(BaseModel):
    providers: list[Provider]
    models: list[Model]
```

---

## 4. 适配器设计

### 4.1 BaseAdapter

```python
# adapters/base.py
from abc import ABC, abstractmethod
from crewai import LLM

class BaseAdapter(ABC):
    """所有厂商适配器的公共接口"""

    @property
    @abstractmethod
    def chat_llm(self) -> LLM:
        """返回 CrewAI LLM 实例，用于 Agent"""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """文本嵌入，返回 float 向量列表"""
        ...

    @abstractmethod
    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[dict]:
        """重排序，返回文档 ID / score 列表"""
        ...

    def supports(self, role: str) -> bool:
        """检查该适配器是否支持给定角色"""
        return role in self._roles
```

### 4.2 AnthropicAdapter（Zhipu / MiniMax）

```python
# adapters/anthropic.py
from crewai import LLM

class AnthropicAdapter(BaseAdapter):
    """
    适用厂商：Zhipu（GLM）、MiniMax
    driver: anthropic
    chat 走 Anthropic 兼容协议（Bearer Token 认证，无 JWT）
    embed/rerank 暂不支持
    """

    def __init__(
        self,
        provider: Provider,
        models: list[Model],
    ):
        self.provider = provider
        self._chat_models = [m for m in models if "chat" in m.roles]
        self._embed_models = [m for m in models if "embed" in m.roles]
        self._rerank_models = [m for m in models if "rerank" in m.roles]

    @property
    def chat_llm(self) -> LLM:
        model = self._chat_models[0]
        return LLM(
            model=model.model,            # 如 "GLM-5.1"
            base_url=self.provider.base_url,
            api_key=self.provider.api_key,
            model_type="anthropic",
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("AnthropicAdapter does not support embed")

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[dict]:
        raise NotImplementedError("AnthropicAdapter does not support rerank")
```

### 4.3 SiliconFlowAdapter（SiliconFlow）

```python
# adapters/siliconflow.py
import httpx
from crewai import LLM
from openai import OpenAI

class SiliconFlowAdapter(BaseAdapter):
    """
    适用厂商：SiliconFlow
    driver: siliconflow
    chat  → OpenAI SDK（/v1/chat/completions）
    embed → OpenAI SDK（/v1/embeddings）
    rerank → httpx（/v1/rerank）
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

    @property
    def chat_llm(self) -> LLM:
        return LLM(
            model=self._chat_model_name,
            base_url=self.provider.base_url,
            api_key=self.provider.api_key,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._embed_model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[dict]:
        with httpx.Client() as client:
            resp = client.post(
                f"{self.provider.base_url}/rerank",
                headers={"Authorization": f"Bearer {self.provider.api_key}"},
                json={
                    "model": self._rerank_models[0].model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                },
            )
            return resp.json()["results"]
```

---

## 5. 工厂函数

```python
# adapters/__init__.py
from config import Config

_registry: dict[str, BaseAdapter] = {}

def init_adapters(config: Config) -> None:
    """根据 config 初始化所有 adapter 并注册到全局注册表"""
    for provider in config.providers:
        models = [m for m in config.models if m.provider == provider.name]
        if provider.driver == "anthropic":
            adapter = AnthropicAdapter(provider, models)
        elif provider.driver == "siliconflow":
            adapter = SiliconFlowAdapter(provider, models)
        else:
            raise ValueError(f"Unknown driver: {provider.driver}")

        for model in models:
            _registry[model.name] = adapter

def load_adapter(model_name: str) -> BaseAdapter:
    """根据模型名获取对应 adapter"""
    return _registry[model_name]
```

---

## 6. CrewAI 集成

```python
# crewai_integration.py
from adapters import init_adapters, load_adapter
from config import load_config
from crewai import Agent, Crew, Task

def build_agent(
    role: str,
    goal: str,
    backstory: str,
    chat_model: str,
    embed_model: str | None = None,
    rerank_model: str | None = None,
) -> Agent:
    """构建带有多厂商 LLM 支持的 CrewAI Agent"""

    chat_adapter = load_adapter(chat_model)

    agent_kwargs = dict(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=chat_adapter.chat_llm,
        verbose=True,
    )

    if embed_model:
        agent_kwargs["embedder"] = load_adapter(embed_model)

    if rerank_model:
        agent_kwargs["reranker"] = load_adapter(rerank_model)

    return Agent(**agent_kwargs)

# 初始化
config = load_config("config.yml")
init_adapters(config)
```

---

## 7. 环境变量

所有 API Key 通过环境变量注入，不硬编码在配置文件中：

| 环境变量 | 用途 |
|---------|------|
| `ZHIPU_API_KEY` | Zhipu 智谱 GLM |
| `MINIMAX_API_KEY` | MiniMax |
| `SILICONFLOW_API_KEY` | SiliconFlow |

---

## 8. 目录结构

```
harness/
├── config.yml              # 统一配置（providers + models）
├── config.py               # Pydantic 配置模型 + 加载逻辑
├── adapters/
│   ├── __init__.py         # 工厂函数 load_adapter / init_adapters
│   ├── base.py             # BaseAdapter 抽象类
│   ├── anthropic.py        # AnthropicAdapter（Zhipu / MiniMax）
│   └── siliconflow.py      # SiliconFlowAdapter（SiliconFlow）
├── crewai_integration.py   # Agent 构建工具函数
└── agents/
    └── __init__.py         # crewai_integration 导出
```

---

## 9. 已知限制

1. **MiniMax reasoning_split**: `AnthropicAdapter.chat_llm` 返回的 LLM 实例暂不支持 `reasoning_split` 参数，如有需要后续扩展 `extra_body` 支持
2. **多 embedding/rerank 模型切换**: 当前设计每个 provider 只取第一个对应角色的模型做实例化；如需同时使用多个 embedding 模型，后续扩展 `models[role]` 为列表
3. **AnthropicAdapter 的 embed/rerank**: 目前 Zhipu / MiniMax 尚未接入 embed/rerank 接口，调用时直接 raise `NotImplementedError`

---

## 10. 验收标准

- [ ] `config.yml` 能正确加载 providers 和 models，`driver` 字段正确路由到对应 Adapter
- [ ] `AnthropicAdapter.chat_llm` 返回的 LLM 实例可正常用于 CrewAI Agent
- [ ] `SiliconFlowAdapter.chat_llm` 返回的 LLM 实例可正常用于 CrewAI Agent
- [ ] `SiliconFlowAdapter.embed()` 通过 OpenAI SDK 返回正确的向量列表
- [ ] `SiliconFlowAdapter.rerank()` 通过 httpx 返回正确的结果列表
- [ ] `load_adapter("GLM-5.1").chat_llm` 和 `load_adapter("MiniMax-M2.7").chat_llm` 返回正确的 base_url 和 api_key
- [ ] 环境变量占位符 `${VAR}` 能正确展开
- [ ] `build_agent()` 可以正确传入 embedder / reranker（可选）
