from harness.config import Config

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
        raise KeyError(
            f"Model not registered: {model_name}. Did you call init_adapters()?"
        )
    return _registry[model_name]
