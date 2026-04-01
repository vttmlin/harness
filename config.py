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
