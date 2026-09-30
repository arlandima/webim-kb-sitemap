"""Конфигурация. Всё берётся из переменных окружения / файла .env (секреты не хранятся в коде)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #")[0].strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)


_load_dotenv(ROOT / ".env")

KB_HOST = "webim.ru"
KB_BASE = "https://webim.ru/kb/"
KB_SITEMAP = "https://webim.ru/kb/sitemap.xml"
USER_AGENT = "WebimAIKnowledge-Prototype/0.1 (internal proof of concept; polite crawler; contact: repo owner)"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR") or ROOT / "data"))
    request_delay: float = float(_env("CRAWL_DELAY", "0.3"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "multilingual-e5-large"))
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "bge-reranker-v2-m3"))
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai").lower())
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL"))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8000")))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "index.sqlite"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache" / "pages"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def llm_configured(self) -> bool:
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key or self.llm_api_key) and bool(self.llm_model)
        if self.llm_provider == "openai":
            return bool(self.llm_base_url) and bool(self.llm_model)
        return False


def get_settings() -> Settings:
    return Settings()
