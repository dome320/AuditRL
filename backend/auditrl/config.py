import hashlib
import json
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AUDITRL_",
        env_file=(".env", "../.env"),
        extra="ignore",
    )

    model: str = "anthropic/claude-sonnet-4-5"
    temperature: float = 0.0
    max_tokens: int = 4096
    data_dir: Path = Path("./var")
    db_url: str = ""
    extraction_chunk_chars: int = 6000
    relative_tolerance: float = 1e-6
    retrieval_top_k: int = 5
    llm_rationale: bool = True
    reviewer: str = "anonymous"

    @property
    def database_url(self) -> str:
        if self.db_url:
            return self.db_url
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{(self.data_dir / 'auditrl.db').resolve()}"

    def config_hash(self) -> str:
        payload = self.model_dump(include={"model", "temperature", "relative_tolerance"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


@lru_cache
def get_settings() -> Settings:
    return Settings()
