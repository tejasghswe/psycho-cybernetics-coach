from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PROJECT_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"), extra="ignore"
    )

    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-5"

    checkpoint_db_path: str = str(_PROJECT_ROOT / ".cache" / "checkpoints.sqlite")

    otel_service_name: str = "monologue-coach"
    otel_console_export: bool = True
    otel_exporter_otlp_endpoint: str | None = None

    # Critic: one retry allowed, i.e. at most 2 Interpreter calls per session.
    max_critic_attempts: int = 1
    # Question Generator: hard cap enforced in the graph, never trusted to the model.
    max_questions: int = 3


settings = Settings()
