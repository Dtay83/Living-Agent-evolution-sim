from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path
    embedding_dimensions: int = 64


def load_settings() -> Settings:
    return Settings(
        database_path=Path(
            os.environ.get("LIVING_AGENT_DB_PATH", "living_agent_v2.db")
        ).expanduser(),
    )
