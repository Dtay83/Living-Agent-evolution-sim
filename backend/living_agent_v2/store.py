from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .memory import MemoryMatch, cosine_similarity, embed_text
from .models import Discovery, WorldState


class WorldAlreadyExistsError(Exception):
    pass


class WorldNotFoundError(Exception):
    pass


class SQLiteStore:
    def __init__(self, path: Path, embedding_dimensions: int = 64) -> None:
        self.path = path
        self.embedding_dimensions = embedding_dimensions

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS worlds (
                    world_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS discoveries (
                    discovery_id TEXT PRIMARY KEY,
                    world_id TEXT NOT NULL,
                    tick INTEGER NOT NULL,
                    discovery_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(world_id) REFERENCES worlds(world_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    world_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(world_id) REFERENCES worlds(world_id) ON DELETE CASCADE
                );
                """
            )

    def create_world(self, world: WorldState) -> WorldState:
        payload = world.model_dump_json()
        try:
            with self.connect() as connection:
                connection.execute(
                    "INSERT INTO worlds(world_id, version, state_json, updated_at) VALUES (?, 0, ?, ?)",
                    (world.world_id, payload, world.updated_at.isoformat()),
                )
        except sqlite3.IntegrityError as error:
            raise WorldAlreadyExistsError(world.world_id) from error
        return world

    def get_world_with_version(self, world_id: str) -> tuple[WorldState, int]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT state_json, version FROM worlds WHERE world_id = ?", (world_id,)
            ).fetchone()
        if row is None:
            raise WorldNotFoundError(world_id)
        return WorldState.model_validate_json(row["state_json"]), int(row["version"])

    def get_world(self, world_id: str) -> WorldState:
        world, _ = self.get_world_with_version(world_id)
        return world

    def update_world(self, world: WorldState, expected_version: int) -> None:
        with self.connect() as connection:
            self._update_world(connection, world, expected_version)

    def commit_tick(
        self,
        world: WorldState,
        discoveries: list[Discovery],
        expected_version: int,
    ) -> None:
        """Persist a tick and its accepted discoveries as one transaction."""
        with self.connect() as connection:
            self._update_world(connection, world, expected_version)
            for discovery in discoveries:
                self._insert_discovery(connection, discovery)

    def _update_world(
        self,
        connection: sqlite3.Connection,
        world: WorldState,
        expected_version: int,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE worlds
            SET state_json = ?, version = version + 1, updated_at = ?
            WHERE world_id = ? AND version = ?
            """,
            (
                world.model_dump_json(),
                world.updated_at.isoformat(),
                world.world_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("world changed concurrently; retry the tick")

    def add_discovery(self, discovery: Discovery) -> None:
        with self.connect() as connection:
            self._insert_discovery(connection, discovery)

    def _insert_discovery(
        self, connection: sqlite3.Connection, discovery: Discovery
    ) -> None:
        vector = embed_text(discovery.hypothesis, self.embedding_dimensions)
        connection.execute(
            """
            INSERT OR IGNORE INTO discoveries(
                discovery_id, world_id, tick, discovery_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                discovery.discovery_id,
                discovery.world_id,
                discovery.tick,
                discovery.model_dump_json(),
                discovery.created_at.isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO memories(
                memory_id, world_id, content, vector_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                discovery.discovery_id,
                discovery.world_id,
                discovery.hypothesis,
                json.dumps(vector),
                discovery.created_at.isoformat(),
            ),
        )

    def list_discoveries(self, world_id: str) -> list[Discovery]:
        self.get_world(world_id)
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT discovery_json FROM discoveries
                WHERE world_id = ? ORDER BY tick, created_at
                """,
                (world_id,),
            ).fetchall()
        return [Discovery.model_validate_json(row["discovery_json"]) for row in rows]

    def search_memory(self, world_id: str, query: str, limit: int = 5) -> list[MemoryMatch]:
        query_vector = embed_text(query, self.embedding_dimensions)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT memory_id, content, vector_json FROM memories WHERE world_id = ?",
                (world_id,),
            ).fetchall()
        matches = [
            MemoryMatch(
                memory_id=row["memory_id"],
                content=row["content"],
                score=cosine_similarity(query_vector, json.loads(row["vector_json"])),
            )
            for row in rows
        ]
        return sorted(matches, key=lambda match: match.score, reverse=True)[:limit]
