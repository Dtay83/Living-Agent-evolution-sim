from pathlib import Path

import pytest

from living_agent_v2.engine import create_world
from living_agent_v2.engine import advance_world
from living_agent_v2.models import CreateWorldRequest
from living_agent_v2.store import SQLiteStore, WorldAlreadyExistsError


def test_world_persists_across_store_instances(tmp_path: Path) -> None:
    path = tmp_path / "worlds.db"
    first_store = SQLiteStore(path)
    first_store.initialize()
    world = create_world(CreateWorldRequest(world_id="persistent", seed=7, agent_count=2))
    first_store.create_world(world)

    second_store = SQLiteStore(path)
    second_store.initialize()

    assert second_store.get_world("persistent") == world


def test_duplicate_world_is_rejected(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "worlds.db")
    store.initialize()
    world = create_world(CreateWorldRequest(world_id="duplicate", seed=7, agent_count=2))
    store.create_world(world)

    with pytest.raises(WorldAlreadyExistsError):
        store.create_world(world)


def test_tick_and_discovery_are_committed_together(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "worlds.db")
    store.initialize()
    world = create_world(CreateWorldRequest(world_id="atomic", seed=7, agent_count=2))
    store.create_world(world)
    updated, discoveries = advance_world(world, 5)

    store.commit_tick(updated, discoveries, expected_version=0)

    assert store.get_world("atomic").tick == 5
    assert store.list_discoveries("atomic") == discoveries


def test_stale_tick_is_rejected(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "worlds.db")
    store.initialize()
    world = create_world(CreateWorldRequest(world_id="concurrent", seed=7, agent_count=2))
    store.create_world(world)
    updated, discoveries = advance_world(world, 5)
    store.commit_tick(updated, discoveries, expected_version=0)

    with pytest.raises(RuntimeError, match="changed concurrently"):
        store.commit_tick(updated, discoveries, expected_version=0)

    assert store.get_world("concurrent").tick == 5
    assert store.list_discoveries("concurrent") == discoveries
