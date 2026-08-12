from living_agent_v2.engine import advance_world, create_world
from living_agent_v2.models import CreateWorldRequest


def test_world_creation_is_deterministic() -> None:
    request = CreateWorldRequest(world_id="test", seed=42, agent_count=4)

    first = create_world(request)
    second = create_world(request)

    assert first.agents == second.agents
    assert first.metrics == second.metrics


def test_ticks_are_deterministic_from_same_state() -> None:
    world = create_world(CreateWorldRequest(world_id="test", seed=42, agent_count=4))

    first, _ = advance_world(world, 3)
    second, _ = advance_world(world, 3)

    assert first.tick == 3
    assert first.agents == second.agents
    assert first.metrics == second.metrics


def test_validated_discovery_is_recorded_on_fifth_tick() -> None:
    world = create_world(CreateWorldRequest(world_id="test", seed=42, agent_count=4))

    updated, discoveries = advance_world(world, 5)

    assert updated.tick == 5
    assert len(discoveries) == 1
    assert discoveries[0].valid is True
    assert updated.accepted_discoveries == discoveries

