from living_agent_v2.engine import SimulationEngine
from living_agent_v2.memory import MemoryStore
from living_agent_v2.models import CreateWorldRequest


def test_create_world_is_deterministic_for_same_seed():
    engine = SimulationEngine(MemoryStore())
    request = CreateWorldRequest(seed=42, width=8, height=6, agent_count=3)

    first = engine.create_world(request)
    second = engine.create_world(request)

    assert [agent.model_dump(exclude={"id"}) for agent in first.agents] == [
        agent.model_dump(exclude={"id"}) for agent in second.agents
    ]
    assert first.metrics == second.metrics


def test_tick_advances_world_and_preserves_dimensions():
    engine = SimulationEngine(MemoryStore())
    world = engine.create_world(CreateWorldRequest(seed=7, width=8, height=6, agent_count=3))

    next_world = engine.tick(world)

    assert next_world.tick == 1
    assert next_world.width == world.width
    assert next_world.height == world.height
    assert len(next_world.grid) == world.height
    assert len(next_world.grid[0]) == world.width

