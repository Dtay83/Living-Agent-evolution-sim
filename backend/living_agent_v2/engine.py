from __future__ import annotations

import random
from datetime import datetime, timezone
from uuid import uuid4

from .discovery import arbitrate_hypothesis
from .genome import mutate_genome
from .infodynamics import calculate_metrics
from .models import AgentState, CreateWorldRequest, Discovery, Genome, WorldState


def _state_values(agents: list[AgentState]) -> list[int]:
    return [min(255, max(0, round(agent.energy * 2.55))) for agent in agents]


def create_world(request: CreateWorldRequest) -> WorldState:
    world_id = request.world_id or f"world-{uuid4().hex[:12]}"
    randomizer = random.Random(request.seed)
    agents = [
        AgentState(
            agent_id=f"agent-{index + 1}",
            energy=round(randomizer.uniform(75.0, 100.0), 6),
            genome=Genome(
                curiosity=randomizer.uniform(0.35, 0.85),
                compression_bias=randomizer.uniform(0.35, 0.85),
                symbolic_precision=randomizer.uniform(0.45, 0.95),
                sociality=randomizer.uniform(0.25, 0.85),
                mutation_rate=randomizer.uniform(0.01, 0.10),
                energy_efficiency=randomizer.uniform(0.45, 0.90),
            ),
        )
        for index in range(request.agent_count)
    ]
    return WorldState(
        world_id=world_id,
        seed=request.seed,
        agents=agents,
        metrics=calculate_metrics(_state_values(agents)),
    )


def _propose_discovery(world: WorldState) -> Discovery | None:
    if world.tick == 0 or world.tick % 5 != 0:
        return None
    agent = max(world.agents, key=lambda item: item.genome.symbolic_precision)
    mass = max(1, round(agent.energy / 20))
    c_value = 10
    exact_energy = mass * c_value**2
    precision = agent.genome.symbolic_precision
    reported_energy = exact_energy if precision >= 0.65 else exact_energy + 1
    hypothesis = (
        f"energy equals {reported_energy}, mass equals {mass}, c equals {c_value}"
    )
    result = arbitrate_hypothesis(hypothesis)
    return Discovery(
        world_id=world.world_id,
        tick=world.tick,
        hypothesis=hypothesis,
        valid=result.valid,
        feedback=result.feedback,
        residual=result.residual,
    )


def advance_world(world: WorldState, steps: int) -> tuple[WorldState, list[Discovery]]:
    discoveries: list[Discovery] = []
    current = world.model_copy(deep=True)

    for _ in range(steps):
        previous_values = _state_values(current.agents)
        next_tick = current.tick + 1
        randomizer = random.Random(f"{current.seed}:{next_tick}")
        updated_agents: list[AgentState] = []
        for agent in current.agents:
            energy_cost = randomizer.uniform(0.2, 1.2) * (
                1.1 - 0.5 * agent.genome.energy_efficiency
            )
            energy_gain = randomizer.uniform(0.0, 0.7) * agent.genome.curiosity
            updated_agents.append(
                agent.model_copy(
                    update={
                        "age": agent.age + 1,
                        "energy": round(max(0.0, agent.energy - energy_cost + energy_gain), 6),
                        "genome": mutate_genome(agent.genome, randomizer),
                    }
                )
            )

        current.tick = next_tick
        current.agents = updated_agents
        current.metrics = calculate_metrics(_state_values(updated_agents), previous_values)
        current.updated_at = datetime.now(timezone.utc)
        proposed = _propose_discovery(current)
        if proposed is not None and proposed.valid:
            current.accepted_discoveries.append(proposed)
            discoveries.append(proposed)

    return current, discoveries

