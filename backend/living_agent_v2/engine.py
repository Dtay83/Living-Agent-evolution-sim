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


# --- Population dynamics tuning -------------------------------------------------
CARRYING_CAPACITY = 60          # soft limit; crowding reduces foraging yield
MAX_AGENTS = 200                # hard cap so ticks stay cheap
REPRODUCTION_ENERGY = 85.0      # energy required before an agent can split
REPRODUCTION_MIN_AGE = 3        # agents must mature before reproducing
OFFSPRING_SHARE = 0.45          # fraction of parent energy handed to the child
SENESCENCE_AGE = 220            # age at which upkeep starts climbing
MIN_POPULATION = 4              # reseed floor to prevent total extinction


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


def _forage_yield(agent: AgentState, crowding: float, randomizer: random.Random) -> float:
    """Energy harvested this tick, throttled by how crowded the world is."""
    base = randomizer.uniform(0.8, 2.6) * (0.4 + 0.6 * agent.genome.curiosity)
    return base * crowding


def _upkeep(agent: AgentState, randomizer: random.Random) -> float:
    """Metabolic cost; rises once an agent passes its senescence age."""
    cost = randomizer.uniform(0.2, 1.0) * (1.1 - 0.5 * agent.genome.energy_efficiency)
    if agent.age > SENESCENCE_AGE:
        cost *= 1.0 + (agent.age - SENESCENCE_AGE) / 120.0
    return cost


def _spawn_offspring(
    parent: AgentState, tick: int, index: int, randomizer: random.Random
) -> tuple[AgentState, AgentState]:
    """Split a parent into a depleted parent and a mutated child."""
    inherited = round(parent.energy * OFFSPRING_SHARE, 6)
    child = AgentState(
        agent_id=f"agent-{tick}-{index}-{randomizer.randrange(1_000_000):06d}",
        energy=inherited,
        age=0,
        genome=mutate_genome(parent.genome, randomizer),
    )
    drained = parent.model_copy(update={"energy": round(parent.energy - inherited, 6)})
    return drained, child


def _reseed_agent(tick: int, index: int, randomizer: random.Random) -> AgentState:
    """Emergency colonist so a world can recover instead of flatlining."""
    return AgentState(
        agent_id=f"seed-{tick}-{index}",
        energy=round(randomizer.uniform(70.0, 95.0), 6),
        genome=Genome(
            curiosity=randomizer.uniform(0.35, 0.85),
            compression_bias=randomizer.uniform(0.35, 0.85),
            symbolic_precision=randomizer.uniform(0.45, 0.95),
            sociality=randomizer.uniform(0.25, 0.85),
            mutation_rate=randomizer.uniform(0.01, 0.10),
            energy_efficiency=randomizer.uniform(0.45, 0.90),
        ),
    )


def advance_world(world: WorldState, steps: int) -> tuple[WorldState, list[Discovery]]:
    discoveries: list[Discovery] = []
    current = world.model_copy(deep=True)

    for _ in range(steps):
        previous_values = _state_values(current.agents)
        next_tick = current.tick + 1
        randomizer = random.Random(f"{current.seed}:{next_tick}")

        population = max(1, len(current.agents))
        crowding = CARRYING_CAPACITY / (CARRYING_CAPACITY + population)

        survivors: list[AgentState] = []
        newborns: list[AgentState] = []

        for agent in current.agents:
            energy = agent.energy - _upkeep(agent, randomizer)
            energy += _forage_yield(agent, crowding, randomizer)
            matured = agent.model_copy(
                update={
                    "age": agent.age + 1,
                    "energy": round(max(0.0, energy), 6),
                    "genome": mutate_genome(agent.genome, randomizer),
                }
            )

            # Starvation removes the agent from the population entirely.
            if matured.energy <= 0.0:
                continue

            can_reproduce = (
                matured.energy >= REPRODUCTION_ENERGY
                and matured.age >= REPRODUCTION_MIN_AGE
                and len(current.agents) + len(newborns) < MAX_AGENTS
            )
            if can_reproduce:
                matured, child = _spawn_offspring(
                    matured, next_tick, len(newborns), randomizer
                )
                newborns.append(child)

            survivors.append(matured)

        updated_agents = survivors + newborns

        # Never let the world go fully extinct; reseed a minimal founder group.
        while len(updated_agents) < MIN_POPULATION:
            updated_agents.append(
                _reseed_agent(next_tick, len(updated_agents), randomizer)
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

