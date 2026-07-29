from __future__ import annotations

import random
import uuid

from .discovery import make_discovery, propose_hypothesis
from .genome import random_genome
from .infodynamics import calculate_metrics, evolve_cell, movement_cost
from .memory import MemoryStore
from .models import AgentMemoryRef, AgentV2, CellV2, CreateWorldRequest, DiscoveryStatus, Terrain, WorldV2


class SimulationEngine:
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    def create_world(self, request: CreateWorldRequest) -> WorldV2:
        rng = random.Random(request.seed)
        grid = self._create_grid(request.width, request.height, rng)
        agents = self._create_agents(request.agent_count, request.width, request.height, rng)
        return WorldV2(
            id=str(uuid.uuid4()),
            seed=request.seed,
            tick=0,
            width=request.width,
            height=request.height,
            agents=agents,
            grid=grid,
            discoveries=[],
            metrics=calculate_metrics(grid, agents),
        )

    def tick(self, world: WorldV2) -> WorldV2:
        rng = random.Random(world.seed + world.tick + 1)
        pressure = world.metrics.information_pressure
        grid = [[evolve_cell(cell, pressure) for cell in row] for row in world.grid]
        agents: list[AgentV2] = []
        discoveries = list(world.discoveries)

        for agent in world.agents:
            moved_agent = self._move_agent(agent, world.width, world.height, rng, pressure)
            cell = grid[moved_agent.y][moved_agent.x]
            energy = max(0.0, moved_agent.energy + cell.resource - movement_cost(moved_agent, pressure))
            cell.resource = 0.0
            cell.information_density += 0.05 * moved_agent.genome.curiosity
            moved_agent = moved_agent.model_copy(update={"energy": round(energy, 6)})

            if moved_agent.energy > 0:
                moved_agent, discoveries = self._maybe_discover(moved_agent, world.tick + 1, rng, discoveries)
                agents.append(moved_agent)

        metrics = calculate_metrics(grid, agents)
        return world.model_copy(
            update={
                "tick": world.tick + 1,
                "grid": grid,
                "agents": agents,
                "discoveries": discoveries,
                "metrics": metrics,
            }
        )

    def _create_grid(self, width: int, height: int, rng: random.Random) -> list[list[CellV2]]:
        grid: list[list[CellV2]] = []
        for _y in range(height):
            row: list[CellV2] = []
            for _x in range(width):
                roll = rng.random()
                if roll < 0.08:
                    row.append(CellV2(terrain=Terrain.RESOURCE, information_density=0.25, resource=4.0))
                elif roll < 0.12:
                    row.append(CellV2(terrain=Terrain.HAZARD, information_density=0.6, resource=0.0))
                else:
                    row.append(CellV2(terrain=Terrain.PLAIN, information_density=rng.random() * 0.2, resource=0.0))
            grid.append(row)
        return grid

    def _create_agents(self, count: int, width: int, height: int, rng: random.Random) -> list[AgentV2]:
        agents: list[AgentV2] = []
        occupied: set[tuple[int, int]] = set()

        while len(agents) < count:
            x = rng.randrange(width)
            y = rng.randrange(height)
            if (x, y) in occupied:
                continue
            occupied.add((x, y))
            agents.append(
                AgentV2(
                    id=str(uuid.uuid4()),
                    x=x,
                    y=y,
                    energy=round(rng.uniform(12, 24), 6),
                    generation=0,
                    genome=random_genome(rng),
                )
            )

        return agents

    def _move_agent(
        self,
        agent: AgentV2,
        width: int,
        height: int,
        rng: random.Random,
        pressure: float,
    ) -> AgentV2:
        if rng.random() < agent.genome.compression_bias:
            dx, dy = 0, 0
        else:
            dx, dy = rng.choice([(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)])

        x = max(0, min(width - 1, agent.x + dx))
        y = max(0, min(height - 1, agent.y + dy))
        return agent.model_copy(update={"x": x, "y": y})

    def _maybe_discover(
        self,
        agent: AgentV2,
        tick: int,
        rng: random.Random,
        discoveries: list,
    ) -> tuple[AgentV2, list]:
        chance = agent.genome.curiosity * agent.genome.symbolic_precision * 0.08
        if rng.random() > chance:
            return agent, discoveries

        hypothesis = propose_hypothesis(agent)
        discovery = make_discovery(agent, tick, hypothesis)
        discoveries = [*discoveries, discovery]

        memory = self.memory_store.add(agent.id, discovery.status.value, hypothesis)
        memories = [*agent.memories, AgentMemoryRef(memory_id=memory.id, concept_type=memory.concept_type)]
        accepted_ids = list(agent.accepted_discovery_ids)

        if discovery.status == DiscoveryStatus.ACCEPTED:
            accepted_ids.append(discovery.id)

        return agent.model_copy(update={"memories": memories, "accepted_discovery_ids": accepted_ids}), discoveries

