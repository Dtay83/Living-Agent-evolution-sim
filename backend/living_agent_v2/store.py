from __future__ import annotations

from .engine import SimulationEngine
from .memory import MemoryStore
from .models import CreateWorldRequest, WorldV2


class WorldStore:
    def __init__(self):
        self.memory_store = MemoryStore()
        self.engine = SimulationEngine(self.memory_store)
        self._worlds: dict[str, WorldV2] = {}

    def create(self, request: CreateWorldRequest) -> WorldV2:
        world = self.engine.create_world(request)
        self._worlds[world.id] = world
        return world

    def get(self, world_id: str) -> WorldV2 | None:
        return self._worlds.get(world_id)

    def tick(self, world_id: str, steps: int) -> WorldV2 | None:
        world = self.get(world_id)
        if world is None:
            return None

        for _ in range(steps):
            world = self.engine.tick(world)

        self._worlds[world_id] = world
        return world

