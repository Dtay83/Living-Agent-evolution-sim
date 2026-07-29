from __future__ import annotations

from enum import StrEnum
from pydantic import BaseModel, Field


class Terrain(StrEnum):
    PLAIN = "plain"
    RESOURCE = "resource"
    HAZARD = "hazard"


class DiscoveryStatus(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class GenomeV2(BaseModel):
    curiosity: float = Field(ge=0, le=1)
    compression_bias: float = Field(ge=0, le=1)
    symbolic_precision: float = Field(ge=0, le=1)
    sociality: float = Field(ge=0, le=1)
    mutation_rate: float = Field(ge=0, le=1)
    energy_efficiency: float = Field(ge=0, le=1)


class AgentMemoryRef(BaseModel):
    memory_id: str
    concept_type: str


class AgentV2(BaseModel):
    id: str
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    energy: float = Field(ge=0)
    generation: int = Field(ge=0)
    genome: GenomeV2
    accepted_discovery_ids: list[str] = Field(default_factory=list)
    memories: list[AgentMemoryRef] = Field(default_factory=list)


class CellV2(BaseModel):
    terrain: Terrain = Terrain.PLAIN
    information_density: float = Field(default=0.0, ge=0)
    resource: float = Field(default=0.0, ge=0)


class Discovery(BaseModel):
    id: str
    agent_id: str
    tick: int = Field(ge=0)
    hypothesis: str
    status: DiscoveryStatus
    feedback: str
    residual: float | None = None


class InfodynamicMetrics(BaseModel):
    entropy: float
    compression_delta: float
    novelty: float
    information_pressure: float


class WorldV2(BaseModel):
    id: str
    seed: int
    tick: int = Field(default=0, ge=0)
    width: int = Field(ge=2)
    height: int = Field(ge=2)
    agents: list[AgentV2]
    grid: list[list[CellV2]]
    discoveries: list[Discovery] = Field(default_factory=list)
    metrics: InfodynamicMetrics


class CreateWorldRequest(BaseModel):
    seed: int = 7
    width: int = Field(default=16, ge=2, le=100)
    height: int = Field(default=10, ge=2, le=100)
    agent_count: int = Field(default=6, ge=1, le=500)


class TickRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=500)


class ArbitrationRequest(BaseModel):
    hypothesis: str = Field(min_length=1)


class ArbitrationResult(BaseModel):
    valid: bool
    feedback: str
    residual: float | None = None

