from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Genome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    curiosity: UnitFloat = 0.60
    compression_bias: UnitFloat = 0.55
    symbolic_precision: UnitFloat = 0.70
    sociality: UnitFloat = 0.50
    mutation_rate: UnitFloat = 0.05
    energy_efficiency: UnitFloat = 0.65


class AgentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    energy: Annotated[float, Field(ge=0.0)] = 100.0
    age: Annotated[int, Field(ge=0)] = 0
    genome: Genome = Field(default_factory=Genome)


class InfodynamicMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entropy: Annotated[float, Field(ge=0.0)]
    compression_delta: float
    novelty: UnitFloat
    information_pressure: Annotated[float, Field(ge=0.0)]


class Discovery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discovery_id: str = Field(default_factory=lambda: str(uuid4()))
    world_id: str
    tick: Annotated[int, Field(ge=0)]
    hypothesis: str
    valid: bool
    feedback: str
    residual: float | None = None
    created_at: datetime = Field(default_factory=utc_now)


class WorldState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: str
    seed: int
    tick: Annotated[int, Field(ge=0)] = 0
    agents: list[AgentState]
    metrics: InfodynamicMetrics
    accepted_discoveries: list[Discovery] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class CreateWorldRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: str | None = Field(default=None, min_length=1, max_length=64)
    seed: int = 1
    agent_count: Annotated[int, Field(ge=1, le=500)] = 12

    @field_validator("world_id")
    @classmethod
    def validate_world_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not all(character.isalnum() or character in "-_" for character in value):
            raise ValueError("world_id may contain only letters, digits, '-' and '_'")
        return value


class TickRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: Annotated[int, Field(ge=1, le=100)] = 1


class ArbitrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(min_length=1, max_length=1000)


class ArbitrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    feedback: str
    residual: float | None


class QuantumValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    circuit: str = Field(min_length=1, max_length=10_000)
    provider: Literal["ibm"] = "ibm"

