from fastapi import FastAPI, HTTPException

from .discovery import arbitrate_hypothesis
from .models import ArbitrationRequest, ArbitrationResult, CreateWorldRequest, TickRequest, WorldV2
from .store import WorldStore


app = FastAPI(title="Living Agent Evolution Sim v2 API", version="0.2.0")
store = WorldStore()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "living-agent-evolution-v2"}


@app.post("/worlds", response_model=WorldV2)
def create_world(request: CreateWorldRequest) -> WorldV2:
    return store.create(request)


@app.get("/worlds/{world_id}", response_model=WorldV2)
def get_world(world_id: str) -> WorldV2:
    world = store.get(world_id)
    if world is None:
        raise HTTPException(status_code=404, detail="World not found")
    return world


@app.post("/worlds/{world_id}/tick", response_model=WorldV2)
def tick_world(world_id: str, request: TickRequest) -> WorldV2:
    world = store.tick(world_id, request.steps)
    if world is None:
        raise HTTPException(status_code=404, detail="World not found")
    return world


@app.post("/hypotheses/arbitrate", response_model=ArbitrationResult)
def arbitrate(request: ArbitrationRequest) -> ArbitrationResult:
    return arbitrate_hypothesis(request.hypothesis)

