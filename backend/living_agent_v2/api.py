from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, status

from . import __version__
from .config import Settings, load_settings
from .discovery import arbitrate_hypothesis
from .engine import advance_world, create_world
from .models import (
    ArbitrationRequest,
    ArbitrationResult,
    CreateWorldRequest,
    Discovery,
    QuantumValidationRequest,
    TickRequest,
    WorldState,
)
from .store import SQLiteStore, WorldAlreadyExistsError, WorldNotFoundError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    store = SQLiteStore(settings.database_path, settings.embedding_dimensions)
    store.initialize()
    app.state.settings = settings
    app.state.store = store
    yield


app = FastAPI(
    title="Living Agent Evolution Sim v2",
    version=__version__,
    lifespan=lifespan,
)


def get_store(request: Request) -> SQLiteStore:
    return request.app.state.store


def not_found(error: WorldNotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail=f"world '{error.args[0]}' not found")


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "service": "living-agent-evolution-v2", "version": __version__}


@app.post("/worlds", response_model=WorldState, status_code=status.HTTP_201_CREATED)
def post_world(
    payload: CreateWorldRequest, store: SQLiteStore = Depends(get_store)
) -> WorldState:
    world = create_world(payload)
    try:
        return store.create_world(world)
    except WorldAlreadyExistsError as error:
        raise HTTPException(status_code=409, detail=f"world '{error.args[0]}' exists") from error


@app.get("/worlds/{world_id}", response_model=WorldState)
def get_world(world_id: str, store: SQLiteStore = Depends(get_store)) -> WorldState:
    try:
        return store.get_world(world_id)
    except WorldNotFoundError as error:
        raise not_found(error) from error


@app.post("/worlds/{world_id}/tick", response_model=WorldState)
def tick_world(
    world_id: str,
    payload: TickRequest,
    store: SQLiteStore = Depends(get_store),
) -> WorldState:
    try:
        world, version = store.get_world_with_version(world_id)
    except WorldNotFoundError as error:
        raise not_found(error) from error

    updated, discoveries = advance_world(world, payload.steps)
    try:
        store.commit_tick(updated, discoveries, version)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return updated


@app.get("/worlds/{world_id}/discoveries", response_model=list[Discovery])
def get_discoveries(
    world_id: str, store: SQLiteStore = Depends(get_store)
) -> list[Discovery]:
    try:
        return store.list_discoveries(world_id)
    except WorldNotFoundError as error:
        raise not_found(error) from error


@app.post("/hypotheses/arbitrate", response_model=ArbitrationResult)
def arbitrate(payload: ArbitrationRequest) -> ArbitrationResult:
    return arbitrate_hypothesis(payload.hypothesis)


@app.post("/quantum/validate")
def quantum_validate(payload: QuantumValidationRequest) -> None:
    del payload
    raise HTTPException(
        status_code=503,
        detail="Quantum validation is not configured in this backend slice.",
    )
