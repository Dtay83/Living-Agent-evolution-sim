# Living Agent Evolution Sim v2 Backend

This is a clean backend replacement for the prototype backend slice. It is
designed to stay lightweight on Windows and stable before dashboard wiring.

## Architecture

1. **Simulation Core**: deterministic tick engine that owns world state.
2. **Infodynamic Physics Engine**: world evolution is scored by information
   pressure, compression delta, entropy, and novelty.
3. **Agent Genome v2**: JSON DNA with mutation, inheritance, and cognitive bias
   fields.
4. **Memory System**: deterministic hash embeddings with cosine search. This is
   a bootstrap layer that can later be replaced with Qdrant and semantic models.
5. **Scientific Discovery Engine**: agents produce hypotheses that must pass
   symbolic validation before becoming accepted discoveries.
6. **Distributed Runtime**: intentionally deferred until the single-process core
   is stable.
7. **Dashboard Integration**: expose API endpoints first; wire React only after
   tests and API behavior are stable.

## Install

```bat
cd backend
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

If Python 3.12 is unavailable, use Python 3.11. Avoid Python 3.14 for now.

## Test

```bat
python -m pytest
```

## Run

```bat
python -m uvicorn living_agent_v2.api:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000/health
```

## API

- `GET /health`
- `POST /worlds`
- `GET /worlds/{world_id}`
- `POST /worlds/{world_id}/tick`
- `POST /hypotheses/arbitrate`

## Notes

- This backend is intentionally single-process.
- The memory embedding is deterministic, local, and dependency-light.
- Accepted discoveries must pass symbolic arbitration.
- No IBM Quantum, Ray, Temporal, Redis, Neo4j, or Qdrant production claims are
  made in this slice.

