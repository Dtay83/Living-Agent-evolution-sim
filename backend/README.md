# Living Agent Evolution Sim v2 Backend

A lightweight, deterministic backend for the Living Agent Evolution Sim. It
provides local world evolution, infodynamic metrics, symbolic hypothesis
arbitration, persistent SQLite state, and deterministic vector memory without
Torch, Hugging Face downloads, Qdrant, or external model calls.

## Production boundary

This slice is a local simulation foundation. It does not execute on quantum
hardware, call an LLM, or provide distributed orchestration. Those capabilities
must remain disabled until credentials, quotas, timeout handling, observability,
and truthful provider-status reporting are implemented.

## Install and run

Use Python 3.11 or 3.12. On Windows, from this `backend` directory:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest
python -m uvicorn living_agent_v2.api:app --reload --port 8000
```

The database defaults to `living_agent_v2.db`. Set `LIVING_AGENT_DB_PATH` to
use another path.

## API

- `GET /health`
- `POST /worlds`
- `GET /worlds/{world_id}`
- `POST /worlds/{world_id}/tick`
- `GET /worlds/{world_id}/discoveries`
- `POST /hypotheses/arbitrate`
- `POST /quantum/validate` (always returns `503` in this slice)

