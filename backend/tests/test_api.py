from pathlib import Path

from fastapi.testclient import TestClient

from living_agent_v2.api import app


def test_world_workflow(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LIVING_AGENT_DB_PATH", str(tmp_path / "api.db"))

    with TestClient(app) as client:
        health = client.get("/health")
        created = client.post(
            "/worlds", json={"world_id": "api-test", "seed": 11, "agent_count": 3}
        )
        ticked = client.post("/worlds/api-test/tick", json={"steps": 5})
        discoveries = client.get("/worlds/api-test/discoveries")

    assert health.status_code == 200
    assert health.json()["service"] == "living-agent-evolution-v2"
    assert created.status_code == 201
    assert ticked.status_code == 200
    assert ticked.json()["tick"] == 5
    assert len(discoveries.json()) == 1


def test_quantum_endpoint_discloses_unavailable_provider(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LIVING_AGENT_DB_PATH", str(tmp_path / "api.db"))

    with TestClient(app) as client:
        response = client.post(
            "/quantum/validate", json={"circuit": "OPENQASM 2.0;", "provider": "ibm"}
        )

    assert response.status_code == 503
    assert response.json()["detail"].startswith("Quantum validation is not configured")

