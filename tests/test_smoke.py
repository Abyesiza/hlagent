from __future__ import annotations

import time
#hese are all tests 
import numpy as np
from fastapi.testclient import TestClient

from super_agent.app.domain.hdc import HDCSpace, associate_task_solution
from super_agent.app.domain.math_schemas import SymCodeRequest
from super_agent.app.domain.quantum_inspired import pick_best_candidate_index
from super_agent.app.infrastructure.sympy_runner import run_symcode
from super_agent.app.main import app


def test_health() -> None:
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_blueprint_status() -> None:
    with TestClient(app) as c:
        r = c.get("/api/v1/status")
    assert r.status_code == 200
    data = r.json()
    assert "items" in data
    assert len(data["items"]) >= 5


def test_sympy_rejects_hallucinated_api_symbol() -> None:
    src = """
import sympy as sp
result = sp.Symbol('API_RESOURCE_EXHAUSTED_ERROR_RECEIVED_INSTEAD_OF_PROBLEM')
"""
    res = run_symcode(SymCodeRequest(source=src))
    assert not res.ok
    assert res.error and "API" in res.error


def test_sympy_integral() -> None:
    src = """
import sympy as sp
x = sp.Symbol("x", real=True)
result = sp.integrate(x**3 * sp.sin(x), x)
"""
    res = run_symcode(SymCodeRequest(source=src))
    assert res.ok
    assert "result" in res.result_repr.lower() or "sin" in res.result_repr


def test_hdc_bind() -> None:
    s = HDCSpace(dim=256)
    v = associate_task_solution("task-a", "sol-a", s)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_quantum_pick() -> None:
    assert pick_best_candidate_index([1.0, 0.5, 0.2]) in (0, 1, 2)


def test_chat_pipeline_neural_no_api_key() -> None:
    with TestClient(app) as c:
        r = c.post("/api/v1/chat", json={"message": "What is 2+2 in one sentence?"})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "neural"
    assert "gemini disabled" in body["answer"].lower() or len(body["answer"]) > 0


def test_chat_pipeline_symbolic_no_api_key() -> None:
    with TestClient(app) as c:
        r = c.post("/api/v1/chat", json={"message": "Integrate x**2 with respect to x"})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "error"
    assert "GEMINI_API_KEY" in body["answer"] or "sympy" in body["answer"].lower()


def test_agent_job_returns_turn() -> None:
    with TestClient(app) as c:
        start = c.post("/api/v1/agent/start", json={"prompt": "Say hi"})
        assert start.status_code == 200
        jid = start.json()["job_id"]
        result = None
        for _ in range(200):
            job = c.get(f"/api/v1/agent/jobs/{jid}")
            assert job.status_code == 200
            data = job.json()
            result = data.get("result")
            if result is not None:
                break
            time.sleep(0.02)
    assert result is not None
    assert result["route"] == "neural"
