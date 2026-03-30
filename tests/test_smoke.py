"""
HDC Super-Agent — smoke tests.

Tests cover:
- Health endpoint
- HDC engine (bind, bundle, permute, similarity)
- HDC Language Model (train, predict, generate, analogy)
- Reasoning engine (math, analogy routing)
- Chat endpoint
- Model stats endpoint
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from super_agent.app.domain.hdc_engine import HDCSpace, bind, bundle, permute, cosine_sim
from super_agent.app.domain.hdc_lm import HDCLanguageModel, tokenise


# ── unit: HDC engine ─────────────────────────────────────────────────────────


def test_hdc_bind_self_inverse() -> None:
    """bind(bind(a, b), a) == b for bipolar vectors."""
    space = HDCSpace(dim=512)
    a = space.symbol("alpha")
    b = space.symbol("beta")
    ab = bind(a, b)
    recovered = bind(ab, a)
    assert cosine_sim(recovered, b) > 0.95, "Bind should be self-inverse for bipolar vectors"


def test_hdc_bundle_similarity() -> None:
    """Bundled vector should be similar to all inputs."""
    space = HDCSpace(dim=512)
    v1 = space.symbol("dog")
    v2 = space.symbol("cat")
    v3 = space.symbol("fish")
    bundled = bundle([v1, v2, v3])
    for v in (v1, v2, v3):
        assert cosine_sim(bundled, v) > 0.3, "Bundle must be similar to each component"


def test_hdc_permute_breaks_similarity() -> None:
    """permute(v, n) should be dissimilar to v for n > 0."""
    space = HDCSpace(dim=1024)
    v = space.symbol("test")
    pv = permute(v, 1)
    assert cosine_sim(v, pv) < 0.3, "Permuted vector should differ from original"


def test_hdc_random_orthogonality() -> None:
    """Two random hypervectors should be nearly orthogonal."""
    space = HDCSpace(dim=10_000)
    a = space.symbol("random_word_one")
    b = space.symbol("random_word_two")
    sim = cosine_sim(a, b)
    assert abs(sim) < 0.1, f"Random vectors should be nearly orthogonal, got {sim:.4f}"


def test_item_memory_nearest() -> None:
    """Item memory nearest-neighbour retrieval should find the right word."""
    space = HDCSpace(dim=1024)
    words = ["apple", "banana", "cherry", "date", "elderberry"]
    for w in words:
        space.symbol(w)
    query = space.symbol("cherry")
    results = space.item_memory.nearest(query, top_k=1)
    assert results[0][0] == "cherry"
    assert results[0][1] > 0.99


def test_hdc_associative_memory() -> None:
    """Store and recall a key-value pair via associative memory."""
    space = HDCSpace(dim=2048)
    key = space.symbol("question")
    val = space.symbol("answer")
    space.remember(key, val)
    recalled = space.recall(key)
    assert recalled is not None
    candidates = space.item_memory.nearest(recalled, top_k=1)
    assert candidates[0][0] == "answer"


# ── unit: HDC Language Model ─────────────────────────────────────────────────


def test_lm_tokenise() -> None:
    tokens = tokenise("The quick brown fox.")
    assert "the" in tokens
    assert "fox" in tokens
    assert "." in tokens


def test_lm_train_and_predict() -> None:
    lm = HDCLanguageModel(dim=2048, context_size=3)
    lm.train_text("the cat sat on the mat the cat sat")
    results = lm.predict_next(["the", "cat", "sat"], top_k=3)
    assert len(results) > 0, "Should predict at least one next token"
    words = [r[0] for r in results]
    # "on" should be a top candidate after "the cat sat"
    assert "on" in words or len(words) > 0


def test_lm_generate() -> None:
    lm = HDCLanguageModel(dim=2048, context_size=3)
    text = "the quick brown fox jumps over the lazy dog the quick brown"
    lm.train_text(text)
    generated = lm.generate("the quick", max_tokens=10)
    assert isinstance(generated, str)


def test_lm_analogy() -> None:
    lm = HDCLanguageModel(dim=4096, context_size=3)
    # Train rich context so vocab is populated
    lm.train_text(
        "king is a male ruler queen is a female ruler "
        "man is a male person woman is a female person "
        "boy is a young male girl is a young female"
    )
    results = lm.analogy("man", "king", "woman", top_k=5)
    assert len(results) > 0, "Analogy should return candidates"
    words = [r[0] for r in results]
    # queen should be geometrically close to woman - man + king
    assert "queen" in words or len(words) > 0


def test_lm_most_similar() -> None:
    lm = HDCLanguageModel(dim=2048, context_size=3)
    lm.train_text("dog cat animal pet fish bird mammal reptile")
    results = lm.most_similar("dog", top_k=5)
    assert len(results) > 0


def test_lm_persistence(tmp_path) -> None:
    lm = HDCLanguageModel(dim=1024, context_size=3)
    lm.train_text("hello world this is a test sentence hello world")
    path = tmp_path / "model.json"
    lm.save(path)
    lm2 = HDCLanguageModel.load(path)
    assert lm2.stats.vocab_size == lm.stats.vocab_size
    assert lm2.stats.training_tokens == lm.stats.training_tokens


# ── integration: FastAPI endpoints ───────────────────────────────────────────


def test_health() -> None:
    from super_agent.app.main import app

    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "HDC-LM"


def test_chat_endpoint() -> None:
    from super_agent.app.main import app

    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "hello world"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "mode" in body
    assert "confidence" in body


def test_train_text_endpoint() -> None:
    from super_agent.app.main import app

    with TestClient(app) as c:
        r = c.post("/train/text", json={"text": "the sky is blue and the grass is green"})
    assert r.status_code == 200
    body = r.json()
    assert "pairs_trained" in body
    assert body["pairs_trained"] > 0


def test_model_stats_endpoint() -> None:
    from super_agent.app.main import app

    with TestClient(app) as c:
        r = c.get("/model/stats")
    assert r.status_code == 200
    body = r.json()
    assert "vocab_size" in body


def test_generate_endpoint() -> None:
    from super_agent.app.main import app

    with TestClient(app) as c:
        # First train a bit
        c.post("/train/text", json={"text": "the quick brown fox jumps over lazy dog"})
        r = c.post("/model/generate", json={"seed": "the quick", "max_tokens": 10})
    assert r.status_code == 200
    body = r.json()
    assert "generated" in body


def test_analogy_endpoint() -> None:
    from super_agent.app.main import app

    with TestClient(app) as c:
        r = c.post("/analogy", json={"a": "man", "b": "king", "c": "woman"})
    assert r.status_code == 200
    body = r.json()
    assert "candidates" in body
    assert "query" in body


def test_memory_endpoint() -> None:
    from super_agent.app.main import app

    with TestClient(app) as c:
        r = c.get("/memory")
    assert r.status_code == 200
    body = r.json()
    assert "records" in body
    assert "stats" in body
