import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from src import main
from src.search import llm_service

KEYWORDS = {"keywords": [{"keyword": "data mining", "priority": 1}]}


@pytest.fixture
def model(monkeypatch):
    """Stands in for client.aio.models.generate_content."""
    answer = SimpleNamespace(text=json.dumps(KEYWORDS), candidates=[], prompt_feedback=None)
    generate = AsyncMock(return_value=answer)
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(llm_service, "client", client)
    return generate


@pytest.fixture
def lifecycle(monkeypatch):
    """Records what the app does at startup and shutdown, without Redis."""
    calls = SimpleNamespace(
        init_cache=AsyncMock(),
        close_cache=AsyncMock(),
        drain=AsyncMock(),
    )
    monkeypatch.setattr(main.redis_service, "init_cache", calls.init_cache)
    monkeypatch.setattr(main.redis_service, "close_cache", calls.close_cache)
    monkeypatch.setattr(main, "drain_background_tasks", calls.drain)
    return calls


@pytest.fixture
def api(lifecycle, model, monkeypatch):
    # No cache during these tests: every search goes to the (fake) model.
    monkeypatch.setattr(main.redis_service, "lookup", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(main.redis_service, "store", AsyncMock())
    with TestClient(main.app) as client:
        yield client


class TestLifespan:
    def test_connects_the_cache_at_startup(self, lifecycle):
        with TestClient(main.app):
            lifecycle.init_cache.assert_awaited_once_with(main.get_settings())
            lifecycle.drain.assert_not_awaited()
            lifecycle.close_cache.assert_not_awaited()

    def test_drains_before_closing_the_cache_at_shutdown(self, lifecycle):
        order = []
        lifecycle.drain.side_effect = lambda: order.append("drain")
        lifecycle.close_cache.side_effect = lambda: order.append("close_cache")

        with TestClient(main.app):
            pass

        assert order == ["drain", "close_cache"]


class TestSearchEndpoint:
    def test_returns_keywords(self, api, model):
        response = api.post("/nlp/search", json={"query": "data mining"})

        assert response.status_code == 200
        assert response.json() == {"data": KEYWORDS}
        model.assert_awaited_once()

    def test_passes_synonyms_through(self, api, model):
        api.post("/nlp/search", json={"query": "data mining", "synonyms": True})

        config = model.await_args.kwargs["config"]
        assert config.response_schema == llm_service.build_keywords_schema(True)

    def test_blank_query_is_400(self, api, model):
        response = api.post("/nlp/search", json={"query": "   "})

        assert response.status_code == 400
        assert response.json() == {"detail": "Query cannot be empty."}
        model.assert_not_awaited()

    @pytest.mark.parametrize("body", [{"query": ""}, {"query": "x" * 401}, {}])
    def test_invalid_body_is_422(self, api, model, body):
        assert api.post("/nlp/search", json=body).status_code == 422
        model.assert_not_awaited()

    def test_model_error_status_reaches_the_client(self, api, model, monkeypatch):
        monkeypatch.setattr(llm_service.settings, "LLM_MAX_RETRIES", 0)
        model.side_effect = genai_errors.APIError(429, {"error": {"code": 429, "message": "rate limited"}})

        response = api.post("/nlp/search", json={"query": "data mining"})

        assert response.status_code == 429


def test_welcome(api):
    assert api.get("/").json() == {"message": "Welcome to NLP Search Service!"}
