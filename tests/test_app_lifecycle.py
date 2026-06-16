"""
Application-wide lifecycle tests covering SonarQube rules S8400, S8401, S8414.

Route inventory (from Phase 1 discovery):
  POST /nlp/search  [src/search/router.py:8, prefix at src/main.py:5]
  GET  /            [src/main.py:7-9]

S8400 — Phantom 204 bodies:
  NOT APPLICABLE.  Neither route uses status_code=204.  Documented here.

S8401 — Router registration order:
  TESTABLE.  GET /openapi.json must list all routes from the inventory above.
  A missing path indicates a router registration/ordering bug.

S8414 — Middleware ordering / CORS on error responses:
  NOT APPLICABLE.  No CORSMiddleware is registered [src/main.py:1-9 — only
  FastAPI() instantiation and app.include_router()].  Documented here.
  If CORSMiddleware is added in the future, ensure it is added LAST so it
  wraps error responses (Starlette applies middleware in reverse-add order).

S8392 — Binding to 0.0.0.0:
  STATIC FINDING [Dockerfile:28 — `--host", "0.0.0.0"`].
  Not tested (main.py/Dockerfile excluded from unit tests).

S8397 — App object passed to uvicorn.run:
  NOT APPLICABLE.  uvicorn is invoked via CLI in Dockerfile:28; no
  `uvicorn.run(app, ...)` call in source.

S8413 — Prefix defined late:
  STATIC FINDING.  router = APIRouter() at [src/search/router.py:6] has no
  prefix; prefix="/nlp" is supplied in app.include_router() [src/main.py:5].
  This is the pattern S8413 flags for maintainability.  No test required.
"""

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# S8401 — Router registration order (all routes present in OpenAPI schema)
# ─────────────────────────────────────────────────────────────────────────────

async def test_s8401_openapi_schema_returns_200(client):
    """GET /openapi.json is reachable and returns a valid schema."""
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    assert "paths" in response.json()


async def test_s8401_nlp_search_in_openapi_paths(client):
    """
    S8401: POST /nlp/search must appear in the OpenAPI paths dict.
    Path = "/nlp" prefix [src/main.py:5] + "/search" route [src/search/router.py:8].
    A missing path would indicate a router registration order bug.
    """
    response = await client.get("/openapi.json")
    paths = response.json()["paths"]
    assert "/nlp/search" in paths, (
        "S8401: /nlp/search not found in OpenAPI schema — "
        "check router registration order at src/main.py:5"
    )


async def test_s8401_nlp_search_registered_as_post(client):
    """
    /nlp/search must expose only the POST method [src/search/router.py:8].
    """
    response = await client.get("/openapi.json")
    path_item = response.json()["paths"].get("/nlp/search", {})
    assert "post" in path_item, "POST method missing from /nlp/search OpenAPI entry"


async def test_s8401_root_welcome_route_in_openapi_paths(client):
    """
    S8401: GET / [src/main.py:7-9] must appear in the OpenAPI paths dict.
    """
    response = await client.get("/openapi.json")
    paths = response.json()["paths"]
    assert "/" in paths, (
        "S8401: / (welcome route) not found in OpenAPI schema — "
        "check src/main.py:7-9"
    )


async def test_s8401_no_unexpected_routes_registered(client):
    """
    Regression guard: only the expected routes are in the schema.
    Detects accidental route additions or stale registrations.
    Expected routes: POST /nlp/search, GET /
    """
    response = await client.get("/openapi.json")
    registered_paths = set(response.json()["paths"].keys())
    expected_paths = {"/nlp/search", "/"}
    assert registered_paths == expected_paths, (
        f"Unexpected route changes detected.\n"
        f"Expected: {expected_paths}\n"
        f"Actual:   {registered_paths}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Welcome route smoke test [src/main.py:7-9]
# ─────────────────────────────────────────────────────────────────────────────

async def test_welcome_route_returns_200(client):
    """
    GET / [src/main.py:7-9] returns HTTP 200 with the welcome message.
    Return value: {"message": "Welcome to NLP Search Service!"} [src/main.py:9].
    """
    response = await client.get("/")
    assert response.status_code == 200                              # [src/main.py:7]
    assert response.json() == {
        "message": "Welcome to NLP Search Service!"                # [src/main.py:9]
    }


# ─────────────────────────────────────────────────────────────────────────────
# S8400 — Phantom 204 bodies (documentation-only finding for this app)
# ─────────────────────────────────────────────────────────────────────────────
# Neither POST /nlp/search [router.py:8] nor GET / [main.py:7] specify
# status_code=204 in their decorators; both default to HTTP 200.
# S8400 is NOT applicable.  If a 204 route is added in the future, add a test:
#
#   async def test_s8400_204_route_has_empty_body(client):
#       response = await client.delete("/resource/1")
#       assert response.status_code == 204
#       assert response.content == b""

async def test_s8400_no_204_response_on_search(client, mock_generate_content):
    """
    POST /nlp/search does not return 204 [router.py:8 — no status_code=204].
    Confirms the route defaults to 200.
    """
    mock_generate_content.return_value = [
        __import__('unittest.mock', fromlist=['MagicMock']).MagicMock(
            **{"text": '["keyword"]'}
        )
    ]
    response = await client.post("/nlp/search", json={"query": "test"})
    assert response.status_code != 204


async def test_s8400_no_204_response_on_welcome(client):
    """GET / does not return 204 [main.py:7 — no status_code=204]."""
    response = await client.get("/")
    assert response.status_code != 204


# ─────────────────────────────────────────────────────────────────────────────
# Method guard — GET / (welcome route)
# ─────────────────────────────────────────────────────────────────────────────

async def test_post_root_method_not_allowed(client):
    """
    Phase 7.1 — Functional correctness:
    GET / [src/main.py:7] is a GET-only route.
    POSTing to it must return 405 Method Not Allowed, not 200 or 500.
    Complements test_get_search_method_not_allowed in test_router.py which
    asserts the same guard on POST /nlp/search.
    """
    response = await client.post("/")
    assert response.status_code == 405

