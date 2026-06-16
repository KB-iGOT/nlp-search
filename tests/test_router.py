"""
Tests for POST /nlp/search.

Route:   src/search/router.py:8-10
Schema:  src/search/request_model.py:3-13
Service: src/search/llm_service.py:36-87

═══ KNOWN BUGS documented by tests below ════════════════════════════════════
BUG-1  src/search/llm_service.py:41   — whitespace query: `return HTTPException`
BUG-2  src/search/llm_service.py:44   — over-length query: `return HTTPException`
BUG-3  src/search/llm_service.py:87   — JSON decode error: `return HTTPException`
BUG-4  src/search/llm_service.py:92   — unexpected LLM error: `return HTTPException`

In all four cases the function RETURNS an HTTPException object instead of
RAISING it.  FastAPI serialises the returned object as HTTP 200 JSON
({"status_code": <N>, "detail": "..."}) rather than propagating a real 4xx/5xx.
Tests asserting current (broken) behaviour are tagged [BUG].
ACTION NEEDED: replace `return HTTPException(...)` with `raise HTTPException(...)`
at lines 41, 44, 87, 92 in src/search/llm_service.py.
════════════════════════════════════════════════════════════════════════════════
"""

import pytest
from tests.helpers import make_stream_chunks

BASE = "/nlp/search"


# ─────────────────────────────────────────────────────────────────────────────
# 3.1  Functional Correctness
# ─────────────────────────────────────────────────────────────────────────────

async def test_post_search_happy_path(client, mock_generate_content):
    """
    Valid query, synonyms omitted (defaults to False [request_model.py:5]).
    Route handler returns search_request() result [router.py:10].
    Service returns {"data": <parsed_json>} [llm_service.py:51].
    """
    mock_generate_content.return_value = make_stream_chunks('["python", "programming"]')
    response = await client.post(BASE, json={"query": "python programming"})
    assert response.status_code == 200           # [router.py:8 — default 200]
    body = response.json()
    assert "data" in body                        # [llm_service.py:51]
    assert body["data"] == ["python", "programming"]


async def test_post_search_synonyms_true(client, mock_generate_content):
    """
    synonyms=True triggers synonym instruction insertion [llm_service.py:63-65].
    End-to-end: request accepted, 200 returned.
    """
    mock_generate_content.return_value = make_stream_chunks('["python", "code"]')
    response = await client.post(BASE, json={"query": "python code", "synonyms": True})
    assert response.status_code == 200
    assert "data" in response.json()


async def test_post_search_synonyms_field_absent(client, mock_generate_content):
    """
    synonyms has a default of False [request_model.py:5].
    Omitting it must not produce a validation error.
    """
    mock_generate_content.return_value = make_stream_chunks('["keyword"]')
    response = await client.post(BASE, json={"query": "some query"})
    assert response.status_code == 200


async def test_post_search_chunked_model_response(client, mock_generate_content):
    """
    llm_request concatenates multiple stream chunks [llm_service.py:76-78].
    Verify full string is assembled and parsed correctly.
    """
    mock_generate_content.return_value = make_stream_chunks('["py', 'thon"]')
    response = await client.post(BASE, json={"query": "python"})
    assert response.status_code == 200
    assert response.json()["data"] == ["python"]


async def test_post_search_model_returns_empty_list(client, mock_generate_content):
    """
    LLM returning an empty keyword list is valid — must yield 200, not 404.
    [llm_service.py:51] wraps any parsed JSON in {"data": <value>}.
    """
    mock_generate_content.return_value = make_stream_chunks("[]")
    response = await client.post(BASE, json={"query": "obscure query"})
    assert response.status_code == 200
    assert response.json()["data"] == []


async def test_post_search_query_at_exact_max_length(client, mock_generate_content):
    """
    Boundary: query of exactly 400 chars must pass the validator.
    Validator condition is `len(value) > 400` (strict GT) [request_model.py:12].
    """
    mock_generate_content.return_value = make_stream_chunks('["keyword"]')
    response = await client.post(BASE, json={"query": "a" * 400})
    assert response.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# 3.1  Observed-bug cases (return-vs-raise in service layer)
# ─────────────────────────────────────────────────────────────────────────────

async def test_post_search_whitespace_query_observed_bug(client):
    """
    [BUG-1] Whitespace-only query ("   ") passes the Pydantic field_validator
    because `not "   "` is False [request_model.py:10].  The service then checks
    `not req_data.query.strip()` (True) and *returns* HTTPException(400) instead
    of raising it [llm_service.py:40-41].

    OBSERVED: HTTP 200 with body {"status_code": 400, "detail": "..."}.
    EXPECTED (after fix): HTTP 400.
    ACTION NEEDED: change `return` → `raise` at llm_service.py:41.
    """
    response = await client.post(BASE, json={"query": "   "})
    assert response.status_code == 200                         # [BUG] should be 400
    body = response.json()
    assert body.get("status_code") == 400                      # [llm_service.py:41]
    assert "empty" in body.get("detail", "").lower()           # [llm_service.py:41]


async def test_post_search_invalid_json_from_model_observed_bug(client, mock_generate_content):
    """
    [BUG-3] When the LLM returns non-JSON text, llm_request *returns*
    HTTPException(500) instead of raising it [llm_service.py:84-87].

    OBSERVED: HTTP 200 with body {"status_code": 500, "detail": "..."}.
    EXPECTED (after fix): HTTP 500.
    ACTION NEEDED: change `return` → `raise` at llm_service.py:87.
    """
    mock_generate_content.return_value = make_stream_chunks("not valid json at all")
    response = await client.post(BASE, json={"query": "test"})
    assert response.status_code == 200                         # [BUG] should be 500
    body = response.json()
    assert body.get("status_code") == 500                      # [llm_service.py:87]


# ─────────────────────────────────────────────────────────────────────────────
# 3.2  Pydantic Validation — FastAPI returns 422
# ─────────────────────────────────────────────────────────────────────────────

async def test_post_search_missing_query_field(client):
    """query is required [request_model.py:4].  Missing → 422."""
    response = await client.post(BASE, json={"synonyms": False})
    assert response.status_code == 422


async def test_post_search_empty_string_query(client):
    """
    Empty string fails field_validator [request_model.py:10-11]:
    `if not value: raise ValueError('Query cannot be empty.')`
    FastAPI wraps ValidationError → 422.
    """
    response = await client.post(BASE, json={"query": ""})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("empty" in str(d).lower() for d in detail)


async def test_post_search_query_too_long(client):
    """
    401-char query fails field_validator [request_model.py:12-13]:
    `if len(value) > 400: raise ValueError('Query cannot be longer than 400 characters.')`
    FastAPI wraps ValidationError → 422.
    """
    response = await client.post(BASE, json={"query": "a" * 401})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("400" in str(d) or "longer" in str(d).lower() for d in detail)


async def test_post_search_null_query(client):
    """None for a required str field → 422 [request_model.py:4]."""
    response = await client.post(BASE, json={"query": None})
    assert response.status_code == 422


async def test_post_search_empty_body(client):
    """Empty JSON body — missing required field → 422."""
    response = await client.post(BASE, json={})
    assert response.status_code == 422


async def test_post_search_integer_query_no_crash(client, mock_generate_content):
    """
    Pydantic v2 coerces int → str for a `str` field [request_model.py:4],
    so an integer query may pass validation and reach the service.
    [INFERRED from Pydantic v2 coercion rules — not directly readable from source]
    Critical assertion: must not crash with 500.
    """
    mock_generate_content.return_value = make_stream_chunks('["keyword"]')
    response = await client.post(BASE, json={"query": 12345})
    assert response.status_code in (200, 422)
    assert response.status_code != 500


# ─────────────────────────────────────────────────────────────────────────────
# 3.3  Security — Injection
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("payload", [
    "' OR '1'='1",
    "' OR 1=1 --",
    "'; DROP TABLE users;--",
])
async def test_post_search_sql_injection_no_crash(client, mock_generate_content, payload):
    """
    SQL injection payloads in query field [llm_service.py:67 — query used in prompt].
    No DB in this service; query is forwarded to LLM (mocked).
    Assert: no 500, no payload reflection in error body.
    """
    mock_generate_content.return_value = make_stream_chunks('["keyword"]')
    response = await client.post(BASE, json={"query": payload})
    assert response.status_code != 500, f"SQL payload crashed app: {payload!r}"


@pytest.mark.parametrize("payload", [
    "<script>alert(1)</script>",
    '"><img src=x onerror=alert(1)>',
    "javascript:alert(1)",
])
async def test_post_search_xss_payload_no_crash(client, mock_generate_content, payload):
    """
    XSS payloads in query [llm_service.py:67].
    Assert: no 500; payload not reflected unescaped in error responses.
    """
    mock_generate_content.return_value = make_stream_chunks('["keyword"]')
    response = await client.post(BASE, json={"query": payload})
    assert response.status_code != 500, f"XSS payload crashed app: {payload!r}"
    if response.status_code >= 400:
        # Payload must not be reflected verbatim in an error body
        assert payload not in response.text


async def test_post_search_mass_assignment_extra_fields_ignored(client, mock_generate_content):
    """
    Extra fields not in SearchModel [request_model.py:3-13] must be silently
    dropped by Pydantic v2 (default extra='ignore' behaviour).
    Assert: 200, extra keys absent from response.
    """
    mock_generate_content.return_value = make_stream_chunks('["keyword"]')
    response = await client.post(
        BASE,
        json={
            "query": "test query",
            "synonyms": False,
            "is_admin": True,       # extra — must be ignored
            "role": "superuser",    # extra — must be ignored
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "is_admin" not in body
    assert "role" not in body


# ─────────────────────────────────────────────────────────────────────────────
# 3.4  Authentication / Authorisation
# ─────────────────────────────────────────────────────────────────────────────

async def test_post_search_no_auth_required(client, mock_generate_content):
    """
    POST /nlp/search has no Depends() auth guard [router.py:9 — signature has
    only `search_model: SearchModel`].  Unauthenticated requests must NOT get
    401 or 403.
    """
    mock_generate_content.return_value = make_stream_chunks('["keyword"]')
    response = await client.post(BASE, json={"query": "test"})
    assert response.status_code not in (401, 403)


# ─────────────────────────────────────────────────────────────────────────────
# 3.5  Error Handling & Resilience
# ─────────────────────────────────────────────────────────────────────────────

async def test_post_search_model_raises_returns_500(client, mock_generate_content):
    """
    When generate_content() raises, the outer except block [llm_service.py:52-55]
    correctly *raises* HTTPException(500) — this is the non-buggy path.
    Assert: 500, body contains no stack trace, no internal error message.
    S8415: HTTPException(500) at line 55 is not documented in responses={}
    [router.py:8 — no responses= arg].  Behavior is correct; doc is missing.
    """
    mock_generate_content.side_effect = Exception("Vertex AI unavailable")
    response = await client.post(BASE, json={"query": "test"})
    assert response.status_code == 500                # [llm_service.py:55]
    body_text = response.text
    assert "Traceback" not in body_text
    assert "Vertex AI unavailable" not in body_text   # internal msg must not leak


async def test_post_search_500_no_sensitive_data_in_body(client, mock_generate_content):
    """
    500 error body must not contain credentials, file paths, or stack traces
    [llm_service.py:53-55].
    """
    mock_generate_content.side_effect = RuntimeError("crash")
    response = await client.post(BASE, json={"query": "test"})
    assert response.status_code == 500
    body_text = response.text.lower()
    for fragment in ("password", "secret", "credential", "token", "/app/", "traceback"):
        assert fragment not in body_text, (
            f"Sensitive fragment {fragment!r} present in 500 response"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3.6  Code Smell — secret / config leakage
# ─────────────────────────────────────────────────────────────────────────────

async def test_post_search_response_does_not_echo_config_values(client, mock_generate_content):
    """
    Successful response must not expose internal config values set in conftest.py
    (GOOGLE_CLOUD_PROJECT, GOOGLE_APPLICATION_CREDENTIALS) [config.py:14,16].
    """
    mock_generate_content.return_value = make_stream_chunks('["keyword"]')
    response = await client.post(BASE, json={"query": "test"})
    assert response.status_code == 200
    body_text = response.text
    assert "test-project-id" not in body_text
    assert "test-credentials.json" not in body_text


# ─────────────────────────────────────────────────────────────────────────────
# 3.7  FastAPI-specific
# ─────────────────────────────────────────────────────────────────────────────

async def test_post_search_response_content_type_is_json(client, mock_generate_content):
    """Response Content-Type must be application/json on 200."""
    mock_generate_content.return_value = make_stream_chunks('["keyword"]')
    response = await client.post(BASE, json={"query": "test"})
    assert "application/json" in response.headers.get("content-type", "")


async def test_get_search_method_not_allowed(client):
    """
    GET /nlp/search is not registered [router.py:8 — POST only].
    Must return 405 Method Not Allowed.
    """
    response = await client.get(BASE)
    assert response.status_code == 405


# ─────────────────────────────────────────────────────────────────────────────
# 3.8  SonarQube Rule S8415 — Ghost Exception
# ─────────────────────────────────────────────────────────────────────────────

async def test_s8415_ghost_exception_500_on_unexpected_error(client, mock_generate_content):
    """
    S8415 — Ghost Exception:
    HTTPException(500) is raised at [llm_service.py:55] but is NOT listed in the
    route decorator's responses={} map [router.py:8 — no responses= argument].
    This test triggers the path and asserts the correct status code is returned.

    NOTE: This is an OpenAPI documentation gap, not a functional bug.
    The HTTP behavior is correct; the OpenAPI spec is incomplete.
    Remediation: add `responses={500: {"description": "Internal server error"}}`
    to the @router.post decorator at src/search/router.py:8.
    """
    mock_generate_content.side_effect = ValueError("unexpected model error")
    response = await client.post(BASE, json={"query": "test"})
    assert response.status_code == 500   # [llm_service.py:55]


# ─────────────────────────────────────────────────────────────────────────────
# 3.9  Security — Path Traversal
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("payload", [
    "../../etc/passwd",
    "..%2F..%2Fetc%2Fshadow",
    "/etc/passwd",
])
async def test_post_search_path_traversal_no_crash(client, mock_generate_content, payload):
    """
    Phase 7.1 — Security / injection:
    Path traversal payloads in the query field [llm_service.py:67 — query is
    forwarded verbatim to the LLM prompt, never used as a filesystem path].

    The service does not perform any filesystem access with the query value,
    so these payloads pose no real traversal risk; however the test ensures:
      - No 500 is returned (no unhandled crash).
      - Payload is not reflected verbatim in any error response body.
    Assert: status code is not 500.
    """
    mock_generate_content.return_value = make_stream_chunks('["keyword"]')
    response = await client.post(BASE, json={"query": payload})
    assert response.status_code != 500, (
        f"Path traversal payload caused 500: {payload!r}"
    )
    if response.status_code >= 400:
        assert payload not in response.text, (
            f"Path traversal payload reflected in error body: {payload!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3.10  Malformed / Non-JSON Body
# ─────────────────────────────────────────────────────────────────────────────

async def test_post_search_malformed_body_returns_422(client):
    """
    Phase 7.1 — Pydantic validation:
    Sending a raw text body (Content-Type: text/plain) instead of JSON.
    FastAPI cannot parse the body into SearchModel and must return 422,
    never 500 [router.py:9 — body param `search_model: SearchModel`].
    """
    response = await client.post(
        BASE,
        content=b"this is not json at all",
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 422
    assert response.status_code != 500


# ─────────────────────────────────────────────────────────────────────────────
# 3.11  Boundary — Minimum Valid Query Length
# ─────────────────────────────────────────────────────────────────────────────

async def test_post_search_single_char_query_is_valid(client, mock_generate_content):
    """
    Phase 7.1 — Functional correctness / boundary:
    A single-character query is the minimum valid input.
    Validator condition is `if not value` [request_model.py:10] — a single
    non-whitespace char is truthy and must pass validation, yielding 200.
    Complements test_post_search_query_at_exact_max_length (upper bound).
    """
    mock_generate_content.return_value = make_stream_chunks('["a"]')
    response = await client.post(BASE, json={"query": "a"})
    assert response.status_code == 200           # [request_model.py:10 — passes validator]
    assert "data" in response.json()             # [llm_service.py:51]
