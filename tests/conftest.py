"""
Shared fixtures for the nlp-search test suite.

CRITICAL — Module-level side effects in src/search/llm_service.py:
  Line 18  : settings = get_settings()   → reads env vars via pydantic-settings
  Line 19-20: sets GOOGLE_APPLICATION_CREDENTIALS in os.environ
  Line 21  : vertexai.init(...)           → real GCP network call
  Line 22-34: model = GenerativeModel(...) → real GCP API call

To prevent credential errors and real API calls during tests, this file:
  1. Sets required env vars via os.environ BEFORE the app is imported.
  2. Stubs sys.modules["vertexai"] and sys.modules["vertexai.generative_models"]
     BEFORE any test module triggers the import of src.search.llm_service.

Both steps must happen at MODULE LEVEL in this file (not inside fixtures),
because conftest.py is executed before any test module is imported.
"""

import os
import sys
from unittest.mock import MagicMock, patch

# ── 1. Required env vars ──────────────────────────────────────────────────────
# Set before pydantic-settings instantiates Settings() [src/config.py:5-26].
# Required fields (no defaults): GOOGLE_CLOUD_PROJECT [line 14],
#   GOOGLE_APPLICATION_CREDENTIALS [line 16],
#   NLP_SEARCH_INSTRUCTION_PROMPT [line 25], NPL_SEARCH_EXAMPLE_PROMPT [line 26].
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project-id"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/test-credentials.json"
os.environ["NLP_SEARCH_INSTRUCTION_PROMPT"] = "Extract search keywords from: "
os.environ["NPL_SEARCH_EXAMPLE_PROMPT"] = ' Output as JSON array: ["keyword1"]'
os.environ["MODEL_NAME"] = "test-model"
os.environ["MAX_OUTPUT_TOKENS"] = "8192"
os.environ["TEMPERATURE"] = "0"
os.environ["TOP_P"] = "0.95"
os.environ["TOP_K"] = "1"
os.environ["MAX_SEARCH_LEN"] = "400"

# ── 2. Stub vertexai before llm_service imports it ────────────────────────────
# src/search/llm_service.py lines 5-6 import:
#   import vertexai
#   from vertexai.generative_models import GenerativeModel
# Lines 21-34 call vertexai.init() and GenerativeModel() at module level.
# Replacing these in sys.modules prevents real API calls and missing-creds errors.
_vertexai_stub = MagicMock(name="vertexai_stub")
_generative_models_stub = MagicMock(name="vertexai.generative_models_stub")
sys.modules["vertexai"] = _vertexai_stub
sys.modules["vertexai.generative_models"] = _generative_models_stub

# ── 3. App import — safe now that stubs are in place ──────────────────────────
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from src.main import app  # [src/main.py:3]
import src.search.llm_service as _llm_svc  # expose module-level `model` for patching


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client():
    """
    Async HTTPX test client wrapping the FastAPI app [src/main.py:3,5].
    No dependency overrides needed — this app has no DB or auth dependencies.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def mock_generate_content():
    """
    Patches src.search.llm_service.model.generate_content for one test
    [src/search/llm_service.py:71].

    Yields the MagicMock; caller sets .return_value or .side_effect.
    The patch is automatically removed after the test, preventing state leakage.

    Usage in tests:
        def test_...(client, mock_generate_content):
            mock_generate_content.return_value = make_stream_chunks('["kw"]')
            ...
    """
    with patch.object(_llm_svc.model, "generate_content") as mock_gc:
        yield mock_gc
