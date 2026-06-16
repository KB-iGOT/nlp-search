"""
Unit tests for src/search/llm_service.py business logic.

Tests call search_request() [lines 36-55] and llm_request() [lines 58-87]
directly, bypassing the HTTP layer.  Module-level env setup and vertexai
stubbing is handled by tests/conftest.py (loaded automatically by pytest).

All assertions cite the exact source line they validate.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

import src.search.llm_service as svc
from src.search.request_model import SearchModel
from tests.helpers import make_stream_chunks


# ─────────────────────────────────────────────────────────────────────────────
# search_request() — [src/search/llm_service.py:36-55]
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchRequest:

    def test_whitespace_query_returns_exception_object_bug(self):
        """
        [BUG] llm_service.py:40-41:
        `if not req_data.query.strip(): return HTTPException(status_code=400, ...)`

        "   " passes Pydantic validation (truthy string) but strip() is empty.
        Service *returns* an HTTPException object instead of raising it.

        OBSERVED: result is an HTTPException instance, not raised.
        ACTION NEEDED: change `return` → `raise` at llm_service.py:41.
        """
        req = SearchModel(query="   ", synonyms=False)
        result = svc.search_request(req)

        assert isinstance(result, HTTPException)         # [BUG: llm_service.py:41]
        assert result.status_code == 400                 # [llm_service.py:41]
        assert "empty" in result.detail.lower()          # [llm_service.py:41]

    def test_too_long_query_returns_exception_object_bug(self):
        """
        [BUG] llm_service.py:43-44:
        `return HTTPException(status_code=400, ...)` for queries > MAX_SEARCH_LEN.

        Uses SearchModel.model_construct() to bypass the Pydantic field_validator
        [request_model.py:12-13] which would normally catch this at the HTTP layer,
        allowing direct testing of the service-layer guard.

        OBSERVED: result is an HTTPException instance, not raised.
        ACTION NEEDED: change `return` → `raise` at llm_service.py:44.
        """
        req = SearchModel.model_construct(query="a" * 401, synonyms=False)
        with patch.object(svc, "settings") as mock_settings:
            mock_settings.MAX_SEARCH_LEN = "400"
            result = svc.search_request(req)

        assert isinstance(result, HTTPException)         # [BUG: llm_service.py:44]
        assert result.status_code == 400                 # [llm_service.py:44]
        assert "400" in result.detail                    # [llm_service.py:44]

    def test_valid_query_returns_data_dict(self, mock_generate_content):
        """
        Happy path: valid query → {"data": <parsed JSON>} [llm_service.py:51].
        """
        mock_generate_content.return_value = make_stream_chunks('["python", "programming"]')
        req = SearchModel(query="python programming", synonyms=False)
        result = svc.search_request(req)

        assert result == {"data": ["python", "programming"]}  # [llm_service.py:51]

    def test_model_exception_propagates_as_http_500(self, mock_generate_content):
        """
        When generate_content() raises, the outer except block [llm_service.py:52-55]
        correctly *raises* HTTPException(500).  This is the non-buggy raise path.
        """
        mock_generate_content.side_effect = RuntimeError("API down")
        req = SearchModel(query="test", synonyms=False)

        with pytest.raises(HTTPException) as exc_info:
            svc.search_request(req)

        assert exc_info.value.status_code == 500             # [llm_service.py:55]
        assert exc_info.value.detail == (
            "Internal server error during request processing."
        )                                                     # [llm_service.py:55]

    def test_500_detail_does_not_leak_internal_error_message(self, mock_generate_content):
        """
        Internal exception message must NOT appear in the HTTPException detail
        [llm_service.py:53-55 — detail is a hardcoded generic string].
        """
        internal_msg = "super-secret-internal-error-XYZ"
        mock_generate_content.side_effect = Exception(internal_msg)
        req = SearchModel(query="test", synonyms=False)

        with pytest.raises(HTTPException) as exc_info:
            svc.search_request(req)

        assert internal_msg not in exc_info.value.detail     # [llm_service.py:55]


# ─────────────────────────────────────────────────────────────────────────────
# llm_request() — [src/search/llm_service.py:58-87]
# ─────────────────────────────────────────────────────────────────────────────

class TestLlmRequest:

    def test_synonyms_false_prompt_has_no_synonym_instruction(self, mock_generate_content):
        """
        synonyms=False: `if req_data.synonyms:` is False [llm_service.py:63],
        so the synonym replacement is NOT applied to the prompt.
        """
        mock_generate_content.return_value = make_stream_chunks('["machine learning"]')
        req = SearchModel(query="machine learning", synonyms=False)
        svc.llm_request(req)

        prompt = mock_generate_content.call_args[0][0]  # first positional arg [line 71]
        assert "Add synonym" not in prompt              # [llm_service.py:64]

    def test_synonyms_true_prompt_contains_synonym_instruction(self, mock_generate_content):
        """
        synonyms=True: `']'` in the example prompt string is replaced with
        `'] \\n Add synonym for keywords wherever possible.'` [llm_service.py:65].
        The example prompt set in conftest.py is:
          ' Output as JSON array: ["keyword1"]'
        which contains `]`, so the replacement fires.
        """
        mock_generate_content.return_value = make_stream_chunks('["machine learning"]')
        req = SearchModel(query="machine learning", synonyms=True)
        svc.llm_request(req)

        prompt = mock_generate_content.call_args[0][0]  # [llm_service.py:71]
        assert "Add synonym" in prompt                   # [llm_service.py:65]

    def test_prompt_contains_original_query(self, mock_generate_content):
        """
        Query must appear verbatim in the assembled prompt [llm_service.py:67]:
        `prompt = instruction + req_data.query + example`
        """
        mock_generate_content.return_value = make_stream_chunks('["data science"]')
        req = SearchModel(query="data science tools", synonyms=False)
        svc.llm_request(req)

        prompt = mock_generate_content.call_args[0][0]  # [llm_service.py:71]
        assert "data science tools" in prompt            # [llm_service.py:67]

    def test_generate_content_called_with_stream_true(self, mock_generate_content):
        """
        generate_content must be called with stream=True [llm_service.py:73-74].
        """
        mock_generate_content.return_value = make_stream_chunks('["test"]')
        req = SearchModel(query="test", synonyms=False)
        svc.llm_request(req)

        _, kwargs = mock_generate_content.call_args
        assert kwargs.get("stream") is True              # [llm_service.py:74]

    def test_backtick_json_wrapper_is_cleaned(self, mock_generate_content):
        """
        LLM often wraps JSON in ```json ... ```.  llm_request strips these
        [llm_service.py:83]: `.replace('```','').replace('json', '')`.

        Input:  '```json\\n["python", "code"]\\n```'
        After:  '\\n["python", "code"]\\n'
        Parsed: ["python", "code"]
        """
        chunk = MagicMock()
        chunk.text = '```json\n["python", "code"]\n```'
        mock_generate_content.return_value = [chunk]

        req = SearchModel(query="test", synonyms=False)
        result = svc.llm_request(req)

        assert result == ["python", "code"]              # [llm_service.py:83]

    def test_invalid_json_response_returns_exception_object_bug(self, mock_generate_content):
        """
        [BUG] llm_service.py:84-87:
        JSONDecodeError handler *returns* HTTPException(500) instead of raising.

        OBSERVED: llm_request returns an HTTPException instance.
        ACTION NEEDED: change `return` → `raise` at llm_service.py:87.
        """
        mock_generate_content.return_value = make_stream_chunks("this is not valid JSON")
        req = SearchModel(query="test", synonyms=False)
        result = svc.llm_request(req)

        assert isinstance(result, HTTPException)         # [BUG: llm_service.py:87]
        assert result.status_code == 500                 # [llm_service.py:87]

    def test_unexpected_exception_in_llm_returns_exception_object_bug(self, mock_generate_content):
        """
        [BUG] llm_service.py:88-92:
        The broad `except Exception` handler also *returns* HTTPException(500)
        instead of raising it.

        OBSERVED: llm_request returns an HTTPException instance.
        ACTION NEEDED: change `return` → `raise` at llm_service.py:92.
        """
        # Make json.loads raise a non-JSONDecodeError exception by patching it
        import json
        mock_generate_content.return_value = make_stream_chunks('["test"]')
        req = SearchModel(query="test", synonyms=False)

        with patch.object(json, "loads", side_effect=MemoryError("OOM")):
            result = svc.llm_request(req)

        assert isinstance(result, HTTPException)         # [BUG: llm_service.py:92]
        assert result.status_code == 500                 # [llm_service.py:92]

    def test_multi_chunk_responses_are_concatenated(self, mock_generate_content):
        """
        llm_request accumulates chunk.text values [llm_service.py:76-78]:
        `for response in responses: res_text_designation += response.text`
        Verify all chunks contribute to the final parsed result.
        """
        mock_generate_content.return_value = make_stream_chunks(
            '["py', 'thon", "code', '"]'
        )
        req = SearchModel(query="python code", synonyms=False)
        result = svc.llm_request(req)

        assert result == ["python", "code"]              # [llm_service.py:76-83]


# ─────────────────────────────────────────────────────────────────────────────
# Module-level guard — GOOGLE_APPLICATION_CREDENTIALS injection
# [src/search/llm_service.py:18-19]
# ─────────────────────────────────────────────────────────────────────────────

def test_google_credentials_env_var_set_from_settings_when_absent():
    """
    Covers llm_service.py line 19:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS

    The guard on line 18 (`if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ`)
    is never True in normal test runs because conftest.py pre-sets the var at line 30.
    This test temporarily removes the var, forces a fresh module import so the guard
    is evaluated from scratch, and asserts the assignment on line 19 fires correctly.

    Technique: sys.modules.pop forces Python to re-execute the module body on the
    next import.  src.config.Settings is patched so the re-import does not fail on
    missing required env fields.  The vertexai stubs installed by conftest.py are
    still in sys.modules, so vertexai.init() and GenerativeModel() are no-ops.
    """
    import os
    import sys

    saved = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
    expected_cred = "/tmp/test-credentials.json"

    mock_settings_instance = MagicMock()
    mock_settings_instance.GOOGLE_APPLICATION_CREDENTIALS = expected_cred
    mock_settings_instance.GOOGLE_CLOUD_PROJECT = "test-project-id"
    mock_settings_instance.GOOGLE_CLOUD_LOCATION = "us-central1"
    mock_settings_instance.MODEL_NAME = "test-model"
    mock_settings_instance.MAX_OUTPUT_TOKENS = "8192"
    mock_settings_instance.TEMPERATURE = "0"
    mock_settings_instance.TOP_P = "0.95"
    mock_settings_instance.TOP_K = "1"
    mock_settings_instance.NLP_SEARCH_INSTRUCTION_PROMPT = "Extract search keywords from: "
    mock_settings_instance.NPL_SEARCH_EXAMPLE_PROMPT = ' Output as JSON array: ["keyword1"]'
    mock_settings_instance.MAX_SEARCH_LEN = "400"

    try:
        # Force the module body to run again on the next import statement.
        sys.modules.pop("src.search.llm_service", None)

        with patch("src.config.Settings", return_value=mock_settings_instance):
            import src.search.llm_service  # noqa: F401 — triggers lines 18-19

        # Line 19 must have written the value into os.environ.
        assert os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") == expected_cred  # [llm_service.py:19]
    finally:
        # Restore the original env value so subsequent tests are unaffected.
        if saved is not None:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = saved
        elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        # Drop the re-loaded module.  The module-level `svc` and `_llm_svc`
        # aliases in other test files still hold a reference to the original
        # module object, so no other test is affected.
        sys.modules.pop("src.search.llm_service", None)
