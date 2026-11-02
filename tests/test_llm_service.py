import asyncio
import json
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.exceptions import HTTPException
from google.auth import exceptions as google_auth_exceptions
from google.genai import errors as genai_errors
from tenacity import wait_none

from src.search import llm_service
from src.search.request_model import SearchModel


# ---------------------------------------------------------------------- #
# Helpers and fixtures                                                    #
# ---------------------------------------------------------------------- #

def api_error(code: int) -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"code": code, "message": f"error {code}", "status": "ERROR"}})


def model_response(text, candidates=None, prompt_feedback=None):
    return SimpleNamespace(
        text=text,
        candidates=candidates or [],
        prompt_feedback=prompt_feedback,
    )


KEYWORDS = {"keywords": [{"keyword": "data mining", "priority": 1}]}


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    # Retries happen at once rather than after 1s, 2s, 4s of real waiting.
    monkeypatch.setattr(llm_service, "wait_exponential_jitter", lambda **_: wait_none())


@pytest.fixture(autouse=True)
async def clear_inflight():
    yield
    tasks = list(llm_service._inflight.values())
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    llm_service._inflight.clear()


@pytest.fixture
def model(monkeypatch):
    """Stands in for client.aio.models.generate_content."""
    generate = AsyncMock(return_value=model_response(json.dumps(KEYWORDS)))
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(llm_service, "client", client)
    return generate


@pytest.fixture
def cache(monkeypatch):
    lookup = AsyncMock(return_value=(None, 1))
    store = AsyncMock()
    monkeypatch.setattr(llm_service.redis_service, "lookup", lookup)
    monkeypatch.setattr(llm_service.redis_service, "store", store)
    return SimpleNamespace(lookup=lookup, store=store)


# ---------------------------------------------------------------------- #
# Response schema and generation config                                   #
# ---------------------------------------------------------------------- #

class TestBuildKeywordsSchema:
    def test_without_synonyms_leaves_the_field_out(self):
        item = llm_service.build_keywords_schema(False)["properties"]["keywords"]["items"]

        assert set(item["properties"]) == {"keyword", "priority"}
        assert item["required"] == ["keyword", "priority"]
        assert item["propertyOrdering"] == ["keyword", "priority"]

    def test_with_synonyms_requires_at_least_one(self):
        item = llm_service.build_keywords_schema(True)["properties"]["keywords"]["items"]

        assert item["properties"]["synonyms"] == {"type": "ARRAY", "items": {"type": "STRING"}, "minItems": 1}
        assert item["required"] == ["keyword", "priority", "synonyms"]
        assert item["propertyOrdering"] == ["keyword", "priority", "synonyms"]

    @pytest.mark.parametrize("synonyms", [False, True])
    def test_top_level_shape(self, synonyms):
        schema = llm_service.build_keywords_schema(synonyms)

        assert schema["type"] == "OBJECT"
        assert schema["required"] == ["keywords"]
        assert schema["properties"]["keywords"]["type"] == "ARRAY"
        assert schema["properties"]["keywords"]["items"]["properties"]["priority"] == {"type": "INTEGER", "minimum": 1}


class TestBuildGenerationConfig:
    def test_uses_settings_and_json_mode(self):
        config = llm_service.build_generation_config(True)

        assert config.system_instruction == llm_service.SYSTEM_INSTRUCTION
        assert config.max_output_tokens == 256
        assert config.temperature == 0
        assert config.top_p == 0.95
        assert config.top_k == 1
        assert config.response_mime_type == "application/json"
        assert config.response_schema == llm_service.build_keywords_schema(True)


# ---------------------------------------------------------------------- #
# Error classification                                                    #
# ---------------------------------------------------------------------- #

class TestIsRetryable:
    @pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
    def test_retryable_status_codes(self, code):
        assert llm_service.is_retryable(api_error(code))

    @pytest.mark.parametrize("code", [400, 401, 403, 404])
    def test_client_errors_are_not_retried(self, code):
        assert not llm_service.is_retryable(api_error(code))

    @pytest.mark.parametrize("error", [
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("slow"),
        TimeoutError(),
        google_auth_exceptions.TransportError("token refresh failed"),
        json.JSONDecodeError("bad", "{", 0),
    ])
    def test_transport_and_parse_errors_are_retried(self, error):
        assert llm_service.is_retryable(error)

    @pytest.mark.parametrize("error", [
        llm_service.EmptyResponseError("blocked"),
        ValueError("other"),
        RuntimeError("other"),
    ])
    def test_other_errors_are_not_retried(self, error):
        assert not llm_service.is_retryable(error)


class TestWhyEmpty:
    def test_prompt_blocked(self):
        response = model_response(None, prompt_feedback=SimpleNamespace(block_reason="SAFETY"))
        assert llm_service._why_empty(response) == "prompt blocked: SAFETY"

    def test_finish_reason_of_first_candidate(self):
        response = model_response(
            None,
            prompt_feedback=SimpleNamespace(block_reason=None),
            candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
        )
        assert llm_service._why_empty(response) == "finish reason: MAX_TOKENS"

    def test_no_candidates(self):
        assert llm_service._why_empty(model_response(None)) == "no candidates returned"


class TestUpstreamStatus:
    @pytest.mark.parametrize("code, expected", [
        (408, 504),
        (429, 429),
        (503, 503),
        (504, 504),
        (400, 502),
        (401, 502),
        (500, 502),
    ])
    def test_mapping(self, code, expected):
        assert llm_service._upstream_status(code) == expected


# ---------------------------------------------------------------------- #
# generate_content: one attempt                                           #
# ---------------------------------------------------------------------- #

class TestGenerateContent:
    async def test_returns_parsed_json(self, model):
        result = await llm_service.generate_content("prompt", True)

        assert result == KEYWORDS
        kwargs = model.await_args.kwargs
        assert kwargs["model"] == "test-model"
        assert kwargs["contents"] == "prompt"
        assert kwargs["config"].response_schema == llm_service.build_keywords_schema(True)

    async def test_empty_answer_raises(self, model):
        model.return_value = model_response("", prompt_feedback=SimpleNamespace(block_reason="SAFETY"))

        with pytest.raises(llm_service.EmptyResponseError, match="prompt blocked: SAFETY"):
            await llm_service.generate_content("prompt", False)

    async def test_unparseable_answer_raises(self, model):
        model.return_value = model_response('{"keywords": [')

        with pytest.raises(json.JSONDecodeError):
            await llm_service.generate_content("prompt", False)

    async def test_api_error_is_reraised(self, model):
        model.side_effect = api_error(429)

        with pytest.raises(genai_errors.APIError) as raised:
            await llm_service.generate_content("prompt", False)

        assert raised.value.code == 429


# ---------------------------------------------------------------------- #
# generate_with_retries                                                   #
# ---------------------------------------------------------------------- #

class TestGenerateWithRetries:
    async def test_retries_a_retryable_error_then_succeeds(self, model):
        model.side_effect = [api_error(503), model_response(json.dumps(KEYWORDS))]

        assert await llm_service.generate_with_retries("prompt", False) == KEYWORDS
        assert model.await_count == 2

    async def test_retries_an_unparseable_answer(self, model):
        model.side_effect = [model_response("not json"), model_response(json.dumps(KEYWORDS))]

        assert await llm_service.generate_with_retries("prompt", False) == KEYWORDS
        assert model.await_count == 2

    async def test_non_retryable_error_fails_on_first_attempt(self, model):
        model.side_effect = api_error(400)

        with pytest.raises(genai_errors.APIError) as raised:
            await llm_service.generate_with_retries("prompt", False)

        assert raised.value.code == 400
        assert model.await_count == 1

    async def test_gives_up_after_max_retries_with_original_error(self, model):
        model.side_effect = api_error(503)

        with pytest.raises(genai_errors.APIError) as raised:
            await llm_service.generate_with_retries("prompt", False)

        assert raised.value.code == 503
        assert model.await_count == llm_service.settings.LLM_MAX_RETRIES + 1

    async def test_zero_retries_calls_once(self, model, monkeypatch):
        monkeypatch.setattr(llm_service.settings, "LLM_MAX_RETRIES", 0)
        model.side_effect = api_error(503)

        with pytest.raises(genai_errors.APIError):
            await llm_service.generate_with_retries("prompt", False)

        assert model.await_count == 1

    async def test_slow_attempt_is_cut_off_and_retried(self, model, monkeypatch):
        monkeypatch.setattr(llm_service.settings, "LLM_ATTEMPT_TIMEOUT_SECONDS", 0.05)
        responses = iter([None, model_response(json.dumps(KEYWORDS))])

        async def first_hangs(**_):
            response = next(responses)
            if response is None:
                await asyncio.sleep(10)
            return response

        model.side_effect = first_hangs

        assert await llm_service.generate_with_retries("prompt", False) == KEYWORDS
        assert model.await_count == 2

    async def test_every_attempt_too_slow_raises_timeout(self, model, monkeypatch):
        monkeypatch.setattr(llm_service.settings, "LLM_ATTEMPT_TIMEOUT_SECONDS", 0.02)
        monkeypatch.setattr(llm_service.settings, "LLM_MAX_RETRIES", 1)

        async def hang(**_):
            await asyncio.sleep(10)

        model.side_effect = hang

        with pytest.raises(TimeoutError):
            await llm_service.generate_with_retries("prompt", False)
        assert model.await_count == 2


# ---------------------------------------------------------------------- #
# llm_request: prompt building and error mapping                          #
# ---------------------------------------------------------------------- #

class TestLlmRequest:
    async def test_builds_prompt_without_synonyms(self, monkeypatch):
        generate = AsyncMock(return_value=KEYWORDS)
        monkeypatch.setattr(llm_service, "generate_with_retries", generate)

        result = await llm_service.llm_request(SearchModel(query="data mining"))

        assert result == KEYWORDS
        assert generate.await_args.args == ("Extract keywords: [data mining] Example: [x]", False)

    async def test_adds_synonym_instruction_once_after_the_query(self, monkeypatch):
        generate = AsyncMock(return_value=KEYWORDS)
        monkeypatch.setattr(llm_service, "generate_with_retries", generate)

        await llm_service.llm_request(SearchModel(query="ai", synonyms=True))

        prompt, synonyms = generate.await_args.args
        assert prompt == "Extract keywords: [ai] \n Add synonym for keywords wherever possible. Example: [x]"
        assert synonyms is True

    async def test_synonym_instruction_stays_out_of_the_example_answers(self, monkeypatch):
        # Prompts shaped like the real ones: every example answer has its own ']'.
        monkeypatch.setattr(llm_service.settings, "NLP_SEARCH_INSTRUCTION_PROMPT", "Learner's search query: [")
        monkeypatch.setattr(
            llm_service.settings, "NPL_SEARCH_EXAMPLE_PROMPT",
            ']\nExamples:\n Response: {"keywords" : [{"keyword" : "a", "priority":1}]}'
            '\n Response: {"keywords" : [{"keyword" : "b", "priority":1}]}',
        )
        generate = AsyncMock(return_value=KEYWORDS)
        monkeypatch.setattr(llm_service, "generate_with_retries", generate)

        await llm_service.llm_request(SearchModel(query="ai", synonyms=True))

        prompt = generate.await_args.args[0]
        assert prompt.count("Add synonym for keywords wherever possible.") == 1
        assert prompt.startswith("Learner's search query: [ai] \n Add synonym for keywords wherever possible.\nExamples:")
        assert prompt.endswith('[{"keyword" : "b", "priority":1}]}')

    @pytest.mark.parametrize("error, status", [
        (api_error(429), 429),
        (api_error(503), 503),
        (api_error(408), 504),
        (api_error(400), 502),
        (TimeoutError(), 504),
        (httpx.ReadTimeout("slow"), 504),
        (httpx.ConnectError("refused"), 503),
        (google_auth_exceptions.TransportError("no token"), 503),
        (json.JSONDecodeError("bad", "{", 0), 502),
        (llm_service.EmptyResponseError("blocked"), 502),
        (RuntimeError("unexpected"), 500),
    ])
    async def test_errors_are_returned_as_http_exceptions(self, monkeypatch, error, status):
        monkeypatch.setattr(llm_service, "generate_with_retries", AsyncMock(side_effect=error))

        result = await llm_service.llm_request(SearchModel(query="data mining"))

        assert isinstance(result, HTTPException)
        assert result.status_code == status


# ---------------------------------------------------------------------- #
# Background fetch, de-duplication and shutdown                           #
# ---------------------------------------------------------------------- #

class TestFetchAndCache:
    async def test_caches_a_successful_answer(self, monkeypatch, cache):
        monkeypatch.setattr(llm_service, "llm_request", AsyncMock(return_value=KEYWORDS))
        req = SearchModel(query="data mining")

        assert await llm_service._fetch_and_cache("key", req) == KEYWORDS
        cache.store.assert_awaited_once_with("key", KEYWORDS, "data mining", llm_service.settings)

    async def test_does_not_cache_a_failure(self, monkeypatch, cache):
        failure = HTTPException(status_code=502, detail="bad")
        monkeypatch.setattr(llm_service, "llm_request", AsyncMock(return_value=failure))

        assert await llm_service._fetch_and_cache("key", SearchModel(query="q")) is failure
        cache.store.assert_not_awaited()

    async def test_unexpected_error_becomes_500(self, monkeypatch, cache):
        monkeypatch.setattr(llm_service, "llm_request", AsyncMock(return_value=KEYWORDS))
        cache.store.side_effect = RuntimeError("redis down")

        result = await llm_service._fetch_and_cache("key", SearchModel(query="q"))

        assert isinstance(result, HTTPException)
        assert result.status_code == 500
        assert result.detail == llm_service.FAILED_MESSAGE


class TestGetOrStart:
    async def test_joins_running_task_and_forgets_it_when_done(self, monkeypatch):
        release = asyncio.Event()

        async def fetch(*_):
            await release.wait()
            return KEYWORDS

        monkeypatch.setattr(llm_service, "_fetch_and_cache", fetch)
        req = SearchModel(query="q")

        first, started_first = llm_service._get_or_start("key", req)
        second, started_second = llm_service._get_or_start("key", req)

        assert first is second
        assert (started_first, started_second) == (True, False)
        assert llm_service._inflight["key"] is first

        release.set()
        assert await first == KEYWORDS
        await asyncio.sleep(0)
        assert "key" not in llm_service._inflight

        third, started_third = llm_service._get_or_start("key", req)
        assert third is not first and started_third
        await third

    async def test_different_keys_get_different_tasks(self, monkeypatch):
        monkeypatch.setattr(llm_service, "_fetch_and_cache", AsyncMock(return_value=KEYWORDS))
        req = SearchModel(query="q")

        a, _ = llm_service._get_or_start("a", req)
        b, _ = llm_service._get_or_start("b", req)

        assert a is not b
        await asyncio.gather(a, b)


class TestDrainBackgroundTasks:
    async def test_nothing_running(self):
        await llm_service.drain_background_tasks()

    async def test_waits_for_tasks_that_finish_in_time(self):
        async def quick():
            await asyncio.sleep(0.01)
            return KEYWORDS

        task = asyncio.create_task(quick())
        llm_service._inflight["key"] = task

        await llm_service.drain_background_tasks()

        assert task.done() and task.result() == KEYWORDS

    async def test_cancels_tasks_past_the_grace_period(self, monkeypatch):
        monkeypatch.setattr(llm_service, "SHUTDOWN_GRACE_SECONDS", 0.02)
        task = asyncio.create_task(asyncio.sleep(10))
        llm_service._inflight["key"] = task

        await llm_service.drain_background_tasks()

        assert task.cancelled()


# ---------------------------------------------------------------------- #
# search_request: the endpoint's entry point                              #
# ---------------------------------------------------------------------- #

class TestSearchRequest:
    @pytest.mark.parametrize("query", [" ", "   \n\t "])
    async def test_blank_query_is_400(self, query, cache):
        with pytest.raises(HTTPException) as raised:
            await llm_service.search_request(SearchModel(query=query))

        assert raised.value.status_code == 400
        assert raised.value.detail == "Query cannot be empty."
        cache.lookup.assert_not_awaited()

    async def test_query_over_max_length_is_400(self, monkeypatch, cache):
        monkeypatch.setattr(llm_service.settings, "MAX_SEARCH_LEN", "5")

        with pytest.raises(HTTPException) as raised:
            await llm_service.search_request(SearchModel(query="too long"))

        assert raised.value.status_code == 400
        assert "5 characters" in raised.value.detail
        cache.lookup.assert_not_awaited()

    async def test_cache_hit_skips_the_model(self, model, cache):
        cache.lookup.return_value = (KEYWORDS, 7)

        result = await llm_service.search_request(SearchModel(query="data mining"))

        assert result == {"data": KEYWORDS}
        model.assert_not_awaited()

    async def test_cache_miss_calls_model_and_caches(self, model, cache):
        result = await llm_service.search_request(SearchModel(query="data mining"))

        assert result == {"data": KEYWORDS}
        model.assert_awaited_once()
        cache.store.assert_awaited_once()
        assert cache.store.await_args.args[1] == KEYWORDS

    async def test_model_failure_raises_mapped_status(self, model, cache, monkeypatch):
        monkeypatch.setattr(llm_service.settings, "LLM_MAX_RETRIES", 0)
        model.side_effect = api_error(429)

        with pytest.raises(HTTPException) as raised:
            await llm_service.search_request(SearchModel(query="data mining"))

        assert raised.value.status_code == 429
        cache.store.assert_not_awaited()

    async def test_unexpected_error_is_500(self, cache):
        cache.lookup.side_effect = RuntimeError("redis exploded")

        with pytest.raises(HTTPException) as raised:
            await llm_service.search_request(SearchModel(query="data mining"))

        assert raised.value.status_code == 500
        assert raised.value.detail == "Internal server error during request processing."

    async def test_cancelled_fetch_is_503(self, monkeypatch, cache):
        task = asyncio.create_task(asyncio.sleep(10))
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        monkeypatch.setattr(llm_service, "_get_or_start", lambda *_: (task, True))

        with pytest.raises(HTTPException) as raised:
            await llm_service.search_request(SearchModel(query="data mining"))

        assert raised.value.status_code == 503

    async def test_identical_concurrent_requests_share_one_model_call(self, model, cache):
        async def slow(**_):
            await asyncio.sleep(0.02)
            return model_response(json.dumps(KEYWORDS))

        model.side_effect = slow
        req = SearchModel(query="data mining")

        results = await asyncio.gather(llm_service.search_request(req), llm_service.search_request(req))

        assert results == [{"data": KEYWORDS}, {"data": KEYWORDS}]
        assert model.await_count == 1
        cache.store.assert_awaited_once()


class TestSearchRequestTimeout:
    @pytest.fixture
    def slow_model(self, model, monkeypatch):
        monkeypatch.setattr(llm_service.settings, "RESPONSE_TIMEOUT_SECONDS", 0.02)
        release = asyncio.Event()

        async def wait_for_release(**_):
            await release.wait()
            return model_response(json.dumps(KEYWORDS))

        model.side_effect = wait_for_release
        return release

    async def test_returns_202_and_caches_in_background(self, slow_model, cache):
        req = SearchModel(query="data mining")

        with pytest.raises(HTTPException) as raised:
            await llm_service.search_request(req)

        assert raised.value.status_code == 202
        assert raised.value.detail == llm_service.TIMED_OUT_MESSAGE
        cache.store.assert_not_awaited()

        key = llm_service.redis_service.build_key(req.query, req.synonyms, llm_service.settings)
        task = llm_service._inflight[key]
        slow_model.set()
        await task

        cache.store.assert_awaited_once()
        assert cache.store.await_args.args[1] == KEYWORDS

    async def test_identical_requests_that_time_out_share_one_model_call(self, slow_model, model, cache):
        req = SearchModel(query="data mining")

        results = await asyncio.gather(
            llm_service.search_request(req), llm_service.search_request(req), return_exceptions=True,
        )

        assert [r.status_code for r in results] == [202, 202]

        slow_model.set()
        await llm_service.drain_background_tasks()

        assert model.await_count == 1
        cache.store.assert_awaited_once()
