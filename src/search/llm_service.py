import os, json
import asyncio
import traceback
import logging
import httpx
from fastapi.exceptions import HTTPException
from google import genai
from google.genai import errors as genai_errors, types
from google.auth import exceptions as google_auth_exceptions
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)
from src.config import get_settings
from src.services.redis_service import redis_service
from src.search.request_model import SearchModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=settings.GOOGLE_APPLICATION_CREDENTIALS

SYSTEM_INSTRUCTION = [
    "You are a helpful language expert.",
    "Your mission is to extract search keywords from queries."
]

def build_keywords_schema(synonyms: bool) -> dict:
    """Shape of every answer. Vertex AI decodes against it, so the model can only
    reply with this JSON: no code fences, no escaped quotes, no echoed "Query: ..."
    lines.

    The two variants differ only in synonyms. A field the schema leaves optional is
    one the model is free to skip, so when synonyms are asked for they are required,
    with at least one per keyword; when they are not, the field is left out and the
    model has no way to produce it.
    """
    properties = {
        "keyword": {"type": "STRING"},
        "priority": {"type": "INTEGER", "minimum": 1},
    }
    if synonyms:
        properties["synonyms"] = {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "minItems": 1
        }

    return {
        "type": "OBJECT",
        "properties": {
            "keywords": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": properties,
                    "required": list(properties),
                    # Gemini writes properties alphabetically unless told otherwise; the
                    # keyword goes first so priority and synonyms are chosen with it in view.
                    "propertyOrdering": list(properties)
                }
            }
        },
        "required": ["keywords"]
    }


def build_generation_config(synonyms: bool) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        max_output_tokens=int(settings.MAX_OUTPUT_TOKENS),
        temperature=float(settings.TEMPERATURE),
        top_p=float(settings.TOP_P),
        top_k=int(settings.TOP_K),
        response_mime_type="application/json",
        response_schema=build_keywords_schema(synonyms)
    )


# No retry_options: the SDK then makes exactly one attempt per call, and every
# retry below is tenacity's. `timeout` (milliseconds) is applied by httpx per
# connect and per read, so a slow answer can outlast it; generate_with_retries
# holds each attempt to the same limit in full.
client = genai.Client(
    vertexai=True,
    project=settings.GOOGLE_CLOUD_PROJECT,
    location=settings.GOOGLE_CLOUD_LOCATION,
    http_options=types.HttpOptions(timeout=int(settings.LLM_ATTEMPT_TIMEOUT_SECONDS * 1000)),
)

# Statuses worth calling the model again for: rate limited, or the service failed
# on its side. Anything else (bad credentials, a malformed request) is raised on
# the first attempt, since repeating it would only cost the caller time.
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)

# Failures that never reached the model or never finished: httpx's own errors
# (httpx.TimeoutException is a TransportError), TimeoutError from the per-attempt
# limit, and google.auth's from refreshing the access token.
RETRYABLE_TRANSPORT_ERRORS = (httpx.TransportError, TimeoutError, google_auth_exceptions.TransportError)

# The model answered, but not with JSON we can read. Retried like a failed call,
# on the chance the next answer parses. An answer with no text at all (blocked by
# the safety filters) raises EmptyResponseError instead, a first-attempt failure.
RETRYABLE_RESPONSE_ERRORS = (json.JSONDecodeError,)


class EmptyResponseError(Exception):
    """The model returned no text, typically because the answer was blocked."""


def is_retryable(error: BaseException) -> bool:
    if isinstance(error, genai_errors.APIError):
        return error.code in RETRYABLE_STATUS_CODES
    return isinstance(error, RETRYABLE_TRANSPORT_ERRORS + RETRYABLE_RESPONSE_ERRORS)


def _why_empty(response) -> str:
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and feedback.block_reason:
        return f"prompt blocked: {feedback.block_reason}"
    if response.candidates:
        return f"finish reason: {response.candidates[0].finish_reason}"
    return "no candidates returned"


async def generate_content(prompt: str, synonyms: bool):
    """Call the model once and parse its answer into JSON.

    The parse sits inside the retried call, so that an answer which arrives but
    cannot be read is retried as well.
    """
    response = await client.aio.models.generate_content(
        model=settings.MODEL_NAME,
        contents=prompt,
        config=build_generation_config(synonyms),
    )
    res_text_designation = response.text
    if not res_text_designation:
        raise EmptyResponseError(_why_empty(response))

    logger.info(f"Model Response :: {res_text_designation}")

    # JSON mode returns bare JSON. It can still fail to parse if the answer was cut
    # off at max_output_tokens, which is why this stays a retryable error.
    return json.loads(res_text_designation)


async def generate_with_retries(prompt: str, synonyms: bool):
    """generate_content with retries, each attempt held to LLM_ATTEMPT_TIMEOUT_SECONDS.

    An attempt still running at the limit is cancelled and raises TimeoutError,
    which is retried like any other timeout. The whole step therefore takes at
    most (LLM_MAX_RETRIES + 1) attempts' worth of time, plus the waits between them.
    """
    async for attempt in AsyncRetrying(
        # LLM_MAX_RETRIES counts retries, so the first call is the extra attempt.
        stop=stop_after_attempt(settings.LLM_MAX_RETRIES + 1),
        # 1s before the first retry, doubling up to 4s, plus up to 1s of jitter
        # so that callers rate limited together do not retry together.
        wait=wait_exponential_jitter(initial=1, max=4, exp_base=2, jitter=1),
        retry=retry_if_exception(is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        # Surface the original error rather than tenacity's RetryError.
        reraise=True,
    ):
        with attempt:
            async with asyncio.timeout(settings.LLM_ATTEMPT_TIMEOUT_SECONDS):
                return await generate_content(prompt, synonyms)


async def search_request(req_data: SearchModel):
    try:
        logger.info(f"Received nlp serch request :: {req_data.model_dump()}")

        if not req_data.query.strip():
            return HTTPException(status_code=400, detail="Query cannot be empty.")

        if len(req_data.query) > int(settings.MAX_SEARCH_LEN):
            return HTTPException(status_code=400, detail=f"Query cannot be longer than {int(settings.MAX_SEARCH_LEN)} characters")

        cache_key = redis_service.build_key(req_data.query, req_data.synonyms, settings)

        # One round trip counts the request and reads back any cached answer.
        cached, count = await redis_service.lookup(cache_key, req_data.query, settings)

        if cached is not None:
            logger.info(f"Cache HIT :: {cache_key} count={count}")
            return {"data" : cached}

        logger.info(f"Cache MISS :: {cache_key} count={count}")

        response = await llm_request(req_data)
        if isinstance(response, Exception):
            return response

        await redis_service.store(cache_key, response, req_data.query, settings)

        logger.info(f"Response :: {response}")
        return {"data" : response}
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error during request processing.")


def _upstream_status(code: int) -> int:
    """Status for a model error we pass on. 429, 503 and 504 mean the same to our
    caller as they did to us; any other error is the model failing us, a 502."""
    return code if code in (429, 503, 504) else 502


async def llm_request(req_data: SearchModel):

    instruction = settings.NLP_SEARCH_INSTRUCTION_PROMPT
    example = settings.NPL_SEARCH_EXAMPLE_PROMPT

    if req_data.synonyms:
        instruction = instruction.replace(']' , '] \n Add synonym for keywords wherever possible.')
        example = example.replace(']' , '] \n Add synonym for keywords wherever possible.')

    prompt = instruction + req_data.query + example

    logger.info(f"Final prompt :: {prompt}")

    try:
        return await generate_with_retries(prompt, req_data.synonyms)
    except genai_errors.APIError as e:
        logger.error(f"Vertex AI request failed: {e}")
        return HTTPException(status_code=_upstream_status(e.code), detail=str(e))
    except (TimeoutError, httpx.TimeoutException) as e:
        logger.error(f"Vertex AI request timed out after {settings.LLM_ATTEMPT_TIMEOUT_SECONDS}s: {e!r}")
        return HTTPException(status_code=504, detail="The query took too long to process. Please try again later!")
    except RETRYABLE_TRANSPORT_ERRORS as e:
        logger.error(f"Could not reach Vertex AI: {e!r}")
        return HTTPException(status_code=503, detail="Failed to process the query. Please try again later!")
    except (json.JSONDecodeError, EmptyResponseError) as e:
        logger.error(f"Unusable LLM response: {e!r}")
        return HTTPException(status_code=502, detail="Failed to process the query. Please try again later!")
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM processing: {e}")
        traceback.print_exc()
        return HTTPException(status_code=500, detail="Failed to process the query. Please try again later!")
