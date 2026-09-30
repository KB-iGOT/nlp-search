import os, json
import traceback
import logging
from fastapi.exceptions import HTTPException
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from google.api_core import exceptions as google_exceptions
from google.auth import exceptions as google_auth_exceptions
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
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


def build_generation_config(synonyms: bool) -> GenerationConfig:
    return GenerationConfig(
        max_output_tokens=int(settings.MAX_OUTPUT_TOKENS),
        temperature=float(settings.TEMPERATURE),
        top_p=float(settings.TOP_P),
        top_k=int(settings.TOP_K),
        response_mime_type="application/json",
        response_schema=build_keywords_schema(synonyms)
    )


vertexai.init(project=settings.GOOGLE_CLOUD_PROJECT, location=settings.GOOGLE_CLOUD_LOCATION)
model = GenerativeModel(
        settings.MODEL_NAME,
        system_instruction=[
            "You are a helpful language expert.",
            "Your mission is to extract search keywords from queries."
        ]
)

# Failures worth calling the model again for: the request never got a verdict of
# its own. Everything else (bad credentials, a malformed request, a refusal) is
# raised on the first attempt, since repeating it would only cost the caller time.
RETRYABLE_LLM_ERRORS = (
    google_exceptions.ServerError,        # 500, 502, 503, 504, incl. DeadlineExceeded
    google_exceptions.TooManyRequests,    # 429
    google_exceptions.ResourceExhausted,  # quota exhausted
    google_exceptions.Aborted,            # contention, safe to repeat
    google_auth_exceptions.TransportError,
    ConnectionError,
    TimeoutError,
)

# The model answered, but not with JSON we can read. Retried like a failed call,
# on the chance the next answer parses. Only JSONDecodeError is listed: a response blocked by the
# safety filters raises a plain ValueError, which stays a first-attempt failure.
RETRYABLE_RESPONSE_ERRORS = (json.JSONDecodeError,)


@retry(
    # LLM_MAX_RETRIES counts retries, so the first call is the extra attempt.
    stop=stop_after_attempt(settings.LLM_MAX_RETRIES + 1),
    # 1s before the first retry, doubling each time after that.
    wait=wait_exponential(multiplier=1),
    retry=retry_if_exception_type(RETRYABLE_LLM_ERRORS + RETRYABLE_RESPONSE_ERRORS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    # Surface the original Vertex AI error rather than tenacity's RetryError.
    reraise=True,
)
def generate_content(prompt: str, synonyms: bool):
    """Call the model, collect the streamed answer and parse it into JSON.

    The stream is drained inside the retried call on purpose: with streaming the
    request is only really finished once the last chunk arrives, so a connection
    dropped mid-stream has to restart the whole generation. The parse sits here
    too, so that an answer which arrives but cannot be read is retried as well.
    """
    responses = model.generate_content(
        prompt,
        generation_config=build_generation_config(synonyms),
        #safety_settings=safety_settings,
        stream=True
    )
    res_text_designation = ""
    for response in responses:
        res_text_designation += response.text

    logger.info(f"Model Response :: {res_text_designation}")

    # JSON mode returns bare JSON. It can still fail to parse if the answer was cut
    # off at max_output_tokens, which is why this stays a retryable error.
    return json.loads(res_text_designation)


def search_request(req_data: SearchModel):
    try:
        logger.info(f"Received nlp serch request :: {req_data.model_dump()}")

        if not req_data.query.strip():
            return HTTPException(status_code=400, detail="Query cannot be empty.")

        if len(req_data.query) > int(settings.MAX_SEARCH_LEN):
            return HTTPException(status_code=400, detail=f"Query cannot be longer than {int(settings.MAX_SEARCH_LEN)} characters")

        cache_key = redis_service.build_key(req_data.query, req_data.synonyms, settings)

        # One round trip counts the request and reads back any cached answer.
        cached, count = redis_service.lookup(cache_key, req_data.query, settings)

        if cached is not None:
            logger.info(f"Cache HIT :: {cache_key} count={count}")
            return {"data" : cached}

        logger.info(f"Cache MISS :: {cache_key} count={count}")

        response = llm_request(req_data)
        if isinstance(response, Exception):
            return response

        redis_service.store(cache_key, response, req_data.query, settings)

        logger.info(f"Response :: {response}")
        return {"data" : response}
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error during request processing.")


def llm_request(req_data: SearchModel):
    
    instruction = settings.NLP_SEARCH_INSTRUCTION_PROMPT
    example = settings.NPL_SEARCH_EXAMPLE_PROMPT

    if req_data.synonyms:
        instruction = instruction.replace(']' , '] \n Add synonym for keywords wherever possible.')
        example = example.replace(']' , '] \n Add synonym for keywords wherever possible.')

    prompt = instruction + req_data.query + example
    
    logger.info(f"Final prompt :: {prompt}")

    try:
        return generate_content(prompt, req_data.synonyms)
    except RETRYABLE_LLM_ERRORS as e:
        logger.error(f"Vertex AI still failing after {settings.LLM_MAX_RETRIES} retries: {e}")
        # Hand back whatever Vertex AI answered with. Errors that never reached
        # the service (a dropped connection, a timeout) carry no status of their
        # own, so those fall back to the same 500 the rest of this file uses.
        return HTTPException(status_code=getattr(e, "code", None) or 500, detail=str(e))
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode LLM response after {settings.LLM_MAX_RETRIES} retries")
        logger.error(f"JSONDecodeError: {e}")
        return HTTPException(status_code=500, detail="Failed to process the query. Please try again later!")
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM processing: {e}")
        traceback.print_exc()
        return HTTPException(status_code=500, detail="Failed to process the query. Please try again later!")