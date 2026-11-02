import os

# Settings are read once, when src.config is first imported, so they are fixed
# here before any test module imports the service. Values set in the environment
# take precedence over .env, which keeps the tests independent of a developer's
# local configuration. Nothing here reaches Vertex AI or Redis: the tests
# replace the client and the cache.
TEST_ENV = {
    "GOOGLE_CLOUD_PROJECT": "test-project",
    "GOOGLE_CLOUD_LOCATION": "us-central1",
    "GOOGLE_APPLICATION_CREDENTIALS": "test-credentials.json",
    "MODEL_NAME": "test-model",
    "MAX_OUTPUT_TOKENS": "256",
    "TEMPERATURE": "0",
    "TOP_P": "0.95",
    "TOP_K": "1",
    "MAX_SEARCH_LEN": "400",
    "LLM_MAX_RETRIES": "2",
    "LLM_ATTEMPT_TIMEOUT_SECONDS": "10",
    "RESPONSE_TIMEOUT_SECONDS": "3",
    "NLP_SEARCH_INSTRUCTION_PROMPT": "Extract keywords: [",
    "NPL_SEARCH_EXAMPLE_PROMPT": "] Example: [x]",
    "REDIS_ENABLED": "false",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "REDIS_DB": "0",
    "REDIS_PASSWORD": "",
    "REDIS_SSL": "false",
    "REDIS_KEY_PREFIX": "test",
    "REDIS_CACHE_TTL_DAYS": "1",
    "REDIS_QUERY_COUNTER_ENABLED": "true",
}
os.environ.update(TEST_ENV)
