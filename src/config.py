import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../.env"
        ),
        extra='ignore'
    )

    GOOGLE_CLOUD_PROJECT: str
    GOOGLE_CLOUD_LOCATION: str = Field(default="us-central1")
    GOOGLE_APPLICATION_CREDENTIALS: str

    MODEL_NAME: str = Field(default="gemini-2.0-flash-lite")
    MAX_OUTPUT_TOKENS: str = Field(default="8192")
    TEMPERATURE: str = Field(default="0")
    TOP_P: str = Field(default="0.95")
    TOP_K: str = Field(default="1")
    MAX_SEARCH_LEN: str = Field(default="400")

    NLP_SEARCH_INSTRUCTION_PROMPT: str
    NPL_SEARCH_EXAMPLE_PROMPT: str

    # Response cache. These are required: the service refuses to start rather
    # than fall back to a Redis that was never configured.
    REDIS_ENABLED: bool
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_DB: int
    REDIS_PASSWORD: str
    REDIS_SSL: bool
    REDIS_KEY_PREFIX: str
    REDIS_CACHE_TTL_DAYS: int
    # Counts how often each query is asked, in the same record as its answer.
    REDIS_QUERY_COUNTER_ENABLED: bool

    @property
    def redis_cache_ttl_seconds(self) -> int:
        return self.REDIS_CACHE_TTL_DAYS * 24 * 60 * 60


@lru_cache
def get_settings():
    return Settings()