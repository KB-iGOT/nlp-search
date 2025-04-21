import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../.env"
        )
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