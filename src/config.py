from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MODEL : str = "gemini-2.5-flash"
    PROMPT_VERSION: str = "latest"


    PROJECT: str
    LOCATION: str
    MAX_OUTPUT_TOKENS: int
    TEMPERATURE: float
    TOP_P: float
    TOP_K: float
    GOOGLE_APPLICATION_CREDENTIALS: str
    MAX_SEARCH_LEN: int
    class Config:
        env_file = ".env"