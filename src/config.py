from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model : str
    PROMPT_VERSION: str = "latest"


    project: str
    location: str
    max_output_tokens: int
    temperature: float
    top_p: float
    top_k: float
    GOOGLE_APPLICATION_CREDENTIALS: str
    max_search_len: int
    class Config:
        env_file = ".env"