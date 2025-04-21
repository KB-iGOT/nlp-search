from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model : str
    nlp_search_instruction_prompt: str
    nlp_search_example_prompt: str
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