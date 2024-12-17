import logging
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OLLAMA_BASE_URL: Optional[str] = "http://localhost:11434"
    OPENAI_API_KEY: Optional[str] = None
    LLM_PROVIDER: str = "ollama"
    MODEL_NAME: str = "mistral-nemo"
    MODEL_TEMPERATURE: float = 0.3
    DATA_DIR: str = "/Users/username/Documents/"
    PROCESSED_DATA_DIR: str = "./processed_data"
    SERPER_API_KEY: Optional[str] = "92160b0fa55442ef5d0b36429129623d1bfc89e0"
    LOGGING_LEVEL: int = logging.DEBUG


settings = Settings()
