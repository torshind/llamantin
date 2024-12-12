from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVER_URL: Optional[str] = "localhost:3000"


settings = Settings()
