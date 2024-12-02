from collections.abc import AsyncIterator
from typing import Union

from ollama import AsyncClient, Client
from ollama._types import ChatResponse

from .config import Settings, settings


class Ollama:
    def __init__(self, model: str, base_url: str, temperature: float):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.client = Client(host=base_url)
        self.aclient = AsyncClient(host=base_url)

    async def achat(
        self, *args, **kwargs
    ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
        return await self.aclient.chat(
            model=self.model,
            *args,
            options={
                "temperature": self.temperature,
            },
            **kwargs,
        )

    def chat(self, *args, **kwargs) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
        return self.client.chat(
            model=self.model,
            *args,
            options={
                "temperature": self.temperature,
            },
            **kwargs,
        )


class LLMProvider:
    @staticmethod
    def create_llm(settings: Settings = settings):
        if settings.LLM_PROVIDER == "ollama":
            return Ollama(
                model=settings.MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=settings.MODEL_TEMPERATURE,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
