from abc import ABC
from langchain_community.utilities import (
    GoogleSerperAPIWrapper,
    DuckDuckGoSearchAPIWrapper,
)
from typing_extensions import Literal

from .agent import Agent
from .config import Settings, settings


class WebSearchAgent(ABC):
    def __init__(self, llm, settings: Settings = settings):
        pass

    async def search(self, query: str) -> str:
        results = []
        async for value in self.agent.graph.astream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Search for {query} and create an engaging report with the most relevant findings.",
                    }
                ]
            },
            stream_mode="values",
        ):
            if "messages" in value:
                results.append(value["messages"][-1]["content"])

        return results[-1]


class GoogleSearchAgent(WebSearchAgent):
    def __init__(self, llm, settings: Settings = settings):
        async def search_query(
            query: str,
            type: Literal["news", "search", "places", "images"] = "search",
        ) -> str:
            """
            Search for a query using GoogleSerperAPIWrapper.

            Args:
            query: The search query string.
            type: The type of search to be performed, can be "news", "search", "places", or "images".

            Returns:
            dict: The search results.
            """
            # Perform the search
            results = await GoogleSerperAPIWrapper(
                serper_api_key=settings.SERPER_API_KEY,
                type=type,
            ).arun(query)

            return results

        self.agent = Agent(llm, search_query, "search_query")


class DuckSearchAgent(WebSearchAgent):
    def __init__(self, llm, settings: Settings = settings):
        async def search_query(
            query: str,
            source: Literal["news", "text"] = "text",
        ) -> str:
            """
            Search for a query using DuckDuckGoSearchAPIWrapper.

            Args:
            query: The search query string.
            type: The type of search to be performed, can be "news", or "text".

            Returns:
            dict: The search results.
            """
            # Perform the search
            results = DuckDuckGoSearchAPIWrapper(
                source=source,
            ).run(query)

            return results

        self.agent = Agent(llm, search_query, "search_query")
