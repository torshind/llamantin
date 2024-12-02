from langchain_community.utilities import GoogleSerperAPIWrapper

from .agent import Agent
from .config import Settings, settings


class WebSearchAgent:
    def __init__(self, llm, settings: Settings = settings):
        async def search_query(query: str) -> dict:
            """
            Search for a query using GoogleSerperAPIWrapper.

            Args:
            query: The search query string.

            Returns:
            dict: The search results.
            """
            # Perform the search
            results = await GoogleSerperAPIWrapper(
                serper_api_key=settings.SERPER_API_KEY
            ).arun(query)

            return results

        self.agent = Agent(llm, search_query, "search_query")

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
