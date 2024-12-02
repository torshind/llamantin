import pytest
from langchain_community.utilities import GoogleSerperAPIWrapper

from llamantin.agent import Agent
from llamantin.config import settings
from llamantin.llm import LLMProvider

client = LLMProvider.create_llm(settings=settings)


async def search_query(query: str) -> dict:
    """
    Search for a query using GoogleSerperAPIWrapper.

    Args:
      query: The search query string.

    Returns:
      dict: The search results.
    """
    # Perform the search
    results = await GoogleSerperAPIWrapper(serper_api_key=settings.SERPER_API_KEY).arun(
        query
    )

    return results


async def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
      a: The first integer number
      b: The second integer number

    Returns:
      int: The sum of the two numbers
    """
    return a + b


@pytest.mark.asyncio(loop_scope="session")
async def test_add():
    agent = Agent(client, add_two_numbers, "add_two_numbers")
    results = []
    async for value in agent.graph.astream(
        {"messages": [{"role": "user", "content": "What is 10 + 10?"}]},
        stream_mode="values",
    ):
        if "messages" in value:
            results.append(value["messages"][-1]["content"])
    assert len(results) > 0
    assert all(isinstance(result, str) for result in results)


@pytest.mark.asyncio(loop_scope="session")
async def test_search():
    agent = Agent(client, search_query, "search_query")
    results = []
    async for value in agent.graph.astream(
        {"messages": [{"role": "user", "content": "Search for news about AI agents"}]},
        stream_mode="values",
    ):
        if "messages" in value:
            results.append(value["messages"][-1]["content"])
    assert len(results) > 0
    assert all(isinstance(result, str) for result in results)
