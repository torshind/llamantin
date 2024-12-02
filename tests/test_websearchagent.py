import pytest

from llamantin.config import settings
from llamantin.llm import LLMProvider
from llamantin.websearchagent import WebSearchAgent


@pytest.fixture
def web_agent():
    llm = LLMProvider.create_llm(settings=settings)
    return WebSearchAgent(llm, settings)


@pytest.mark.asyncio
async def test_websearchagent_basic_search(web_agent):
    query = "What is Python programming language?"
    result = await web_agent.search(query)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Python" in result


@pytest.mark.asyncio
async def test_websearchagent_complex_query(web_agent):
    query = "Latest developments in quantum computing 2024"
    result = await web_agent.search(query)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "quantum" in result.lower()


@pytest.mark.asyncio
async def test_websearchagent_empty_query(web_agent):
    with pytest.raises(Exception):
        await web_agent.search("")


@pytest.mark.asyncio
async def test_websearchagent_special_characters(web_agent):
    query = "Python @ programming #language"
    result = await web_agent.search(query)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_websearchagent_long_query(web_agent):
    query = "Explain in detail the process of photosynthesis in plants and how it relates to cellular respiration"
    result = await web_agent.search(query)
    assert isinstance(result, str)
    assert len(result) > 0
