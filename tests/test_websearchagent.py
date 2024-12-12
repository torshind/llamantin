import pytest

from llamantin.config import settings
from llamantin.llm import LLMProvider
from llamantin.websearchagent import DuckSearchAgent, GoogleSearchAgent


@pytest.fixture
def google_agent():
    llm = LLMProvider.create_llm(settings=settings)
    return GoogleSearchAgent(llm, settings)


@pytest.fixture
def duck_agent():
    llm = LLMProvider.create_llm(settings=settings)
    return DuckSearchAgent(llm, settings)


@pytest.mark.parametrize("agent_fixture", ["google_agent", "duck_agent"])
@pytest.mark.asyncio
async def test_websearchagent_basic_search(request, agent_fixture):
    agent = request.getfixturevalue(agent_fixture)
    query = "What is Python programming language?"
    result = await agent.search(query)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Python" in result


@pytest.mark.parametrize("agent_fixture", ["google_agent", "duck_agent"])
@pytest.mark.asyncio
async def test_websearchagent_complex_query(request, agent_fixture):
    agent = request.getfixturevalue(agent_fixture)
    query = "Latest developments in quantum computing 2024"
    result = await agent.search(query)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "quantum" in result.lower()


@pytest.mark.parametrize("agent_fixture", ["google_agent", "duck_agent"])
@pytest.mark.asyncio
async def test_websearchagent_empty_query(request, agent_fixture):
    agent = request.getfixturevalue(agent_fixture)
    with pytest.raises(Exception):
        await agent.search("")


@pytest.mark.parametrize("agent_fixture", ["google_agent", "duck_agent"])
@pytest.mark.asyncio
async def test_websearchagent_special_characters(request, agent_fixture):
    agent = request.getfixturevalue(agent_fixture)
    query = "Python @ programming #language"
    result = await agent.search(query)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.parametrize("agent_fixture", ["google_agent", "duck_agent"])
@pytest.mark.asyncio
async def test_websearchagent_long_query(request, agent_fixture):
    agent = request.getfixturevalue(agent_fixture)
    query = "Explain in detail the process of photosynthesis in plants and how it relates to cellular respiration"
    result = await agent.search(query)
    assert isinstance(result, str)
    assert len(result) > 0
