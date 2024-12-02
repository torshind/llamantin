import pytest
from ollama._types import ChatResponse
from vcr import VCR

from llamantin.config import Settings
from llamantin.llm import LLMProvider

vcr = VCR(
    cassette_library_dir="tests/cassettes",
    record_mode="once",
    match_on=["method", "scheme", "host", "port", "path", "query", "body"],
)


@pytest.fixture
def ollama_settings():
    return Settings(
        LLM_PROVIDER="ollama",
        MODEL_NAME="mistral-nemo",
        OLLAMA_BASE_URL="http://localhost:11434",
        MODEL_TEMPERATURE=0.0,
    )


def test_create_llm_basic(ollama_settings):
    """Test basic LLM creation and configuration"""
    llm = LLMProvider.create_llm(settings=ollama_settings)
    assert llm.model == "mistral-nemo"
    assert llm.base_url == "http://localhost:11434"


@pytest.mark.vcr()
def test_llm_chat(ollama_settings):
    """Test LLM can make basic calls"""
    llm = LLMProvider.create_llm(settings=ollama_settings)
    messages = [{"role": "system", "content": "Say hello"}]
    response = llm.chat(messages=messages)
    assert isinstance(response, ChatResponse)
    assert len(response.message.content) > 0


@pytest.mark.vcr()
def test_llm_chat_streaming(ollama_settings):
    """Test LLM can make basic calls with streaming"""
    llm = LLMProvider.create_llm(settings=ollama_settings)
    messages = [{"role": "system", "content": "Say hello"}]
    response_stream = llm.chat(messages=messages, stream=True)

    collected_messages = []
    for response in response_stream:
        assert isinstance(response, ChatResponse)
        collected_messages.append(response.message.content)

    assert len(collected_messages) > 0
    assert any(len(message) > 0 for message in collected_messages)


def test_invalid_provider():
    """Test error handling for invalid provider"""
    invalid_settings = Settings(LLM_PROVIDER="invalid", MODEL_NAME="test")
    with pytest.raises(ValueError) as exc_info:
        LLMProvider.create_llm(settings=invalid_settings)
    assert "Unsupported LLM provider" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_llm_achat(ollama_settings):
    """Test LLM can make basic async calls"""
    llm = LLMProvider.create_llm(settings=ollama_settings)
    messages = [{"role": "system", "content": "Say hello"}]
    response = await llm.achat(messages=messages)
    assert isinstance(response, ChatResponse)
    assert len(response.message.content) > 0


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_llm_achat_streaming(ollama_settings):
    """Test LLM can make basic calls with streaming"""
    llm = LLMProvider.create_llm(settings=ollama_settings)
    messages = [{"role": "system", "content": "Say hello"}]
    response_stream = await llm.achat(messages=messages, stream=True)

    collected_messages = []
    async for response in response_stream:
        assert isinstance(response, ChatResponse)
        collected_messages.append(response.message.content)

    assert len(collected_messages) > 0
    assert any(len(message) > 0 for message in collected_messages)
