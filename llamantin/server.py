import asyncio
import json
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Dict
from uuid import UUID

from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

from .collector import Collector
from .config import Settings, settings
from .docsearchagent import DocSearchAgent
from .llm import LLMProvider
from .websearchagent import DuckSearchAgent, GoogleSearchAgent


# Models
class AgentType(str, Enum):
    GOOGLE_SEARCH = "google_search"
    DUCK_SEARCH = "duck_search"
    DOC_SEARCH = "doc_search"


class AgentRequest(BaseModel):
    agent_type: AgentType
    query: str


# Agent Factory
class AgentFactory:
    @staticmethod
    async def create_agent(agent_type: AgentType, settings: Settings = settings):
        llm = LLMProvider.create_llm(settings=settings)
        if agent_type == AgentType.GOOGLE_SEARCH:
            return GoogleSearchAgent(llm=llm, settings=settings)
        elif agent_type == AgentType.DUCK_SEARCH:
            return DuckSearchAgent(llm=llm, settings=settings)
        elif agent_type == AgentType.DOC_SEARCH:
            return DocSearchAgent(llm=llm, settings=settings, collector=collector)
        raise ValueError(f"Unknown agent type: {agent_type}")


# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.connections: Dict[UUID, Dict[str, Any]] = {}

    async def connect(self, task_id: UUID, websocket: WebSocket):
        await websocket.accept()
        self.connections[task_id] = {"websocket": websocket, "queue": asyncio.Queue()}
        print("WebSocket connected")
        print("Task ID:", task_id)
        print("Status: {}".format(self.connections))

    async def disconnect(self, task_id: UUID):
        if task_id in self.connections:
            del self.connections[task_id]

    async def send_update(self, task_id: UUID, data: dict):
        if task_id in self.connections:
            await self.connections[task_id]["websocket"].send_text(json.dumps(data))


manager = ConnectionManager()
collector = Collector(settings.DATA_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Starting up...")
    collector.initialize_database_in_background()
    collector.start()

    yield

    # Shutdown logic
    print("Shutting down...")
    collector.stop()


# FastAPI app
app = FastAPI(lifespan=lifespan)


async def process_agent_query(task_id: UUID, agent_type: AgentType, query: str):
    print("Processing agent query")
    try:
        if agent_type == AgentType.DOC_SEARCH and not collector.is_initialized():
            await manager.send_update(
                task_id,
                {
                    "status": "waiting",
                    "message": "Database is initializing, please wait...",
                },
            )
            return

        agent = await AgentFactory.create_agent(agent_type)
        result = await agent.search(query)
        await manager.send_update(task_id, {"status": "completed", "result": result})
    except Exception as e:
        await manager.send_update(task_id, {"status": "failed", "error": str(e)})
    finally:
        await manager.disconnect(task_id)


@app.websocket("/ws/{task_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    task_id: UUID,
):
    await manager.connect(task_id, websocket)

    try:
        # Get the initial request from the WebSocket
        data = await websocket.receive_json()

        # Start the agent task
        await process_agent_query(task_id, data["agent_type"], data["query"])

    except Exception as e:
        await manager.send_update(task_id, {"status": "error", "message": str(e)})
    finally:
        await manager.disconnect(task_id)
