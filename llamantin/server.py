import asyncio
import json
from enum import Enum
from typing import Any, Dict
from uuid import UUID

from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

from .config import Settings, settings
from .llm import LLMProvider
from .websearchagent import WebSearchAgent


# Models
class AgentType(str, Enum):
    WEB_SEARCH = "web_search"


class AgentRequest(BaseModel):
    agent_type: AgentType
    query: str


# Agent Factory
class AgentFactory:
    @staticmethod
    async def create_agent(agent_type: AgentType, settings: Settings = settings):
        if agent_type == AgentType.WEB_SEARCH:
            # Initialize your LLM here
            llm = LLMProvider.create_llm(settings=settings)
            return WebSearchAgent(llm=llm, settings=settings)
        # Add other agent types here
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

# FastAPI app
app = FastAPI()


async def process_agent_query(task_id: UUID, agent_type: AgentType, query: str):
    print("Processing agent query")
    try:
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
