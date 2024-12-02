import operator
from typing import Annotated, Literal

from langchain_core.messages import ChatMessage
from langchain_core.outputs.llm_result import ChatGeneration, LLMResult
from langchain_core.runnables.config import (
    ensure_config,
    get_callback_manager_for_config,
)
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, operator.add]


class Agent:
    def __init__(self, llm, tool, tool_name):
        self.llm = llm
        self.tool = tool
        # define mapping to look up functions when running tools
        self.function_name_to_function = {tool_name: tool}

        workflow = StateGraph(State)
        workflow.add_edge(START, "model")
        workflow.add_node("model", self._call_model)
        workflow.add_node("tools", self._call_tools)
        workflow.add_conditional_edges("model", self._should_continue)
        workflow.add_edge("tools", "model")
        self.graph = workflow.compile()

    async def _call_model(self, state, config=None):
        config = ensure_config(config | {"tags": ["agent_llm"]})
        callback_manager = get_callback_manager_for_config(config)
        messages = state["messages"]

        llm_run_manager = callback_manager.on_chat_model_start({}, [messages])[0]
        response = await self.llm.achat(
            messages=messages,
            tools=[self.tool],
            stream=False,
        )

        role = None
        response_content = ""

        tool_call_function_name = None
        tool_call_function_arguments = ""

        message = response.message
        if message.role is not None:
            role = message.role

        if message.content:
            response_content = message.content
            llm_run_manager.on_llm_end(
                LLMResult(
                    generations=[
                        [
                            ChatGeneration(
                                message=ChatMessage(
                                    role=role,
                                    content=response_content,
                                )
                            )
                        ]
                    ]
                )
            )

        if message.tool_calls:
            # note: for simplicity we're only handling a single tool call here
            if message.tool_calls[0].function.name is not None:
                tool_call_function_name = message.tool_calls[0].function.name
                tool_call_function_arguments = message.tool_calls[0].function.arguments

            llm_run_manager.on_llm_end(
                LLMResult(
                    generations=[
                        [
                            ChatGeneration(
                                message=ChatMessage(
                                    role=role,
                                    content="",
                                    additional_kwargs={
                                        "tool_calls": [
                                            message.tool_calls[0].model_dump()
                                        ]
                                    },
                                )
                            )
                        ]
                    ]
                )
            )

        if tool_call_function_name is not None:
            tool_calls = [
                {
                    "function": {
                        "name": tool_call_function_name,
                        "arguments": tool_call_function_arguments,
                    },
                    "type": "function",
                }
            ]
        else:
            tool_calls = None

        response_message = {
            "role": role,
            "content": response_content,
            "tool_calls": tool_calls,
        }
        return {"messages": [response_message]}

    async def _call_tools(self, state):
        messages = state["messages"]

        tool_call = messages[-1]["tool_calls"][0]
        function_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]

        function_response = await self.function_name_to_function[function_name](**arguments)
        tool_message = {
            "role": "tool",
            "name": function_name,
            "content": str(function_response),
        }
        return {"messages": [tool_message]}

    def _should_continue(self, state) -> Literal["tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message["tool_calls"]:
            return "tools"
        return END
