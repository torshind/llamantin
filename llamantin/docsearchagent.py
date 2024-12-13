from .agent import Agent
from .collector import Collector
from .config import Settings, settings


class DocSearchAgent:
    def __init__(
        self,
        llm,
        settings: Settings = settings,
        collector: Collector = None,
        cutoff: float = 0.25,
    ):
        self.collector = collector
        self.cutoff = cutoff
        self.agent = Agent(llm, {}, "")

    async def search(self, query: str) -> str:
        results = []

        doc_tuples = (
            await self.collector.vector_db.asimilarity_search_with_relevance_scores(
                query=query
            )
        )

        doc_tuples[:] = [
            doc_tuple for doc_tuple in doc_tuples if doc_tuple[1] >= self.cutoff
        ]

        print(doc_tuples, flush=True)

        context = "\n\n".join([doc_tuple[0].page_content for doc_tuple in doc_tuples])

        async for value in self.agent.graph.astream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Answer the {query} using the following context: '{context}'.",
                    }
                ]
            },
            stream_mode="values",
        ):
            if "messages" in value:
                results.append(value["messages"][-1]["content"])

        return results[-1]
