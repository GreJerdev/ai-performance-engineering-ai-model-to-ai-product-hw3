from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from app.ai_agent.graph import build_graph


class LangGraphAgent:
    def __init__(self, llm, checkpointer=None):
        self.llm = llm
        
        self.graph = build_graph(self.llm, checkpointer=checkpointer)

    def invoke(self, message: str, config: RunnableConfig | None = None,) -> str:
        result = self.graph.invoke({"messages": [HumanMessage(content=message)]}, config)
        return result["messages"][-1].content
