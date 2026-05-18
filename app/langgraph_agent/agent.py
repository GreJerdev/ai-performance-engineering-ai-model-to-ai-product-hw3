from langchain_core.messages import HumanMessage

from app.langgraph_agent.graph import build_graph


class LangGraphAgent:
    def __init__(self, llm):
        self.llm = llm
        self.graph = build_graph(self.llm)

    def invoke(self, message: str) -> str:
        result = self.graph.invoke({"messages": [HumanMessage(content=message)]})
        return result["messages"][-1].content
