from langgraph.graph import END, START, StateGraph

from app.ai_agent.nodes import create_call_model_node
from app.ai_agent.state import AgentState


def build_graph(llm):
    graph = StateGraph(AgentState)
    graph.add_node("agent", create_call_model_node(llm))
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()
