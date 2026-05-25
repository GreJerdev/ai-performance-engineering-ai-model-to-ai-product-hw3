from langgraph.graph import END, START, StateGraph
from app.ai_agent.nodes import end_node, max_iterations_node, plan_execution_node, plan_thinking_node, route_query_node, structured_query_node, think_router_node, unstructured_query_node, out_of_scope_query_node,enter_node
from app.ai_agent.state import AgentState




def build_graph(llm, checkpointer):
    graph = StateGraph(AgentState)
    graph.add_node("enter_node", enter_node)
    graph.add_node("structured_query_node",
                   structured_query_node(llm))
    graph.add_node("unstructured_query_node",
                   unstructured_query_node(llm))
    graph.add_node("out_of_scope_query_node",out_of_scope_query_node(llm))

    graph.add_node("plan_thinking", plan_thinking_node(llm))
    graph.add_node("plan_execution", plan_execution_node(llm))
    graph.add_node("max_iterations", max_iterations_node(llm))
    graph.add_node("end_node", end_node)

    graph.add_edge("unstructured_query_node", "plan_thinking")

    graph.add_conditional_edges("plan_thinking", think_router_node(llm))
    graph.add_edge("plan_execution", "plan_thinking")

    graph.add_edge(START, "enter_node")
    graph.add_conditional_edges("enter_node", route_query_node(llm))
    graph.add_edge("structured_query_node", END)
    graph.add_edge("unstructured_query_node", END)
    graph.add_edge("out_of_scope_query_node", END)
    graph.add_edge("max_iterations", END)
    return graph.compile(checkpointer=checkpointer)
