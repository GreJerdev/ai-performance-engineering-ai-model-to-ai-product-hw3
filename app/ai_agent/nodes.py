from collections.abc import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.ai_agent.prompts import DEFAULT_SYSTEM_PROMPT, MAX_ITERATIONS_PROMPT, ROUTER_PROMPT
from app.ai_agent.state import AgentState

from app.tools.bitext_tools import create_bitext_tools
from app.data_layer.bitext_data_layer import BitextDataLayer

DATA_LAYER = BitextDataLayer("bitext_data.csv")
BITEXT_TOOLS = create_bitext_tools(DATA_LAYER)
TOOLS_BY_NAME = {tool.name: tool for tool in BITEXT_TOOLS}
MAX_ITERATIONS = 15

def _invoke_with_tools(llm, messages: list, max_iterations: int = 5) -> AIMessage:
    llm_with_tools = llm.bind_tools(BITEXT_TOOLS)
    response = llm_with_tools.invoke(messages)
    messages = [*messages, response]

    iteration = 0
    while getattr(response, "tool_calls", None) and iteration < max_iterations:
        for tool_call in response.tool_calls:
            tool = TOOLS_BY_NAME[tool_call["name"]]
            tool_result = tool.invoke(tool_call["args"])
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )
        print("--------------------------------")    
        print(f"Use tool for {tool_call['name']} with args {tool_call['args']}")    
        print("++++++++++++++++++++++++++++++++")
        #print(str(tool_result))
        #print("################################")
        response = llm_with_tools.invoke(messages)
        iteration += 1
        messages.append(response)
    else:
        if iteration == max_iterations:
            print(f"Max iterations reached after {iteration} iterations")
            response = llm_with_tools.invoke(messages)
        

    return response

def enter_node(state: AgentState) -> dict:
    print("------------------Enter node------------------")
    last_message = state["messages"][-1]
    print(f"Last message: {last_message}")
    print("------------------Enter node------------------")
        # Use ChatOpenAI's invoke with tools, system prompt, etc
        # The input for ChatOpenAI should be a list of messages
       
    return state


def route_query_node(llm) -> Callable[[AgentState], dict]:
    """
    A LangGraph route node which takes the latest human message and chooses a branch:
      - 'structured_query_node' if the message is suited for the dataset tools (bitext_tools)
      - 'unstructured_query_node' if the query is informational but not processable by tools
      - 'out_of_scope_query_node' otherwise
    The decision is made using an LLM prompt describing available fields, categories, and intents.
    Returns: dict with a 'route' key whose value is a string node name.
    """
    def route(state: AgentState) -> dict:
        print("------------------Route node------------------")
        message = state["messages"][-1]
        last_message = message.content[-1]["text"]
        print(f"Last message: {last_message}")
        print("------------------Route node------------------")
        router_prompt_filled = ROUTER_PROMPT.replace("%%%query%%%", last_message)
        llm.bind_tools(BITEXT_TOOLS)
        response = llm.invoke(
            [
                {"role": "system", "content": "You are a helpful AI query router."},
                {"role": "user", "content": router_prompt_filled},
            ]
            # No tools required for routing
        )
        text = response.content if hasattr(response, "content") else str(response)
        branch = text.strip().split()[0].lower()
        valid_branches = {
            "structured_query_node",
            "unstructured_query_node",
            "out_of_scope_query_node",
        }
        route_name = branch if branch in valid_branches else "out_of_scope_query_node"
        return route_name
    return route

def structured_query_node(llm) -> Callable[[AgentState], dict]:
    def structured_execution(state: AgentState) -> dict:
        # Use llm and tools from bitext_tools.py to get query results
        last_message = state["messages"][-1]
        print("------------------Structured query node------------------")
        print(f"Last message: {last_message}")
        print("------------------Structured query node------------------")

        response = _invoke_with_tools(
            llm, [last_message ],
        )

        return {"messages": [response if isinstance(response, AIMessage) else AIMessage(content=str(response))]}
    return structured_execution

def unstructured_query_node(llm) -> Callable[[AgentState], dict]:
    def plan_execution(state: AgentState) -> dict:
        messages = state["messages"]
        iterations_number = 0
        # last_message = messages[-1]
        # for tool_call in last_message.tool_calls:
        #     tool = TOOLS_BY_NAME[tool_call["name"]]
        #     tool_result = tool.invoke(tool_call["args"])
        #     messages.append(
        #         ToolMessage(
        #             content=str(tool_result),
        #             tool_call_id=tool_call["id"],
        #         )
        #     )
        # print("--------------plan_execution--------------")    
        # print(f"Use tool for {tool_call['name']} with args {tool_call['args']}")    
        # print("++++++++++++++plan_execution++++++++++++++++++")
        return {"iterations_number":iterations_number}
    return plan_execution

def out_of_scope_query_node(llm) -> Callable[[AgentState], dict]:
    def out_of_scope_query_node(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        # Compose an explanation for why this query is deemed out of scope.
        explain_prompt = (
            f"You are an AI query router. "
            "A user submitted the following query:\n"
            f"\"{last_message.content}\"\n\n"
            "Using the following router instructions:\n"
            f"{ROUTER_PROMPT.replace('%%%query%%%', last_message.content)}\n\n"
            "Explain why this user query is considered out of scope for the possible categories and provide a short, clear reasoning to the user."
        )
        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that explains what types of questions are considered out of scope for the available AI services.",
                },
                {"role": "user", "content": explain_prompt},
            ]
        )
        return {"messages": [response if isinstance(response, AIMessage) else AIMessage(content=str(response))]}
 
    return out_of_scope_query_node

def plan_execution_node(llm) -> Callable[[AgentState], dict]:
    def plan_execution(state: AgentState) -> dict:
        messages = state["messages"]
        iterations_number = state["iterations_number"]
        print("------------------Plan execution node------------------")
        print(f"Last message: {messages[-1]}")
        print("------------------Plan execution node------------------")
        last_message = messages[-1]
        for tool_call in last_message.tool_calls:
            tool = TOOLS_BY_NAME[tool_call["name"]]
            tool_result = tool.invoke(tool_call["args"])
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )
            print("--------------plan_execution--------------")    
            print(f"Use tool for {tool_call['name']} with args {tool_call['args']}")    
            print("++++++++++++++plan_execution++++++++++++++++++")
        return {"messages": messages, "iterations_number": iterations_number + 1}
    return plan_execution

def plan_thinking_node(llm) -> Callable[[AgentState], dict]:
    def plan_thinking(state: AgentState) -> dict:
        # The LLM reasons about the state and can output a tool call
        prompt = "You must think step-by-step before answering. Use tools if necessary."
        response = llm.bind_tools(BITEXT_TOOLS).invoke([HumanMessage(content=prompt)] + state["messages"])
        return {"messages": [response if isinstance(response, AIMessage) else AIMessage(content=str(response))]}
    return plan_thinking


def think_router_node(llm) -> Callable[[AgentState], dict]:
    def think_router(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        iterations_number = state["iterations_number"]
        if iterations_number > MAX_ITERATIONS:
            return "max_iterations" # Go to act_node
        elif getattr(last_message, "tool_calls", []):
            return "plan_execution" # Go to act_node
        return "end_node" # Finish node
    return think_router

def messages_to_text(messages):
    return "\n\n".join(
        f"{m.type.upper()}: {m.content}"
        for m in messages
        if getattr(m, "content", None)
    )

def max_iterations_node(llm) -> Callable[[AgentState], dict]:
    def max_iterations(state: AgentState) -> dict:
        final_messages =[ SystemMessage(content=MAX_ITERATIONS_PROMPT), *state["messages"]]
        response = llm.invoke(final_messages)
        return {"messages": [response if isinstance(response, AIMessage) else AIMessage(content=str(response))]}
    
    return max_iterations

def end_node(state: AgentState) -> dict:
    return {"messages": state["messages"]}