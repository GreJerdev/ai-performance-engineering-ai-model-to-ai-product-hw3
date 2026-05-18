from collections.abc import Callable

from langchain_core.messages import AIMessage

from app.ai_agent.prompts import DEFAULT_SYSTEM_PROMPT
from app.ai_agent.state import AgentState


def create_call_model_node(llm) -> Callable[[AgentState], dict]:
    def call_model(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        response = llm.generate_response(
            prompt=last_message.content,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
        return {"messages": [AIMessage(content=response)]}

    return call_model
