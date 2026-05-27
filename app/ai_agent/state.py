from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_profile: Annotated[list[AnyMessage], add_messages]
    last_user_request: NotRequired[str]
    iterations_number: NotRequired[int]
