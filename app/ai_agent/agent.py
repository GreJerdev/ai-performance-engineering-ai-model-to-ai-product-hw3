from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from app.ai_agent.graph import build_graph
from app.data_layer.bitext_data_layer import BitextDataLayer
from app.llms.llm_factory import LLMFactory, LLMSizes, Models

load_dotenv()


def _warmup_data() -> None:
    data_layer = BitextDataLayer("bitext_data.csv")
    data_layer.get_data()


_warmup_data()


def get_llm() -> dict:
    print("#"*30)
    print("get_llm")
    print("#"*30)
    return {
        LLMSizes.SMALL:LLMFactory.get_llm(Models.GPT_OSS_120B),
        LLMSizes.MEDIUM:LLMFactory.get_llm(Models.GPT_OSS_120B),
        LLMSizes.BIG:LLMFactory.get_llm(Models.DEEPSEEK_AI_V4_PRO)
    }

#_llm = LLMFactory.get_llm(Models.GPT_OSS_120B)# LLMFactory.get_llm(Models.NEMOTRON_3_NANO_OMNI)

# Exported for LangGraph CLI (`langgraph dev`, Studio, deploy)
#graph = build_graph(get_llm())


class LangGraphAgent:

    def __init__(self, llm, checkpointer=None):
        self.llm = llm
        self.graph = build_graph(self.llm, checkpointer=checkpointer)
        print("#"*100)
        print("load langgrapg")
        print("#"*100)


    def invoke(
        self,
        message: str,
        config: RunnableConfig | None = None,
    ) -> str:
        print(f"------------------Invoking agent with message: {message}")
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config,
        )
        return result["messages"][-1].content


def start_agent():
    """Run a short scripted demo with SQLite checkpointing (non-CLI)."""
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as cp:
        agent = LangGraphAgent(get_llm(), cp)
        print("agent created ########")
        config = {"configurable": {"thread_id": "session-2"}}
        print(agent.invoke(
            "How do customer service representatives typically respond to cancellation requests?",
            config,
         ))
        print(agent.invoke("Same for order questions?", config))
    return agent
