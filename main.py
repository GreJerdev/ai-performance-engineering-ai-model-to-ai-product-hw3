from app.data_layer.bitext_data_layer import BitextDataLayer
from app.llms.llm_factory import LLMFactory
from app.ai_agent.agent import LangGraphAgent
from app.llms.llm_factory import Models
from dotenv import load_dotenv

def main():
    load_dotenv()
    data_layer = BitextDataLayer("bitext_data.csv")
    data_layer.get_data()
    #llm = LLMFactory.get_llm(Models.GPT_OSS_120B)
    llm = LLMFactory.get_llm(Models.NEMOTRON_3_NANO_OMNI)
    agent = LangGraphAgent(llm)
    #response = agent.invoke("Show me 3 examples from the SHIPPING intent.")
    response = agent.invoke("How do customer service representatives typically respond to cancellation requests?")
    #response = agent.invoke("Summarize the FEEDBACK category.")
    #response = agent.invoke("Show me 3 examples from the REFUND category")

    #response = agent.invoke("Write me a poem about customer service.")
    print(response)


if __name__ == "__main__":
    main()
    # import os
    # from openai import OpenAI
    # from langchain_openai import ChatOpenAI

    # llm = ChatOpenAI(
    #     model="openai/gpt-oss-120b",
    #     base_url="https://api.tokenfactory.nebius.com/v1/",
    #     api_key=os.environ.get("NEBIUS_API_KEY")
    # )

    # messages = [
    #     {"role": "system", "content": "SYSTEM_PROMPT"},
    #     {"role": "user", "content": "USER_MESSAGE"}
    # ]

    # response = llm.invoke(messages)
    # print(response)