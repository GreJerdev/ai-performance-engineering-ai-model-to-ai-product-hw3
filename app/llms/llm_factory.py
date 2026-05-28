import os
from enum import Enum
from langchain_openai import ChatOpenAI


class Models(Enum):
    DEEPSEEK_AI_V4_PRO = "deepseek-ai/DeepSeek-V4-Pro"
    DEEPSEEK_V3 = "deepseek-ai/DeepSeek-R1-0528"
    GPT_OSS_120B = "openai/gpt-oss-120b"
    NEMOTRON_3_NANO_OMNI = "nvidia/Nemotron-3-Nano-Omni"


class LLMSizes(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    BIG = "big"


class LLMFactory:
    BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

    def __init__(self):
        print("LLMFactory.__init__")

    @classmethod
    def get_llm(cls, model: Models) -> ChatOpenAI:
        api_key = os.getenv("NEBIUS_API_KEY")
        if api_key is None:
            raise ValueError("NEBIUS_API_KEY environment variable is not set")

        if model in Models:
            print(f"Model {model.value} supported")
            print(cls)

            print(api_key[0:5] + "..." + api_key[-5:])
            chat_model =  ChatOpenAI(
                model=model.value, base_url=cls.BASE_URL, api_key=api_key
            )
            print("#"*100)
            print(f"Model {model.value} created")
            print("#"*100)
            return chat_model
        else:
            raise ValueError(f"Model {model} not supported")
