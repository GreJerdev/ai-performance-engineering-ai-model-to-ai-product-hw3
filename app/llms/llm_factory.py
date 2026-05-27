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

    API_KEY = os.getenv("NEBIUS_API_KEY")

    def __init__(self):
        print("__init__",os.getenv("NEBIUS_API_KEY"))

    @classmethod
    def get_llm(cls, model: Models) -> ChatOpenAI:
        if model in Models:
            print(f"Model {model.value} supported")
            print(cls.API_KEY[0:5] + "..." + cls.API_KEY[-5:])
            return ChatOpenAI(
                model=model.value, base_url=cls.BASE_URL, api_key=cls.API_KEY
            )
        else:
            raise ValueError(f"Model {model} not supported")
