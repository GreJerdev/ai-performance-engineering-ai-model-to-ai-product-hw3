from enum import Enum
from app.providers.llm_base import LLMDeepSeekAIv4Pro

class Models(Enum):
    DEEPSEEK_AI_V4_PRO = "deepseek-ai/deepseek-ai-v4-pro"

class LLMFactory:

    

    
    def __init__(self):
       pass
       


    @classmethod
    def get_llm(cls, model: Models):
       if model == Models.DEEPSEEK_AI_V4_PRO:
        return LLMDeepSeekAIv4Pro()
       else:
        raise ValueError(f"Model {model} not supported")