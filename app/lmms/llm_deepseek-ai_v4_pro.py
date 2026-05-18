from app.providers.llm_base import LLMBase

class LLMDeepSeekAIv4Pro(LLMBase):
    
    MODEL_NAME="deepseek-ai/deepseek-ai-v4-pro"
    def __init__(self):
        super().__init__()

    def generate_response(self, prompt: str, system_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content