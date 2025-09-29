import os 
from typing import List, Dict 
from openai import OpenAI 
from providers.base import Provider 
class ChatGPTProvider(Provider): 

    def __init__(self, model: str = "openai/gpt-4.1-mini"): 
        self.client = OpenAI( api_key=os.getenv("OPENAI_API_KEY"), 
                             base_url="https://openrouter.ai/api/v1"
                            ) 
        self.model = model 

    @property 
    def name(self) -> str: 
        return "chatgpt" 

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return {
            "text": response.choices[0].message.content,
            "usage": response.usage
        }