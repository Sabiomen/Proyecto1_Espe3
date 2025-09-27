import os
from openai import OpenAI
from typing import List, Dict, Any
from providers.base import Provider

class DeepSeekProvider(Provider):

   def __init__(self, model: str = "deepseek-chat"): 
    self.client = OpenAI( api_key=os.getenv("DEEPSEEK_API_KEY"), 
                         base_url="https://api.deepseek.com"
                        ) 
    self.model = model 

    @property 
    def name(self) -> str: 
        return "deepseek" 

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str: 
        response = self.client.chat.completions.create( 
            model=self.model, 
            messages=messages,
            **kwargs
            ) 
        return response.choices[0].message.content