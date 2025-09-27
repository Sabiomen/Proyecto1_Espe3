from typing import List, Dict, Any, Protocol

class Provider(Protocol):
    name: str

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Ejecuta un chat con un LLM.
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        kwargs: parámetros adicionales (temperature, max_tokens, etc.)
        return: {"text": str, "usage": dict, "meta": dict}
        """