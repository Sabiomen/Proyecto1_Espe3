from rag.retrieve import Retriever
import time
from rag.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
import re

class RAGPipeline:
    def __init__(self, retriever: Retriever, provider):
        self.retriever = retriever
        self.provider = provider

    def rewrite_query(self, query: str) -> str:
        """Por ahora passthrough (sin reescritura)."""
        return query

    def synthesize(self, query: str, top_k=4, max_tokens=512, temperature=0.0):
        start_total = time.time()

        q_rewritten = self.rewrite_query(query)

        start_retrieve = time.time()
        hits = self.retriever.query(q_rewritten, top_k=top_k)
        latency_retrieve = time.time() - start_retrieve

        snippets = "\n\n".join([
            f"[{h['title']}, p{h.get('page')}] {h['text']}" for h in hits
        ])

        user_prompt = USER_PROMPT_TEMPLATE.format(question=query, snippets=snippets)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        start_llm = time.time()
        answer = self.provider.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        latency_llm = time.time() - start_llm

        total_latency = time.time() - start_total

        # Extraer tokens usados si el proveedor lo devuelve en algún atributo response.usage
        tokens_prompt = None
        tokens_completion = None
        tokens_total = None

        # Si tu método chat devuelve también usage, por ejemplo:
        # response = self.provider.chat(...)
        # tokens_prompt = response['usage']['prompt_tokens']
        # tokens_completion = response['usage']['completion_tokens']
        # tokens_total = response['usage']['total_tokens']

        # Por ahora, si no tienes ese dato, puedes estimar con un contador de palabras o usar un wrapper para OpenAI completions que devuelva uso tokens.

        return {
            "answer": answer,
            "hits": hits,
            "latency_retrieve": latency_retrieve,
            "latency_llm": latency_llm,
            "latency_total": total_latency,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "tokens_total": tokens_total,
        }