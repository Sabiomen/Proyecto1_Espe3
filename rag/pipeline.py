from rag.retrieve import Retriever
import time
from rag.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
import re

class RAGPipeline:
    def __init__(self, retriever, provider):
        self.retriever = retriever
        self.provider = provider

    def rewrite_query(self, query: str) -> str:
        return query

    def synthesize(self, query: str, top_k=4, max_tokens=512, temperature=0.0, with_usage=False):
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
        if with_usage and hasattr(self.provider, "chat_with_usage"):
            response = self.provider.chat_with_usage(
                messages, max_tokens=max_tokens, temperature=temperature)
            answer = response["text"]
            usage = response.get("usage")
        else:
            answer = self.provider.chat(
                messages, max_tokens=max_tokens, temperature=temperature)
            usage = None

        latency_llm = time.time() - start_llm
        total_latency = time.time() - start_total

        citations = re.findall(r"\[([^\]]+)\]", answer)

        # Extraer tokens del uso si viene (objeto), o None
        tokens_prompt = getattr(usage, "prompt_tokens", None) if usage else None
        tokens_completion = getattr(usage, "completion_tokens", None) if usage else None
        tokens_total = getattr(usage, "total_tokens", None) if usage else None

        return {
            "answer": answer,
            "citations": citations,
            "hits": hits,
            "latency_retrieve": latency_retrieve,
            "latency_llm": latency_llm,
            "latency_total": total_latency,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "tokens_total": tokens_total,
        }