from rag.retrieve import Retriever
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
        q_rewritten = self.rewrite_query(query)
        hits = self.retriever.query(q_rewritten, top_k=top_k)

        snippets = "\n\n".join([
            f"[{h['title']}, p{h.get('page')}] {h['text']}" for h in hits
        ])

        user_prompt = USER_PROMPT_TEMPLATE.format(question=query, snippets=snippets)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # provider.chat devuelve un string
        answer = self.provider.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        print(answer)

        # buscar citas en formato [ ... ]
        citations = re.findall(r"\[([^\]]+)\]", answer)


        return {
            "answer": answer,
            "citations": citations,
            "hits": hits
        }