from rag.retrieve import Retriever
import time
import logging
from rag.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
import re

# Configura logging simple
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, retriever: Retriever, provider):
        self.retriever = retriever
        self.provider = provider
        self.max_retries = 3
        self.retry_wait = 2  # segundos

    def rewrite_query(self, query: str) -> str:
        return query

    def _call_provider_with_retries(self, call_func, *args, **kwargs):
        """
        Función interna para llamar al proveedor con manejo básico de errores y reintentos.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Llamada al proveedor, intento {attempt}")
                result = call_func(*args, **kwargs)
                return result
            except Exception as e:
                logger.warning(f"Error en intento {attempt}: {e}")
                if attempt == self.max_retries:
                    logger.error(f"Máximo de reintentos alcanzado, abortando con error: {e}")
                    raise
                time.sleep(self.retry_wait)

    def synthesize(self, query: str, top_k=4, max_tokens=512, temperature=0.0, with_usage=False):
        start_total = time.time()
        q_rewritten = self.rewrite_query(query)

        # Recuperación
        start_retrieve = time.time()
        hits = self.retriever.query(q_rewritten, top_k=top_k)
        latency_retrieve = time.time() - start_retrieve
        logger.info(f"Recuperación completada en {latency_retrieve:.2f}s, {len(hits)} fragmentos obtenidos.")

        snippets = "\n\n".join([
            f"[{h['title']}, p{h.get('page')}] {h['text']}" for h in hits
        ])

        user_prompt = USER_PROMPT_TEMPLATE.format(question=query, snippets=snippets)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # Llamada al LLM con manejo de errores y reintentos
        start_llm = time.time()
        if with_usage and hasattr(self.provider, "chat_with_usage"):
            call_func = self.provider.chat_with_usage
        else:
            call_func = self.provider.chat

        response = self._call_provider_with_retries(
            call_func, messages, max_tokens=max_tokens, temperature=temperature
        )

        latency_llm = time.time() - start_llm
        total_latency = time.time() - start_total
        logger.info(f"Llamada LLM completada en {latency_llm:.2f}s, latencia total {total_latency:.2f}s.")

        # Procesar respuesta y citas
        if with_usage and isinstance(response, dict):
            answer = response.get("text", "")
            usage = response.get("usage")
        else:
            answer = response
            usage = None

        citations = re.findall(r"\[([^\]]+)\]", answer)

        tokens_prompt = getattr(usage, "prompt_tokens", None) if usage else None
        tokens_completion = getattr(usage, "completion_tokens", None) if usage else None
        tokens_total = getattr(usage, "total_tokens", None) if usage else None

        # Abstención simple si no hay fragmentos o citas (puedes ajustar la política)
        if not hits or (not citations and with_usage):
            abstention_msg = "No encontrado en normativa UFRO. Sugiero consultar a la oficina correspondiente."
            logger.warning(f"Aplicando política de abstención para consulta: {query}")
            return {
                "answer": abstention_msg,
                "citations": [],
                "hits": hits,
                "latency_retrieve": latency_retrieve,
                "latency_llm": latency_llm,
                "latency_total": total_latency,
                "tokens_prompt": tokens_prompt,
                "tokens_completion": tokens_completion,
                "tokens_total": tokens_total,
            }

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