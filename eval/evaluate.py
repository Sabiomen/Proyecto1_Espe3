import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import time
import csv
from collections import defaultdict
from statistics import mean
import re
from rag.pipeline import RAGPipeline
from rag.retrieve import Retriever
from rag.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def em_score(pred: str, gold: str) -> int:
    return int(pred.strip().lower() == gold.strip().lower())

def coverage_citations(pred_citations, gold_refs):
    found = sum(1 for c in pred_citations if c in gold_refs)
    return found / max(len(gold_refs), 1)

def load_gold(path="eval/gold_set.jsonl"):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def estimate_cost(tokens_prompt, tokens_completion, provider_name):
    if tokens_prompt is None or tokens_completion is None:
        return None
    if provider_name == "chatgpt":
        return (tokens_prompt / 1000) * 0.03 + (tokens_completion / 1000) * 0.06
    elif provider_name == "deepseek":
        return (tokens_prompt / 1000) * 0.025 + (tokens_completion / 1000) * 0.05
    else:
        return None

def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    retriever = Retriever()
    providers = [ChatGPTProvider(), DeepSeekProvider()]
    pipelines = {p.name: RAGPipeline(retriever, p) for p in providers}
    model_embed = SentenceTransformer("all-MiniLM-L6-v2")
    gold_set = load_gold()

    # Diccionarios: proveedor -> lista resultados o métricas
    detailed_results = {p.name: [] for p in providers}
    metrics_accum = {p.name: defaultdict(list) for p in providers}  # {provider: {(question): [metricas]}}

    for item in gold_set:
        question = item["question"]
        expected_answer = item["expected_answer"]
        gold_refs = item.get("references", [])

        for pname, pipeline in pipelines.items():
            start_total = time.time()
            query = pipeline.rewrite_query(question)

            hits = retriever.query(query, top_k=5)
            snippets = "\n\n".join([f"[{h['title']}, p{h.get('page')}] {h['text']}" for h in hits])

            user_prompt = USER_PROMPT_TEMPLATE.format(question=question, snippets=snippets)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            start_llm = time.time()
            if hasattr(pipeline.provider, "chat_with_usage"):
                response = pipeline.provider.chat_with_usage(messages, max_tokens=512, temperature=0.0)
                answer = response.get("text", "")
                usage = response.get("usage", None)
            else:
                answer = pipeline.provider.chat(messages, max_tokens=512, temperature=0.0)
                usage = None
            latency_total = time.time() - start_total
            latency_llm = time.time() - start_llm

            tokens_prompt = getattr(usage, "prompt_tokens", None) if usage else None
            tokens_completion = getattr(usage, "completion_tokens", None) if usage else None

            citations = re.findall(r"\[([^\]]+)\]", answer) if answer else []

            em = em_score(answer, expected_answer) if answer else 0
            coverage = coverage_citations(citations, gold_refs)
            cost = estimate_cost(tokens_prompt, tokens_completion, pname)

            # Guardar resultados separando provedor
            detailed_results[pname].append({
                "question": question,
                "provider": pname,
                "answer": answer,
                "references": "; ".join(citations)
            })

            # Acumular métricas por pregunta
            metrics_accum[pname][question].append({
                "latency": latency_total,
                "em": em,
                "coverage": coverage,
                "cost": cost if cost is not None else 0.0
            })

    # Guardar CSVs de resultados separados
    result_fields = ["question", "provider", "answer", "references"]
    for pname, rows in detailed_results.items():
        write_csv(f"eval/results_{pname}.csv", result_fields, rows)

    # Guardar CSVs métricas - promedio y p95
    metric_fields = ["question", "provider", "em", "citation_coverage", "latency_avg_sec", "latency_p95_sec", "estimated_cost_usd"]
    for pname, questions_metrics in metrics_accum.items():
        metrics_summary = []
        for question, entries in questions_metrics.items():
            latencies = [e["latency"] for e in entries]
            ems = [e["em"] for e in entries]
            coverages = [e["coverage"] for e in entries]
            costs = [e["cost"] for e in entries]

            latency_avg = mean(latencies) if latencies else 0
            latency_p95 = np.percentile(latencies, 95) if latencies else 0
            em_avg = mean(ems) if ems else 0
            coverage_avg = mean(coverages) if coverages else 0
            cost_avg = mean(costs) if costs else 0

            metrics_summary.append({
                "question": question,
                "provider": pname,
                "em": round(em_avg, 3),
                "citation_coverage": round(coverage_avg, 3),
                "latency_avg_sec": round(latency_avg, 3),
                "latency_p95_sec": round(latency_p95, 3),
                "estimated_cost_usd": round(cost_avg, 6)
            })
        write_csv(f"eval/metrics_{pname}.csv", metric_fields, metrics_summary)

    print("Evaluación finalizada. CSVs generados por proveedor en la carpeta eval/")

if __name__ == "__main__":
    main()