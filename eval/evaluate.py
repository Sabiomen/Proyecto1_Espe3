import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import time
import csv
from collections import defaultdict
from statistics import mean
from rag.pipeline import RAGPipeline
from rag.retrieve import Retriever
from rag.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from dotenv import load_dotenv

load_dotenv()

# Funciones métricas previas (em_score, coverage_citations) aquí...

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

def main():
    retriever = Retriever()
    providers = [ChatGPTProvider(), DeepSeekProvider()]
    pipelines = {p.name: RAGPipeline(retriever, p) for p in providers}
    model_embed = SentenceTransformer("all-MiniLM-L6-v2")
    gold_set = load_gold()

    # Para almacenar resultados detallados (por cada consulta)
    detailed_results = []

    # Para acumular métricas para agregados (por pregunta y provider)
    metrics_accum = defaultdict(list)  # {(question, provider): [latencies, ems, coverage, costos]}

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
            response = pipeline.provider.chat(messages, max_tokens=512, temperature=0.0)
            latency_total = time.time() - start_total
            latency_llm = time.time() - start_llm

            if isinstance(response, dict):
                answer = response.get("text", "")
                usage = response.get("usage", None)
            else:
                answer = response
                usage = None

            tokens_prompt = None
            tokens_completion = None
            if usage is not None:
                tokens_prompt = getattr(usage, "prompt_tokens", None)
                tokens_completion = getattr(usage, "completion_tokens", None)

            citations = re.findall(r"\[([^\]]+)\]", answer) if answer else []

            em = em_score(answer, expected_answer) if answer else 0
            coverage = coverage_citations(citations, gold_refs)

            cost = estimate_cost(tokens_prompt, tokens_completion, pname)

            # Guardar resultados detallados (para CSV resultados con respuestas y referencias)
            detailed_results.append({
                "question": question,
                "provider": pname,
                "answer": answer,
                "references": "; ".join(citations)
            })

            # Acumular métricas para agregación posterior
            metrics_accum[(question, pname)].append({
                "latency": latency_total,
                "em": em,
                "coverage": coverage,
                "cost": cost if cost is not None else 0.0
            })

    # Generar CSV de resultados con respuestas + referencias
    with open("eval/results.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = ["question", "provider", "answer", "references"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_results)

    # Calcular métricas agregadas y guardar en CSV aparte
    metrics_summary = []
    for (question, provider), entries in metrics_accum.items():
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
            "provider": provider,
            "em": round(em_avg, 3),
            "citation_coverage": round(coverage_avg, 3),
            "latency_avg_sec": round(latency_avg, 3),
            "latency_p95_sec": round(latency_p95, 3),
            "estimated_cost_usd": round(cost_avg, 6)
        })

    # Guardar CSV de métricas
    with open("eval/metrics.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = ["question", "provider", "em", "citation_coverage", "latency_avg_sec", "latency_p95_sec", "estimated_cost_usd"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_summary)

    print("Evaluación completada. CSVs generados: eval/results.csv y eval/metrics.csv")

if __name__ == "__main__":
    main()