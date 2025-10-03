import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, index_path="data/index.faiss", meta_path="data/processed/chunks.parquet"):
        self.index = faiss.read_index(index_path)
        self.df = pd.read_parquet(meta_path)
        self.model = SentenceTransformer(MODEL)

    def query(self, question: str, top_k=5):
        q_emb = self.model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            row = self.df.iloc[idx].to_dict()
            row["score"] = float(score)
            results.append(row)
        return results

