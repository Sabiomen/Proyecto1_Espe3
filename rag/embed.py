from pathlib import Path
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "all-MiniLM-L6-v2"

def build_index(chunks_parquet="data/processed/chunks.parquet",
                index_path="data/index.faiss",
                meta_path="data/processed/chunks.parquet"):
    df = pd.read_parquet(chunks_parquet)
    texts = df["text"].tolist()
    model = SentenceTransformer(MODEL)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    df.reset_index(drop=True, inplace=True)
    df.to_parquet(meta_path, index=False)
    print(f"Índice FAISS guardado en {index_path} con {len(df)} vectores.")

if __name__ == "__main__":
    build_index()