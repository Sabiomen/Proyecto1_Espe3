from pathlib import Path
import re
import pandas as pd
from pypdf import PdfReader

CHUNK_SIZE = 900
OVERLAP = 120

def clean_text(text: str) -> str:
    text = text.replace("\x0c", " ")  # saltos de página
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        yield " ".join(words[start:end])
        if end == len(words):
            break
        start = end - overlap

def pdf_to_chunks(path: Path, doc_id: str, title: str):
    reader = PdfReader(str(path))
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue
        for chunk in chunk_text(text):
            chunks.append({
                "doc_id": doc_id,
                "title": title,
                "page": page_num,
                "text": chunk
            })
    return chunks

def txt_to_chunks(path: Path, doc_id: str, title: str):
    text = clean_text(path.read_text(encoding="utf-8"))
    chunks = []
    for chunk in chunk_text(text):
        chunks.append({
            "doc_id": doc_id,
            "title": title,
            "page": None,
            "text": chunk
        })
    return chunks

def ingest_raw(raw_dir="data/raw", out_parquet="data/processed/chunks.parquet"):
    raw_path = Path(raw_dir)
    rows = []
    for f in raw_path.iterdir():
        if f.suffix.lower() == ".pdf":
            rows.extend(pdf_to_chunks(f, f.stem, f.stem))
        elif f.suffix.lower() == ".txt":
            rows.extend(txt_to_chunks(f, f.stem, f.stem))
    df = pd.DataFrame(rows)
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"Procesados {len(df)} chunks de {len(list(raw_path.iterdir()))} archivos")
    return df

if __name__ == "__main__":
    ingest_raw()