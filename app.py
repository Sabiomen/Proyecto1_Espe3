import click
from rag.ingest import ingest_raw
from rag.embed import build_index
from rag.retrieve import Retriever
from rag.pipeline import RAGPipeline
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider

from dotenv import load_dotenv

load_dotenv()

@click.group()
def cli():
    """Asistente UFRO CLI (RAG con ChatGPT / DeepSeek)"""
    pass

@cli.command()
def ingest():
    """Procesar documentos en data/raw -> data/processed/chunks.parquet"""
    ingest_raw()

@cli.command()
def index():
    """Construir embeddings e índice FAISS"""
    build_index()

@cli.command()
@click.option("--provider", type=click.Choice(["chatgpt","deepseek"]), default="chatgpt")
@click.option("--k", default=4, help="Número de chunks recuperados")
@click.option("--temperature", default=0.0, help="Temperatura de generación")
@click.option("--max-tokens", default=512, help="Máx. tokens en la respuesta")
def chat(provider, k, temperature, max_tokens):
    """Iniciar chatbot interactivo"""
    retriever = Retriever("data/index.faiss", "data/processed/chunks.parquet")

    if provider == "chatgpt":
        prov = ChatGPTProvider()
    else:
        prov = DeepSeekProvider()

    pipeline = RAGPipeline(retriever, prov)

    print(f"\nChatbot UFRO ({provider.upper()}) listo. Escribe tu pregunta (Enter vacío para salir).\n")

    while True:
        q = input("Pregunta> ").strip()
        if not q:
            break
        res = pipeline.synthesize(q, top_k=k, temperature=temperature, max_tokens=max_tokens)
        print("\n--- Respuesta ---\n")
        print(res["answer"])
        print("\n--- Citas detectadas ---")
        for cit in res["citations"]:
            print("-", cit)
        print("\n--- Fragmentos usados ---")
        for h in res["hits"]:
            print(f"- {h['title']} (p{h.get('page')}) score={h['score']:.3f}")
        print("\n============================\n")

if __name__ == "__main__":
    cli()