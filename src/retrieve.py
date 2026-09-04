from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
MODELS_DIR = BASE_DIR / "models" / "multilingual-e5-small"
EMBEDDING_MODEL = str(MODELS_DIR)
_embedder = None
_collection = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        _collection = client.get_or_create_collection(name="tesi_documenti")
    return _collection

def retrieve(query, top_k=4):
    embedder = _get_embedder()
    collection = _get_collection()
    query_embedding = embedder.encode([f"query: {query}"], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return list(zip(chunks, sources))

if __name__ == "__main__":
    query = input("Domanda di prova: ")
    for chunk, source in retrieve(query):
        print(f"\n--- da {source} ---\n{chunk[:300]}...")
