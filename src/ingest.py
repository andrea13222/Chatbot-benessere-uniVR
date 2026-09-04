import os
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "documenti"
WEB_DIR = BASE_DIR / "data" / "sito_scraped"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
MODELS_DIR = BASE_DIR / "models" / "multilingual-e5-small"
EMBEDDING_MODEL = str(MODELS_DIR)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

def read_pdf(path):
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def read_html(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text(separator="\n")

def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def load_all_documents():
    documents = []
    for folder in [DOCS_DIR, WEB_DIR]:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.suffix.lower() == ".pdf":
                text = read_pdf(path)
            elif path.suffix.lower() in [".html", ".htm"]:
                text = read_html(path)
            elif path.suffix.lower() == ".txt":
                text = read_txt(path)
            else:
                continue
            for chunk in chunk_text(text):
                documents.append((chunk, str(path.name)))
    return documents

def main():
    print("Caricamento documenti...")
    docs = load_all_documents()
    print(f"Trovati {len(docs)} chunk totali.")
    if not docs:
        print("Nessun dato trovato.")
        return

    print("Caricamento modello di embedding...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("Creazione embedding...")

    texts = [d[0] for d in docs]
    sources = [d[1] for d in docs]
    texts_con_prefisso = [f"passage: {t}" for t in texts]
    embeddings = embedder.encode(texts_con_prefisso, show_progress_bar=True, normalize_embeddings=True)
    print("Salvataggio in ChromaDB...")

    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    collection = client.get_or_create_collection(name="tesi_documenti")
    ids = [f"doc_{i}" for i in range(len(texts))]
    metadatas = [{"source": s} for s in sources]
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Completato. {len(texts)} chunk indicizzati in {VECTORSTORE_DIR}")

if __name__ == "__main__":
    main()
