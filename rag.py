"""
RAG CLI — Command-line Retrieval-Augmented Generation system.
Supports PDF and TXT document ingestion, chunking, and local vector retrieval.
"""

import os
import sys
import json
import math
import shutil
import argparse
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

# Configuration
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./db")
DOCS_DIR = os.environ.get("DOCS_DIR", "./docs")
EMBED_DIM = 1024
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


class LocalTFIDFEmbedder:
    """
    Local TF-IDF embedding generator that requires zero external API keys.
    Converts raw text chunks into normalized sparse-dense feature vectors.
    """
    def __init__(self, max_features: int = EMBED_DIM):
        self.max_features = max_features
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.num_docs = 0

    def _tokenize(self, text: str) -> List[str]:
        words = []
        token = []
        for char in text.lower():
            if char.isalnum():
                token.append(char)
            else:
                if token:
                    words.append("".join(token))
                    token = []
        if token:
            words.append("".join(token))
        return words

    def fit(self, documents: List[str]):
        """Fits vocabulary and computes inverse document frequencies (IDF)."""
        self.num_docs = len(documents)
        doc_freq = Counter()

        # Count document occurrences for each term
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                doc_freq[t] += 1

        # Select most common features up to max_features
        most_common = doc_freq.most_common(self.max_features)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}

        # Compute smoothed IDF
        self.idf = {}
        for word, count in most_common:
            self.idf[word] = math.log((1 + self.num_docs) / (1 + count)) + 1.0

    def embed_query(self, text: str) -> List[float]:
        """Generates embedding vector for a single query."""
        return self.embed_documents([text])[0]

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Transforms documents into L2-normalized TF-IDF vector embeddings."""
        embeddings = []
        dim = len(self.vocabulary) or self.max_features

        for doc in documents:
            vec = [0.0] * dim
            tokens = self._tokenize(doc)
            if not tokens or not self.vocabulary:
                embeddings.append(vec)
                continue

            tf = Counter(tokens)
            doc_len = len(tokens)

            for token, count in tf.items():
                if token in self.vocabulary:
                    idx = self.vocabulary[token]
                    tf_val = count / doc_len
                    vec[idx] = tf_val * self.idf.get(token, 1.0)

            # L2 normalization
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            embeddings.append(vec)

        return embeddings

    def save(self, path: str):
        """Serializes vocabulary and IDF metrics to disk."""
        data = {
            "vocabulary": self.vocabulary,
            "idf": self.idf,
            "num_docs": self.num_docs,
            "max_features": self.max_features,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Loads vocabulary and IDF metrics from disk."""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.vocabulary = data.get("vocabulary", {})
                self.idf = data.get("idf", {})
                self.num_docs = data.get("num_docs", 0)
                self.max_features = data.get("max_features", self.max_features)


class LocalVectorStore:
    """
    ChromaDB-compatible vector database interface.
    Persists document chunks, metadata, and embeddings locally in the DB directory.
    """
    def __init__(self, db_dir: str = CHROMA_DIR):
        self.db_dir = db_dir
        self.data_file = os.path.join(db_dir, "collection_data.json")
        self.embedder_file = os.path.join(db_dir, "embedder.json")
        self.embedder = LocalTFIDFEmbedder()
        self.chunks: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                self.embedder.load(self.embedder_file)
            except Exception:
                self.chunks = []

    def save(self):
        os.makedirs(self.db_dir, exist_ok=True)
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)
        self.embedder.save(self.embedder_file)

    def reset(self):
        self.chunks = []
        if os.path.exists(self.db_dir):
            shutil.rmtree(self.db_dir)
        os.makedirs(self.db_dir, exist_ok=True)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Indexes new document chunks and fits embeddings."""
        existing_docs = [c["text"] for c in self.chunks]
        all_docs = existing_docs + [d["text"] for d in documents]

        # Fit TF-IDF on complete corpus
        self.embedder.fit(all_docs)

        # Generate vectors for new documents
        new_texts = [d["text"] for d in documents]
        new_embeddings = self.embedder.embed_documents(new_texts)

        for d, emb in zip(documents, new_embeddings):
            d["embedding"] = emb
            self.chunks.append(d)

        # Update existing embeddings with new vocabulary
        if existing_docs:
            updated_existing = self.embedder.embed_documents(existing_docs)
            for chunk, emb in zip(self.chunks[:len(existing_docs)], updated_existing):
                chunk["embedding"] = emb

        self.save()


def load_txt(file_path: str) -> str:
    """Reads text from a plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    """Extracts text from a PDF file using pypdf if available."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_text.append(text)
        return "\n\n".join(pages_text)
    except ImportError:
        raise ImportError(
            "pypdf is required to parse PDF documents. Install it with: pip install pypdf"
        )


def load_document(file_path: str) -> str:
    """Dispatches to appropriate loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return load_txt(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Only .pdf and .txt are supported.")


def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Splits long document text into overlapping chunks respecting line or word boundaries."""
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if end < len(text):
            last_newline = chunk.rfind("\n")
            last_space = chunk.rfind(" ")
            break_point = max(last_newline, last_space)
            if break_point > chunk_size // 2:
                chunk = text[start : start + break_point]
                start += break_point + 1
            else:
                start += chunk_size - chunk_overlap
        else:
            start += chunk_size

        chunk_clean = chunk.strip()
        if chunk_clean:
            chunks.append(chunk_clean)

        if start >= len(text):
            break

    return chunks


def discover_documents(target_path: str) -> List[str]:
    """Finds all supported documents (.pdf, .txt) in the specified file or directory path."""
    if os.path.isfile(target_path):
        ext = os.path.splitext(target_path)[1].lower()
        if ext in [".pdf", ".txt"]:
            return [target_path]
        return []

    found = []
    if os.path.isdir(target_path):
        for root, _, files in os.walk(target_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in [".pdf", ".txt"]:
                    found.append(os.path.join(root, file))
    return sorted(found)


def ingest(target_path: str = DOCS_DIR, reset: bool = False):
    """Loads documents, splits into chunks, and stores into the vector database."""
    print(f"[*] Initializing ingestion pipeline (Target: {target_path})")
    store = LocalVectorStore()

    if reset:
        print("[!] Reset flag provided. Clearing existing vector database...")
        store.reset()

    doc_files = discover_documents(target_path)
    if not doc_files:
        print(f"[!] No valid .pdf or .txt documents discovered in '{target_path}'.")
        return

    print(f"[+] Found {len(doc_files)} document(s) to process.")
    all_chunks = []
    total_chars = 0

    for file_path in doc_files:
        try:
            content = load_document(file_path)
            chunks = split_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
            total_chars += len(content)
            base_name = os.path.basename(file_path)

            for i, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "id": f"{base_name}_chunk_{i}",
                    "text": chunk_text,
                    "metadata": {
                        "source": file_path,
                        "filename": base_name,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "char_count": len(chunk_text),
                    }
                })
            print(f"  -> Processed '{base_name}': {len(chunks)} chunks created.")
        except Exception as e:
            print(f"  [X] Failed to process '{file_path}': {e}")

    if all_chunks:
        print(f"[*] Generating TF-IDF vector embeddings for {len(all_chunks)} chunks...")
        store.add_documents(all_chunks)
        print(f"[SUCCESS] Ingestion completed! Stored {len(store.chunks)} total chunks in '{CHROMA_DIR}'.")
    else:
        print("[!] No chunks were produced.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG CLI — Retrieval-Augmented Generation")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into vector store")
    ingest_parser.add_argument("path", nargs="?", default=DOCS_DIR, help="Path to file or folder")
    ingest_parser.add_argument("--reset", action="store_true", help="Clear existing database before ingest")

    args = parser.parse_args()
    if args.command == "ingest":
        ingest(args.path, args.reset)
    else:
        parser.print_help()
