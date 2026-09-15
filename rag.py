"""
RAG CLI — Command-line Retrieval-Augmented Generation system.
Supports PDF and TXT document ingestion, chunking, and local vector retrieval.
"""

import os
import sys
import time
import json
import math
import shutil
import argparse
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

# Ensure standard streams handle UTF-8 properly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Configuration
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./db")
DOCS_DIR = os.environ.get("DOCS_DIR", "./docs")
EMBED_DIM = 1024
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

BANNER = r"""
=============================================================
   ___            ___      ____ _     ___ 
  | _ \  __ _  __/ __|    / ___| |   |_ _|
  |   / / _` || (_ | --- | |   | |    | | 
  |_|_\ \__,_| \___|      \____|___| |___|
  Local Semantic Retrieval-Augmented Generation Engine
=============================================================
"""


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

        for doc in documents:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                doc_freq[t] += 1

        most_common = doc_freq.most_common(self.max_features)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}

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
    Vector database interface.
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

        self.embedder.fit(all_docs)

        new_texts = [d["text"] for d in documents]
        new_embeddings = self.embedder.embed_documents(new_texts)

        for d, emb in zip(documents, new_embeddings):
            d["embedding"] = emb
            self.chunks.append(d)

        if existing_docs:
            updated_existing = self.embedder.embed_documents(existing_docs)
            for chunk, emb in zip(self.chunks[:len(existing_docs)], updated_existing):
                chunk["embedding"] = emb

        self.save()

    def similarity_search(self, query: str, k: int = 4, min_score: float = 0.0) -> List[Tuple[Dict[str, Any], float]]:
        """Computes cosine similarity against all chunks and returns top-k matches filtered by min_score."""
        if not self.chunks:
            return []

        q_vec = self.embedder.embed_query(query)
        q_norm = math.sqrt(sum(x * x for x in q_vec))
        if q_norm == 0:
            return [(self.chunks[i], 0.0) for i in range(min(k, len(self.chunks)))]

        scores = []
        for chunk in self.chunks:
            c_vec = chunk.get("embedding", [])
            if not c_vec or len(c_vec) != len(q_vec):
                dot = 0.0
            else:
                dot = sum(a * b for a, b in zip(q_vec, c_vec))
            if dot >= min_score:
                scores.append((chunk, dot))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:k]


def load_txt(file_path: str) -> str:
    """Reads text from a plain text file, stripping BOM if present."""
    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
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
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Target document '{file_path}' does not exist.")

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
    start_time = time.time()
    print(BANNER)
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

    for file_path in doc_files:
        try:
            content = load_document(file_path)
            chunks = split_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
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
        elapsed = time.time() - start_time
        print(f"[SUCCESS] Ingestion completed in {elapsed:.2f}s! Stored {len(store.chunks)} total chunks in '{CHROMA_DIR}'.")
    else:
        print("[!] No chunks were produced.")


def format_search_results(results: List[Tuple[Dict[str, Any], float]]):
    """Utility to print search results in a clean, readable format."""
    if not results:
        print("\n[!] No matching document chunks exceeded the relevance threshold.")
        return

    for rank, (chunk, score) in enumerate(results, start=1):
        meta = chunk.get("metadata", {})
        fname = meta.get("filename", "unknown")
        idx = meta.get("chunk_index", 0)
        tot = meta.get("total_chunks", 1)
        sim_pct = max(0.0, score * 100)

        print(f"\n[Result #{rank}] Relevance: {sim_pct:.1f}% | Source: {fname} (chunk {idx + 1}/{tot})")
        print("~" * 60)
        lines = chunk["text"].splitlines()
        for line in lines:
            print(f"  {line}")


def query_store(query_text: str, k: int = 4, min_score: float = 0.0):
    """Performs one-shot semantic search and prints formatted results."""
    store = LocalVectorStore()
    if not store.chunks:
        print("[!] Vector database is empty. Please run 'python rag.py ingest' first.")
        return

    print(f"\n[QUERY] \"{query_text}\" (retrieving top-{k} chunks)\n" + "-" * 60)
    start_time = time.time()
    results = store.similarity_search(query_text, k=k, min_score=min_score)
    format_search_results(results)
    elapsed = (time.time() - start_time) * 1000
    print(f"\n[Search completed in {elapsed:.1f} ms]")


def chat_session(k: int = 4, min_score: float = 0.0):
    """Starts an interactive command-line session for continuous retrieval."""
    store = LocalVectorStore()
    if not store.chunks:
        print("[!] Vector database is empty. Please run 'python rag.py ingest' first.")
        return

    print(BANNER)
    print("       Interactive Retrieval Session")
    print(f"       Chunks per query: {k} | Type 'exit' or 'quit' to end")
    print("=" * 60)

    while True:
        try:
            prompt = input("\nrag> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", ":q"]:
                print("Exiting RAG session. Goodbye!")
                break

            results = store.similarity_search(prompt, k=k, min_score=min_score)
            format_search_results(results)
        except (KeyboardInterrupt, EOFError):
            print("\nSession terminated by user. Goodbye!")
            break


def show_info():
    """Prints status and statistics about the vector store."""
    store = LocalVectorStore()
    print("\n" + "=" * 50)
    print("           RAG CLI - VECTOR STORE INFO")
    print("=" * 50)
    print(f"Storage Directory  : {os.path.abspath(store.db_dir)}")
    print(f"Total Chunks       : {len(store.chunks)}")

    sources = set(c.get("metadata", {}).get("filename", "") for c in store.chunks)
    sources.discard("")
    print(f"Unique Documents   : {len(sources)}")
    for s in sorted(sources):
        chunk_count = sum(1 for c in store.chunks if c.get("metadata", {}).get("filename") == s)
        print(f"  * {s} ({chunk_count} chunks)")

    print(f"Vocabulary Size    : {len(store.embedder.vocabulary)} tokens")
    if store.chunks:
        avg_chars = sum(len(c["text"]) for c in store.chunks) / len(store.chunks)
        print(f"Avg Chunk Length   : {avg_chars:.1f} characters")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG CLI — Command-line Retrieval-Augmented Generation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python rag.py ingest\n  python rag.py query \"What is RAG?\"\n  python rag.py chat -k 3\n  python rag.py info",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into vector store")
    ingest_parser.add_argument("path", nargs="?", default=DOCS_DIR, help="Path to file or folder")
    ingest_parser.add_argument("--reset", action="store_true", help="Clear existing database before ingest")

    # Query command
    query_parser = subparsers.add_parser("query", help="One-shot semantic query")
    query_parser.add_argument("query_text", type=str, help="Search query string")
    query_parser.add_argument("-k", "--top-k", type=int, default=4, help="Number of chunks to return")
    query_parser.add_argument("--min-score", type=float, default=0.0, help="Minimum similarity score threshold (0.0 to 1.0)")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive terminal chat session")
    chat_parser.add_argument("-k", "--top-k", type=int, default=4, help="Number of chunks to return")
    chat_parser.add_argument("--min-score", type=float, default=0.0, help="Minimum similarity score threshold (0.0 to 1.0)")

    # Info command
    info_parser = subparsers.add_parser("info", help="Display vector store info and statistics")

    args = parser.parse_args()
    if args.command == "ingest":
        ingest(args.path, args.reset)
    elif args.command == "query":
        query_store(args.query_text, args.top_k, args.min_score)
    elif args.command == "chat":
        chat_session(args.top_k, args.min_score)
    elif args.command == "info":
        show_info()
    else:
        parser.print_help()

