"""
RAG CLI — Command-line Retrieval-Augmented Generation system.
Supports PDF and TXT document ingestion, chunking, and local vector retrieval.
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Tuple

# Configuration
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./db")
DOCS_DIR = os.environ.get("DOCS_DIR", "./docs")
EMBED_DIM = 1024
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


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


if __name__ == "__main__":
    print(f"RAG CLI module loaded. Default Docs: {DOCS_DIR}, DB: {CHROMA_DIR}")
