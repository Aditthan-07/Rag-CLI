# RAG CLI — LangChain + ChromaDB

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-performance command-line RAG (Retrieval-Augmented Generation) system that ingests **PDF** and **TXT** files, computes vector embeddings, stores them in a local ChromaDB-compatible vector store, and lets you query them semantically from your terminal.

No external API keys required — embeddings run 100% locally via TF-IDF (scikit-learn) with optional dense embedding plug-ins.

---

## Setup (PowerShell / Windows)

```powershell
# 1. Navigate to the project folder
cd Rag-CLI

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
venv\Scripts\activate

# If activation is blocked on Windows, run this once then retry:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 4. Install dependencies
pip install -r requirements.txt
```

> ⚠️ Use Python from [python.org](https://python.org), not the Microsoft Store version — the Store version breaks `venv`.

---

## Usage

### 1. Ingest documents

Drop your `.pdf` or `.txt` files into the `docs/` folder, then run:

```powershell
python rag.py ingest
```

Point at a specific file or folder:

```powershell
python rag.py ingest path\to\myfile.pdf
python rag.py ingest path\to\myfolder\
```

Re-ingest from scratch (clears existing DB):

```powershell
python rag.py ingest --reset
```

---

### 2. Interactive chat

Start a retrieval session — type questions, get the most relevant chunks back:

```powershell
python rag.py chat
```

Control how many chunks are returned and filter by minimum similarity:

```powershell
python rag.py chat -k 6
python rag.py chat -k 4 --min-score 0.10
```

Type `exit` or `quit` to stop.

---

### 3. One-shot query

```powershell
python rag.py query "What is ChromaDB used for?"
python rag.py query "What is RAG?" -k 2
```

---

### 4. Vector store info

```powershell
python rag.py info
```

---

### 5. Running Automated Tests

Run the comprehensive unit and integration test suite:

```powershell
python -m unittest discover tests
```

---

## Project Structure

```
Rag-CLI/
├── rag.py                  # Main CLI application & retrieval engine
├── requirements.txt        # Python dependencies
├── docs/                   # Drop your PDF / TXT files here
│   ├── intro_to_rag.txt    # Sample RAG architecture guide
│   └── chroma_cheatsheet.txt # ChromaDB reference notes
├── tests/                  # Automated unit and integration tests
│   └── test_rag.py         # Test suite for loaders, chunker, & vector store
└── db/                     # ChromaDB vector store (auto-created on ingest)
```

---

## How It Works

| Step | What happens |
|------|-------------|
| **Load** | `load_document` reads your `.pdf` (via `pypdf`) and `.txt` files with UTF-8/BOM sanitization |
| **Split** | `split_text` chunks them (800 chars, 100 overlap) respecting word/newline boundaries |
| **Embed** | Local TF-IDF converts each chunk to an L2-normalized sparse-dense vector |
| **Store** | Persists vectors and metadata locally in `./db` |
| **Retrieve** | Cosine similarity search ranks and returns top-k relevant chunks |

---

## Config (top of `rag.py`)

```python
CHROMA_DIR    = "./db"
DOCS_DIR      = "./docs"
EMBED_DIM     = 1024
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
```

Tune `CHUNK_SIZE` / `CHUNK_OVERLAP` for your document density.

---

## Upgrading to Dense Embeddings

TF-IDF works quickly and requires zero GPU/model downloads. For deep semantic similarity, swap in `sentence-transformers`:

```powershell
pip install sentence-transformers langchain-huggingface
```

Then replace the embedder in `rag.py` with:

```python
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

The first run will download the model (~90 MB) and cache it locally. Fully offline after that.

---

## Common Issues

| Problem | Fix |
|---|---|
| `activate` is blocked | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `python` not found | Reinstall from python.org with "Add to PATH" checked |
| `(venv)` missing from prompt | Re-run `venv\Scripts\activate` |
| Added new docs | Run `python rag.py ingest --reset` to re-index |
