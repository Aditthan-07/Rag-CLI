# RAG CLI — LangChain + ChromaDB

A command-line RAG (Retrieval-Augmented Generation) system that ingests **PDF** and **TXT** files, stores them in a local ChromaDB vector store, and lets you query them semantically from your terminal.

No API keys required — embeddings run locally via TF-IDF (scikit-learn).

---

## Setup (PowerShell)

```powershell
# 1. Navigate to the project folder
cd rag-cli

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
venv\Scripts\activate

# If activation is blocked, run this once then retry:
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

Control how many chunks are returned:

```powershell
python rag.py chat -k 6
```

Type `exit` or `quit` to stop.

---

### 3. One-shot query

```powershell
python rag.py query "What is ChromaDB used for?"
```

---

### 4. Vector store info

```powershell
python rag.py info
```

---

## Project Structure

```
rag-cli/
├── rag.py            # Main CLI application
├── requirements.txt
├── docs/             # Drop your PDF / TXT files here
│   └── intro_to_rag.txt   # Sample document
└── db/               # ChromaDB vector store (auto-created on ingest)
```

---

## How It Works

| Step | What happens |
|------|-------------|
| **Load** | `PyPDFLoader` / `TextLoader` reads your files |
| **Split** | `RecursiveCharacterTextSplitter` chunks them (800 chars, 100 overlap) |
| **Embed** | TF-IDF (scikit-learn) converts each chunk to a sparse vector |
| **Store** | ChromaDB persists vectors locally in `./db` |
| **Retrieve** | Cosine similarity search returns the top-k relevant chunks |

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

TF-IDF works well for keyword-heavy queries. For semantic similarity (better quality), swap in `sentence-transformers`:

```powershell
pip install sentence-transformers langchain-huggingface
```

Then replace `TFIDFEmbeddingFunction` in `rag.py` with:

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