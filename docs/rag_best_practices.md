# RAG Best Practices and Engineering Guide

## 1. Chunking Strategies

Optimizing text chunk size is one of the most impactful decisions in any RAG architecture:

- **Fixed-size chunking**: Split by exact character or token counts (e.g. 500-1000 chars) with 10-20% overlap.
- **Sentence/Paragraph-aware chunking**: Respect semantic boundaries such as paragraphs (`\n\n`) and sentence punctuation.
- **Hierarchical chunking**: Maintain parent chunks for broad context and child chunks for precise semantic embedding match.

## 2. Embedding Selection

- **Sparse Models (TF-IDF, BM25)**:
  - Highly effective for domain-specific terminology, code keywords, product SKUs, and precise acronyms.
  - Zero computational overhead, runs locally without GPU or heavy weights.
- **Dense Models (e.g. all-MiniLM-L6-v2, BGE)**:
  - Captures contextual nuance, synonyms, and intent across varied phrasing.
  - Recommended when questions use phrasing different from the source documents.

## 3. Top-k and Score Thresholding

- Avoid returning chunks with negligible similarity scores to reduce prompt noise.
- Setting a `--min-score` of `0.10` or higher filters out spurious matches.
- Dynamically adjust `k` based on document complexity (typically 3 to 6 chunks).
