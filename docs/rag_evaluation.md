# Evaluation Frameworks for RAG Systems

## 1. The RAG Evaluation Triad
Evaluating Retrieval-Augmented Generation systems requires separating the retrieval pipeline from the synthesis/generation model. The RAG Triad evaluates the three fundamental relationships:

```text
       [ Query ]
        /     \
       /       \
  Context       Answer
  Relevance    Relevance
     /           \
    /             \
[ Retrieved ] --- [ Generated ]
[ Context   ]     [ Answer    ]
        Faithfulness
```

### a. Context Precision & Recall (Retrieval Quality)
- **Context Precision**: Measures the proportion of retrieved chunks that are genuinely relevant to the query. High precision prevents irrelevant context from diluting the generator prompt.
- **Context Recall**: Measures whether all key facts needed to answer the question were successfully retrieved from the knowledge base.

### b. Faithfulness (Groundedness)
- Validates that every claim made in the generated answer can be mathematically or logically inferred directly from the retrieved context.
- Directly prevents hallucinations by penalizing claims unsupported by evidence.

### c. Answer Relevance
- Measures whether the final output directly answers the user query, independent of factual correctness.

## 2. Automated Evaluation with Synthetic Datasets
Modern pipelines generate synthetic question-context-answer triples using an LLM critic (e.g. Ragas or TruLens framework).

```python
# Conceptual verification scoring
faithfulness_score = count_supported_claims(answer, context) / total_claims(answer)
```

## 3. Practical Guardrails for CLI RAG
- **Relevance Thresholding**: Enforce a strict minimum similarity score (`--min-score 0.10`) to eliminate noise chunks.
- **Reciprocal Rank Fusion**: Combine lexical and semantic retrievers (`--algorithm hybrid`) to maximize recall across both keyword and conceptual questions.
