"""
Unit and integration tests for RAG CLI.
Tests document loaders, chunking mechanics, TF-IDF embedder, and vector retrieval.
"""

import os
import shutil
import tempfile
import unittest
from rag import (
    split_text,
    discover_documents,
    load_document,
    LocalTFIDFEmbedder,
    LocalVectorStore,
)


class TestRagPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_dir = os.path.join(self.test_dir, "test_db")
        os.makedirs(self.db_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_split_text_small(self):
        text = "Hello world! This is a simple test."
        chunks = split_text(text, chunk_size=100, chunk_overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_split_text_chunking(self):
        text = "word " * 300  # 1500 characters
        chunks = split_text(text, chunk_size=400, chunk_overlap=50)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 450)

    def test_discover_documents(self):
        f1 = os.path.join(self.test_dir, "doc1.txt")
        f2 = os.path.join(self.test_dir, "doc2.pdf")
        f3 = os.path.join(self.test_dir, "ignore.exe")

        with open(f1, "w", encoding="utf-8") as f:
            f.write("Test doc 1")
        with open(f2, "w", encoding="utf-8") as f:
            f.write("Fake pdf header")
        with open(f3, "w", encoding="utf-8") as f:
            f.write("Binary content")

        found = discover_documents(self.test_dir)
        self.assertIn(f1, found)
        self.assertIn(f2, found)
        self.assertNotIn(f3, found)

    def test_tfidf_embedder(self):
        docs = [
            "Vector databases are optimized for similarity search.",
            "Retrieval-Augmented Generation bridges LLMs with domain documents.",
            "Python CLI tools provide rapid interactive interfaces.",
        ]
        embedder = LocalTFIDFEmbedder(max_features=50)
        embedder.fit(docs)

        self.assertGreater(len(embedder.vocabulary), 0)
        embeddings = embedder.embed_documents(docs)
        self.assertEqual(len(embeddings), 3)

        # Check vector normalization (length close to 1.0)
        for emb in embeddings:
            norm = sum(x * x for x in emb) ** 0.5
            self.assertAlmostEqual(norm, 1.0, places=4)

        # Query embedding
        q_emb = embedder.embed_query("vector search")
        self.assertEqual(len(q_emb), len(embedder.vocabulary))

    def test_vector_store_end_to_end(self):
        store = LocalVectorStore(db_dir=self.db_dir)
        docs = [
            {
                "id": "chunk_0",
                "text": "ChromaDB stores embeddings and performs fast nearest neighbor lookups.",
                "metadata": {"filename": "chroma.txt", "chunk_index": 0, "total_chunks": 1},
            },
            {
                "id": "chunk_1",
                "text": "LangChain provides abstractions for prompt templates and chains.",
                "metadata": {"filename": "langchain.txt", "chunk_index": 0, "total_chunks": 1},
            },
        ]

        store.add_documents(docs)
        self.assertEqual(len(store.chunks), 2)

        # Test similarity search retrieval
        results = store.similarity_search("ChromaDB vector lookups", k=1)
        self.assertEqual(len(results), 1)
        best_match, score = results[0]
        self.assertEqual(best_match["id"], "chunk_0")
        self.assertGreater(score, 0.0)

        # Test persistence
        reloaded_store = LocalVectorStore(db_dir=self.db_dir)
        self.assertEqual(len(reloaded_store.chunks), 2)


if __name__ == "__main__":
    unittest.main()
