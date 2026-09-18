# Vector Indexing with HNSW (Hierarchical Navigable Small World)

## 1. Overview
Approximate Nearest Neighbor (ANN) search is fundamental to modern vector databases (e.g. ChromaDB, FAISS, Milvus). Exact nearest neighbor search (k-NN) with brute-force cosine computation scales with $O(N \cdot D)$, which becomes impractical when collections scale into millions of high-dimensional vectors.

HNSW is currently the gold standard graph-based indexing algorithm, achieving logarithmic search time $O(\log N)$ while maintaining exceptionally high recall (>95%).

## 2. Multi-Layer Graph Architecture
HNSW organizes vectors into a hierarchy of layers:
- **Top Layers**: Sparse graphs with long-range skip connections, facilitating fast navigation across large distances in vector space.
- **Bottom Layer (Layer 0)**: Dense graph containing all data points with short-range edges connecting close neighbors.

During retrieval, search enters at the top layer, performs greedy routing to find the closest entry point, drops to the layer below, and repeats until terminating in Layer 0.

## 3. Core Tuning Hyperparameters

| Parameter | Recommended Default | Impact on Indexing | Impact on Search |
|---|---|---|---|
| `M` | `16 - 64` | Higher `M` increases build time and memory usage. | Improves recall for clustered or high-dimensional datasets. |
| `efConstruction` | `64 - 200` | Controls build exploration depth; higher values yield higher graph quality. | Does not affect search latency directly. |
| `efSearch` | `32 - 128` | No build-time impact. | Controls search beam width; trade-off between query speed and recall. |

## 4. Distance Metrics
- **Cosine Distance ($1 - \cos(\theta)$)**: Best suited for semantic text embeddings where direction captures meaning regardless of vector magnitude.
- **Euclidean / L2 Distance**: Standard geometric distance metric; requires vectors to be normalized for angular equivalency.
- **Inner Product (IP)**: Maximum inner product search; mathematically equivalent to cosine distance when all vectors are unit-normalized.
