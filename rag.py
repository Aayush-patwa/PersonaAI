# rag.py - RAG engine for JARVIS
# Dense retrieval (sentence-transformers + FAISS) + sparse retrieval (BM25),
# fused with Reciprocal Rank Fusion (RRF).
import os
import re
import json
import math
from collections import Counter

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────
# CONFIG  (the evaluation script reads these too)
# ─────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "database", "vector_store")
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 only embeds the first 256 word-pieces of its input, so a chunk
# must stay below that (~160 words) or its tail is silently ignored.
CHUNK_SIZE = 160      # words per chunk
CHUNK_OVERLAP = 40    # words shared between neighbouring chunks
TOP_K = 4             # chunks sent to the LLM
RETRIEVAL_MODE = "hybrid"   # "dense" | "bm25" | "hybrid"  (confirm the best one with eval_rag.py --ablate)
RRF_K = 60            # standard RRF constant

# ─────────────────────────────────────────
# EMBEDDING MODEL (loaded once, on first use)
# ─────────────────────────────────────────

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        print("🧠 Loading embedding model...")
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
        print("✅ Embedding model loaded")
    return _embedder


def warmup():
    """Load the model ahead of time so the first user query isn't slow."""
    get_embedder().encode(["warmup"], show_progress_bar=False)


def _embed(texts):
    vecs = get_embedder().encode(
        texts, batch_size=32, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True
    )
    return np.ascontiguousarray(vecs, dtype="float32")


# ─────────────────────────────────────────
# CHUNKING + TOKENISING
# ─────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping word chunks."""
    words = text.split()
    if not words:
        return []
    overlap = max(0, min(overlap, chunk_size - 1))
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):   # last chunk reached the end
            break
    return chunks


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in", "on", "at", "for",
    "and", "or", "but", "it", "this", "that", "with", "as", "by", "from", "what", "how",
    "why", "when", "where", "which", "who", "do", "does", "did", "i", "me", "my", "you",
}


def tokenize(text: str) -> list:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS and len(t) > 1]


# ─────────────────────────────────────────
# BM25 (sparse / keyword retrieval)
# ─────────────────────────────────────────

class BM25:
    def __init__(self, docs_tokens, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs_tokens)
        self.doc_len = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.tf = [Counter(d) for d in docs_tokens]
        df = Counter()
        for d in docs_tokens:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens):
        out = [0.0] * self.n
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                    out[i] += idf * f * (self.k1 + 1) / denom
        return out


# ─────────────────────────────────────────
# INDEX BUILDING + SEARCH (in memory)
# ─────────────────────────────────────────

def make_index(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> dict:
    """Build an in-memory index: chunks + FAISS vectors + BM25 statistics."""
    chunks = chunk_text(text, chunk_size, overlap)
    if not chunks:
        raise ValueError("No text to index")
    vectors = _embed(chunks)
    faiss_index = faiss.IndexFlatIP(vectors.shape[1])   # inner product == cosine (vectors are normalised)
    faiss_index.add(vectors)
    return {
        "chunks": chunks,
        "faiss": faiss_index,
        "bm25": BM25([tokenize(c) for c in chunks]),
        "meta": {
            "embed_model": EMBED_MODEL_NAME,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "n_chunks": len(chunks),
        },
    }


def search_index(index: dict, query: str, top_k: int = TOP_K, mode: str = RETRIEVAL_MODE) -> list:
    """Return [(chunk_text, score), ...] best first. Scores are comparable only within one mode."""
    n = len(index["chunks"])
    k = min(top_k, n)
    pool = min(n, max(top_k * 4, 12))    # candidates taken from each retriever before fusion

    dense, sparse = [], []
    if mode in ("dense", "hybrid"):
        scores, ids = index["faiss"].search(_embed([query]), pool)
        dense = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
    if mode in ("bm25", "hybrid"):
        sc = index["bm25"].scores(tokenize(query))
        order = sorted(range(n), key=lambda i: sc[i], reverse=True)[:pool]
        sparse = [(i, sc[i]) for i in order if sc[i] > 0]

    if mode == "dense":
        ranked = dense[:k]
    elif mode == "bm25":
        ranked = sparse[:k]
    elif mode == "hybrid":
        fused = {}
        for ranking in (dense, sparse):
            for rank, (i, _) in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    return [(index["chunks"][i], score) for i, score in ranked]


# ─────────────────────────────────────────
# PERSISTENCE  (one index per document, keyed by its unique stored filename)
# ─────────────────────────────────────────

_CACHE = {}


def _paths(doc_key: str):
    base = os.path.join(VECTOR_STORE_PATH, str(doc_key))
    return base + ".faiss", base + ".json"


def build_index(doc_key: str, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> int:
    """Build, save and cache the index for a document. Returns the number of chunks."""
    print(f"🔨 Building RAG index for {doc_key}...")
    index = make_index(text, chunk_size, overlap)
    faiss_path, json_path = _paths(doc_key)
    faiss.write_index(index["faiss"], faiss_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": index["meta"], "chunks": index["chunks"]}, f)
    _CACHE[doc_key] = index
    print(f"✅ RAG index built: {index['meta']['n_chunks']} chunks "
          f"(chunk_size={chunk_size}, overlap={overlap})")
    return index["meta"]["n_chunks"]


def load_index(doc_key: str):
    """Load an index from cache/disk. Returns None if missing or built with different settings."""
    if doc_key in _CACHE:
        return _CACHE[doc_key]
    faiss_path, json_path = _paths(doc_key)
    if not (os.path.exists(faiss_path) and os.path.exists(json_path)):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data["meta"]
        if (meta.get("embed_model") != EMBED_MODEL_NAME
                or meta.get("chunk_size") != CHUNK_SIZE
                or meta.get("overlap") != CHUNK_OVERLAP):
            return None      # stale: settings changed, caller should rebuild
        chunks = data["chunks"]
        index = {
            "chunks": chunks,
            "faiss": faiss.read_index(faiss_path),
            "bm25": BM25([tokenize(c) for c in chunks]),
            "meta": meta,
        }
    except Exception as e:
        print(f"⚠️ Could not load index {doc_key}: {e}")
        return None
    _CACHE[doc_key] = index
    return index


def index_is_current(doc_key: str) -> bool:
    return load_index(doc_key) is not None


def retrieve(doc_key: str, query: str, top_k: int = TOP_K, mode: str = RETRIEVAL_MODE):
    """Return the most relevant excerpts as one formatted string for the LLM prompt (or None)."""
    index = load_index(doc_key)
    if index is None:
        return None
    results = search_index(index, query, top_k, mode)
    if not results:
        return None
    parts = [f"[Retrieved {len(results)} excerpts from the document, most relevant first]"]
    for i, (chunk, _score) in enumerate(results, start=1):
        parts.append(f"[Excerpt {i}]\n{chunk}")
    return "\n\n".join(parts)


def delete_index(doc_key: str):
    """Remove a document's index from memory and disk."""
    _CACHE.pop(doc_key, None)
    for path in _paths(doc_key):
        if os.path.exists(path):
            os.remove(path)
    print(f"🗑️ Deleted RAG index for {doc_key}")