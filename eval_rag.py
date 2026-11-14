# eval_rag.py - Evaluate the RAG pipeline: retrieval quality, answer quality, latency
"""
Usage (from the project folder, venv active):

    python eval_rag.py                    full evaluation with the settings in rag.py
    python eval_rag.py --retrieval-only   retrieval metrics only (no LLM calls, free + fast)
    python eval_rag.py --ablate           compare retrieval modes x chunk sizes (no LLM calls)

Test set: eval_set.json  (copy eval_set.example.json and fill it with 30-40 questions)
Each entry:
    "document":  the document's file name as shown in the Documents tab (partial match is fine)
    "question":  what you would ask JARVIS
    "reference": the correct answer, one line
    "evidence":  a short VERBATIM phrase (5-12 words) from the passage that answers it
"""
import os
import sys
import json
import time
import argparse

import numpy as np

import rag
from database import get_all_documents
from doc_reader import extract_text, DocumentError

TESTSET_PATH = os.path.join(rag.BASE_DIR, "eval_set.json")
UPLOAD_FOLDER = os.path.join(rag.BASE_DIR, "uploads")
PAUSE_BETWEEN_QUESTIONS_S = 2.0      # stay under Groq free-tier rate limits

ABLATION_MODES = ["dense", "bm25", "hybrid"]
ABLATION_CHUNKS = [(100, 25), (160, 40), (250, 60)]   # (chunk_size, overlap) in words
ABLATION_KS = (1, 3, 5)


def norm(s: str) -> str:
    """Lower-case and collapse whitespace, so PDF line breaks don't break matching."""
    return " ".join(s.lower().split())


def name_key(s: str) -> str:
    """Compare file names ignoring case, spaces, underscores and hyphens (uploads store 'My Notes' as 'My_Notes')."""
    return norm(s.replace("_", " ").replace("-", " "))


# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────

def load_testset():
    if not os.path.exists(TESTSET_PATH):
        sys.exit(f"❌ {TESTSET_PATH} not found. Copy eval_set.example.json to eval_set.json and fill it in.")
    with open(TESTSET_PATH, encoding="utf-8") as f:
        rows = json.load(f)

    docs = {name_key(d["original_name"]): d for d in get_all_documents()}
    texts, items = {}, []
    for n, row in enumerate(rows, start=1):
        want = name_key(row["document"])
        doc = docs.get(want) or next((d for name, d in docs.items() if want in name), None)
        if doc is None:
            print(f"⚠️  #{n}: no uploaded document matches '{row['document']}'. "
                  f"Available: {', '.join(d['original_name'] for d in docs.values()) or 'none'}")
            continue
        if doc["id"] not in texts:
            try:
                texts[doc["id"]] = extract_text(os.path.join(UPLOAD_FOLDER, doc["filename"]))
            except DocumentError as e:
                print(f"⚠️  {doc['original_name']}: {e}")
                continue
        if norm(row["evidence"]) not in norm(texts[doc["id"]]):
            print(f"⚠️  #{n}: evidence phrase not found verbatim in {doc['original_name']} - fix it: "
                  f"\"{row['evidence'][:60]}\"")
            continue
        items.append({**row, "doc_id": doc["id"]})

    if not items:
        sys.exit("❌ No valid test questions. Fix the warnings above.")
    print(f"✅ Loaded {len(items)} valid questions from {len(texts)} document(s)")
    return items, texts


_INDEX_CACHE = {}


def get_index(texts, doc_id, chunk_size, overlap):
    key = (doc_id, chunk_size, overlap)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = rag.make_index(texts[doc_id], chunk_size, overlap)
    return _INDEX_CACHE[key]


def rank_of_evidence(results, evidence):
    """1-based rank of the first retrieved chunk containing the evidence, or None."""
    ev = norm(evidence)
    for rank, (chunk, _score) in enumerate(results, start=1):
        if ev in norm(chunk):
            return rank
    return None


# ─────────────────────────────────────────
# RETRIEVAL METRICS
# ─────────────────────────────────────────

def retrieval_metrics(items, texts, chunk_size, overlap, mode, ks):
    max_k = max(ks)
    ranks, latencies = [], []
    for it in items:
        index = get_index(texts, it["doc_id"], chunk_size, overlap)
        t0 = time.perf_counter()
        results = rag.search_index(index, it["question"], max_k, mode)
        latencies.append((time.perf_counter() - t0) * 1000)
        ranks.append(rank_of_evidence(results, it["evidence"]))

    n = len(items)
    metrics = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in ks}
    metrics["mrr"] = float(np.mean([1 / r if r else 0 for r in ranks]))
    metrics["retrieval_p50_ms"], metrics["retrieval_p95_ms"] = (float(x) for x in np.percentile(latencies, [50, 95]))
    return metrics, ranks


def run_ablation(items, texts):
    print("\n===== ABLATION (retrieval only) =====")
    rows = []
    for chunk_size, overlap in ABLATION_CHUNKS:
        for mode in ABLATION_MODES:
            m, _ = retrieval_metrics(items, texts, chunk_size, overlap, mode, ABLATION_KS)
            rows.append({"mode": mode, "chunk_size": chunk_size, "overlap": overlap, **m})

    header = "| mode | chunk | " + " | ".join(f"Hit@{k}" for k in ABLATION_KS) + " | MRR |"
    print(header)
    print("|" + "---|" * (len(ABLATION_KS) + 3))
    for r in rows:
        hits = " | ".join(f"{r[f'hit@{k}']:.1%}" for k in ABLATION_KS)
        print(f"| {r['mode']} | {r['chunk_size']} | {hits} | {r['mrr']:.3f} |")

    best = max(rows, key=lambda r: (r["mrr"], r["hit@3"]))
    print(f"\n🏆 Best by MRR: mode={best['mode']}, chunk_size={best['chunk_size']}, overlap={best['overlap']} "
          f"(Hit@3 {best['hit@3']:.1%}, MRR {best['mrr']:.3f})")
    print("   Put those values in rag.py (RETRIEVAL_MODE, CHUNK_SIZE, CHUNK_OVERLAP), then run the full evaluation.")
    return rows


# ─────────────────────────────────────────
# GENERATION METRICS (LLM answer + LLM-as-judge)
# ─────────────────────────────────────────

def llm_text(prompt_messages):
    from agent import _llm
    resp = _llm(prompt_messages, None, effort="low", max_tokens=1500, temperature=0)
    return (resp.choices[0].message.content or "").strip()


def generate_answer(question, context):
    return llm_text([
        {"role": "system", "content": "Answer the question using ONLY the context below. Be specific. "
                                      "If the context does not contain the answer, say you don't know.\n\n"
                                      f"CONTEXT:\n{context}"},
        {"role": "user", "content": question},
    ])


def judge(prompt):
    return llm_text([{"role": "user", "content": prompt}]).upper().startswith("YES")


def full_eval(items, texts, retrieval_only=False):
    top_k, mode = rag.TOP_K, rag.RETRIEVAL_MODE
    cs, ov = rag.CHUNK_SIZE, rag.CHUNK_OVERLAP
    rows, correct, faithful, e2e_ms = [], [], [], []
    ranks, ret_ms = [], []

    for i, it in enumerate(items, start=1):
        index = get_index(texts, it["doc_id"], cs, ov)
        t0 = time.perf_counter()
        results = rag.search_index(index, it["question"], top_k, mode)
        ret_ms.append((time.perf_counter() - t0) * 1000)
        rank = rank_of_evidence(results, it["evidence"])
        ranks.append(rank)
        row = {"question": it["question"], "rank": rank}

        if not retrieval_only:
            context = "\n\n".join(c for c, _ in results)
            answer = generate_answer(it["question"], context)
            e2e_ms.append((time.perf_counter() - t0) * 1000)     # retrieval + generation (judging excluded)
            ok = judge(f"Question: {it['question']}\nReference answer: {it['reference']}\n"
                       f"Candidate answer: {answer}\n\nDoes the candidate answer agree with the reference "
                       "answer on the key facts? Reply with exactly YES or NO.")
            grounded = judge(f"Context:\n{context}\n\nAnswer: {answer}\n\nIs every claim in the answer "
                             "supported by the context? Reply with exactly YES or NO.")
            correct.append(ok)
            faithful.append(grounded)
            row.update({"answer": answer, "correct": ok, "faithful": grounded})
            time.sleep(PAUSE_BETWEEN_QUESTIONS_S)

        rows.append(row)
        print(f"  [{i}/{len(items)}] rank={rank} {'✓' if rank else '✗'}  {it['question'][:60]}")

    n = len(items)
    summary = {
        "n_questions": n, "mode": mode, "chunk_size": cs, "overlap": ov, "top_k": top_k,
        f"hit@{top_k}": sum(1 for r in ranks if r) / n,
        "mrr": float(np.mean([1 / r if r else 0 for r in ranks])),
        "retrieval_p50_ms": float(np.percentile(ret_ms, 50)),
        "retrieval_p95_ms": float(np.percentile(ret_ms, 95)),
    }
    if not retrieval_only:
        summary.update({
            "correctness": sum(correct) / n,
            "faithfulness": sum(faithful) / n,
            "e2e_p50_ms": float(np.percentile(e2e_ms, 50)),
            "e2e_p95_ms": float(np.percentile(e2e_ms, 95)),
        })
    return summary, rows


def print_summary(s):
    print(f"\n===== RAG EVALUATION  ({s['n_questions']} questions) =====")
    print(f"Config:            mode={s['mode']}, chunk={s['chunk_size']}w, overlap={s['overlap']}w, top_k={s['top_k']}")
    k = s["top_k"]
    hit = s["hit@" + str(k)]
    print(f"Hit Rate@{k}:       {hit:.1%}")
    print(f"MRR:               {s['mrr']:.3f}")
    if "correctness" in s:
        print(f"Answer correctness:{s['correctness']:.1%}   (LLM-as-judge vs your reference answers)")
        print(f"Faithfulness:      {s['faithfulness']:.1%}   (answer supported by retrieved context)")
    print(f"Retrieval latency: p50 {s['retrieval_p50_ms']:.0f} ms / p95 {s['retrieval_p95_ms']:.0f} ms")
    if "e2e_p50_ms" in s:
        print(f"End-to-end:        p50 {s['e2e_p50_ms']:.0f} ms / p95 {s['e2e_p95_ms']:.0f} ms (retrieval + LLM answer)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate JARVIS's RAG pipeline")
    parser.add_argument("--ablate", action="store_true", help="compare modes and chunk sizes (retrieval only)")
    parser.add_argument("--retrieval-only", action="store_true", help="skip LLM answer + judge calls")
    args = parser.parse_args()

    items, texts = load_testset()
    rag.warmup()

    if args.ablate:
        rows = run_ablation(items, texts)
        with open(os.path.join(rag.BASE_DIR, "eval_ablation.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print("💾 Saved eval_ablation.json")
        return

    summary, rows = full_eval(items, texts, retrieval_only=args.retrieval_only)
    print_summary(summary)
    with open(os.path.join(rag.BASE_DIR, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "questions": rows}, f, indent=2)
    print("💾 Saved eval_results.json  (open it to inspect every miss)")


if __name__ == "__main__":
    main()