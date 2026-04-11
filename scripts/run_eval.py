from __future__ import annotations

from pprint import pprint

from ai_consensus_clone.core.retrieval.bm25 import BM25Search
from ai_consensus_clone.core.evaluation.metrics import run_eval, summarize_eval


def main():
    search = BM25Search.from_disk("data/indices/bm25")
    results = run_eval(search=search, k=5)
    summary = summarize_eval(results)

    print("\n=== SUMMARY ===")
    pprint(summary)

    print("\n=== DETAILS ===")
    for r in results:
        print("-" * 80)
        print(f"ID: {r['id']}")
        print(f"Q: {r['q']}")
        print(f"Expected: {r['expected_mode']}")
        print(f"Predicted: {r['predicted_mode']}")
        print(f"Confidence: {r['confidence']}")
        print(f"Citations: {r['n_citations']}")
        print(f"Evidences: {r['n_evidences']}")
        print(f"OK: {r['ok']}")
        print(f"Conclusion: {r['conclusion']}")


if __name__ == "__main__":
    main()
