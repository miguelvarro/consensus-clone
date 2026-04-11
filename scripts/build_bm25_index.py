import argparse
from ai_consensus_clone.core.retrieval.bm25 import BM25Search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL con papers normalizados")
    ap.add_argument("--outdir", required=True, help="Directorio del índice BM25")
    args = ap.parse_args()

    idx = BM25Search.build_from_jsonl(args.input)
    idx.save_to_disk(args.outdir)
    print(f"BM25 index guardado en: {args.outdir}")

if __name__ == "__main__":
    main()

