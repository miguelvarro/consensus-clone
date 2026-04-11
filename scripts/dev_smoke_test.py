from ai_consensus_clone.core.retrieval.bm25 import BM25Search

def main():
    idx = BM25Search.build_from_jsonl("data/processed/papers_sample.jsonl")
    print(idx.search("creatine strength training", k=5))

if __name__ == "__main__":
    main()

