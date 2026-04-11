import argparse
import orjson

from typing import Any, Dict, Iterable

from ai_consensus_clone.core.domain.paper import Paper
from ai_consensus_clone.core.enrichment.fulltext import FullTextEnricher


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                yield orjson.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "wb") as f:
        for r in rows:
            f.write(orjson.dumps(r) + b"\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    rows = list(iter_jsonl(args.input))
    enricher = FullTextEnricher()

    updated = 0

    for i, obj in enumerate(rows):
        if updated >= args.limit:
            break

        paper = Paper(**obj)

        before = paper.full_text
        paper = enricher.enrich(paper)

        rows[i] = paper.model_dump()

        if not before and paper.full_text:
            updated += 1

    write_jsonl(args.output, rows)

    print(f"Enriquecidos: {updated}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
