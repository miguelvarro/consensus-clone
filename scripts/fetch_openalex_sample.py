import argparse
import orjson
from ai_consensus_clone.utils.text import clean_text
from ai_consensus_clone.core.ingestion.connectors.openalex import (
    OpenAlexClient, extract_doi, extract_oa_landing_url, extract_oa_pdf_url, extract_abstract
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="consulta para OpenAlex")
    ap.add_argument("--n", type=int, default=5, help="número de works")
    ap.add_argument("--out", required=True, help="ruta salida JSONL")
    args = ap.parse_args()

    client = OpenAlexClient()
    works = client.search_works(args.q, per_page=args.n, filters="is_oa:true")

    with open(args.out, "wb") as f:
        for w in works:
            paper = {
		    "paper_id": (w.get("id") or "").split("/")[-1],
		    "title": clean_text(w.get("title") or ""),
		    "abstract": (clean_text(extract_abstract(w) or "") or None),
		    "year": w.get("publication_year"),
		    "venue": (w.get("primary_location") or {}).get("source", {}).get("display_name"),
		    "doi": extract_doi(w),
		    "authors": [
			(a.get("author") or {}).get("display_name")
			for a in (w.get("authorships") or [])
			if (a.get("author") or {}).get("display_name")
		    ],
		    "oa_url": extract_oa_landing_url(w),
		    "pdf_url": extract_oa_pdf_url(w),   
		    "full_text": None,
		}
            f.write(orjson.dumps(paper) + b"\n")

    print(f"Guardado: {args.out}")

if __name__ == "__main__":
    main()

