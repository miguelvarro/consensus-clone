from __future__ import annotations

from ai_consensus_clone.core.retrieval.online import OnlinePaperRetriever


def main():
    retriever = OnlinePaperRetriever()

    query = "creatine muscle strength"
    papers = retriever.search_openalex(query=query, n=5, oa_only=True)

    print(f"\nConsulta: {query}")
    print(f"Resultados recuperados: {len(papers)}\n")

    for i, p in enumerate(papers, start=1):
        print("=" * 80)
        print(f"[{i}] {p.title}")
        print(f"paper_id: {p.paper_id}")
        print(f"year: {p.year}")
        print(f"venue: {p.venue}")
        print(f"doi: {p.doi}")
        print(f"oa_url: {p.oa_url}")
        print(f"pdf_url: {p.pdf_url}")
        print(f"pmc_url: {p.pmc_url}")
        print(f"full_text_source: {p.full_text_source}")
        print(f"has_full_text: {bool((p.full_text or '').strip())}")
        print(f"full_text_len: {len((p.full_text or '').strip())}")
        print(f"authors: {p.authors[:3]}")
        print(f"abstract: {(p.abstract or '')[:300]}")
        print(f"full_text_preview: {(p.full_text or '')[:500]}")
        print()


if __name__ == "__main__":
    main()
