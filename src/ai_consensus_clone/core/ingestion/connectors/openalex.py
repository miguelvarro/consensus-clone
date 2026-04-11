from __future__ import annotations
from typing import Any, Dict, List, Optional
import httpx

class OpenAlexClient:
    def __init__(self, base_url: str = "https://api.openalex.org", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def search_works(self, query: str, per_page: int = 25, filters: Optional[str] = None) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/works"
        params = {"search": query, "per-page": per_page}
        if filters:
            params["filter"] = filters
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        return data.get("results", [])


def extract_doi(work: Dict[str, Any]) -> Optional[str]:
    doi = work.get("doi")
    if doi and doi.startswith("https://doi.org/"):
        return doi.replace("https://doi.org/", "")
    return doi

def _first_non_empty(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    return None

def extract_oa_landing_url(work: Dict[str, Any]) -> Optional[str]:
    """
    Devuelve una landing page OA si existe (no DOI).
    Prioriza best_oa_location y luego locations.
    """
    best = work.get("best_oa_location") or {}
    if best:
        # landing OA
        url = _first_non_empty(best.get("landing_page_url"), best.get("url"))
        if url and "doi.org" not in url:
            return url

    # fallback: locations
    for loc in (work.get("locations") or []):
        url = _first_non_empty(loc.get("landing_page_url"), loc.get("url"))
        if url and "doi.org" not in url:
            return url

    # open_access.oa_url a veces está, pero si apunta a doi.org no lo queremos
    oa = work.get("open_access") or {}
    url = oa.get("oa_url")
    if url and "doi.org" not in url:
        return url

    return None

def extract_oa_pdf_url(work: Dict[str, Any]) -> Optional[str]:
    """
    Devuelve un PDF URL OA si existe.
    """
    best = work.get("best_oa_location") or {}
    pdf = _first_non_empty(best.get("pdf_url"))
    if pdf:
        return pdf

    for loc in (work.get("locations") or []):
        pdf = _first_non_empty(loc.get("pdf_url"))
        if pdf:
            return pdf

    return None

def extract_abstract(work: Dict[str, Any]) -> Optional[str]:
    """
    OpenAlex puede traer el abstract como abstract_inverted_index.
    Esto lo reconstruye a texto.
    """
    inv = work.get("abstract_inverted_index")
    if not inv:
        return None

    # inv: { "word": [pos1, pos2, ...], ...}
    # reconstruimos lista de tokens por posición
    max_pos = -1
    for positions in inv.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    if max_pos < 0:
        return None

    tokens = [""] * (max_pos + 1)
    for word, positions in inv.items():
        for p in positions:
            if 0 <= p <= max_pos:
                tokens[p] = word

    text = " ".join(t for t in tokens if t)
    return text.strip() or None

