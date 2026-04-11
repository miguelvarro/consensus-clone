from __future__ import annotations

import io
from typing import Optional

import httpx
from pypdf import PdfReader

from ai_consensus_clone.core.enrichment.models import FullTextResult
from ai_consensus_clone.core.enrichment.normalize import normalize_full_text
from ai_consensus_clone.core.enrichment.quality import quality_score


class PDFFetcher:
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "ai-consensus-clone/1.0",
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        }

    def fetch(self, url: str) -> FullTextResult:
        try:
            with httpx.Client(timeout=self.timeout, headers=self.headers, follow_redirects=True) as client:
                r = client.get(url)
                r.raise_for_status()

                content = r.content or b""
                content_type = (r.headers.get("content-type") or "").lower()

                looks_like_pdf = (
                    "application/pdf" in content_type
                    or url.lower().endswith(".pdf")
                    or str(r.url).lower().endswith(".pdf")
                    or content[:5] == b"%PDF-"
                )

                if not looks_like_pdf:
                    return FullTextResult(
                        text=None,
                        source=None,
                        url=str(r.url),
                        notes=["response does not look like PDF"],
                    )

                reader = PdfReader(io.BytesIO(content))
                parts = []

                for page in reader.pages:
                    try:
                        txt = page.extract_text() or ""
                    except Exception:
                        txt = ""
                    txt = normalize_full_text(txt)
                    if txt:
                        parts.append(txt)

                merged = normalize_full_text("\n".join(parts))
                score = quality_score(merged, "pdf")

                return FullTextResult(
                    text=merged,
                    source="pdf" if merged else None,
                    url=str(r.url),
                    quality_score=score,
                    extracted_from="pdf",
                )
        except Exception as e:
            return FullTextResult(
                text=None,
                source=None,
                url=url,
                notes=[f"pdf fetch error: {type(e).__name__}"],
            )
