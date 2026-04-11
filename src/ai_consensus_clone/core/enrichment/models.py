from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class FullTextResult:
    text: Optional[str]
    source: Optional[str]
    url: Optional[str] = None
    quality_score: float = 0.0
    extracted_from: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool((self.text or "").strip())


@dataclass
class EnrichmentDecision:
    selected: Optional[FullTextResult]
    candidates: List[FullTextResult] = field(default_factory=list)

    @property
    def best_text(self) -> Optional[str]:
        if self.selected:
            return self.selected.text
        return None

    @property
    def best_source(self) -> Optional[str]:
        if self.selected:
            return self.selected.source
        return None
