from typing import Optional
from ai_consensus_clone.core.domain.paper import Paper

def year_filter(p: Paper, year_from: Optional[int], year_to: Optional[int]) -> bool:
    if p.year is None:
        return True
    if year_from is not None and p.year < year_from:
        return False
    if year_to is not None and p.year > year_to:
        return False
    return True

