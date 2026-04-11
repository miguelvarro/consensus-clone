from __future__ import annotations

from typing import List, Dict


def get_eval_queries() -> List[Dict[str, str]]:
    """
    Conjunto mínimo de queries para evaluación manual.
    Cada entrada incluye:
    - id: identificador corto
    - q: pregunta
    - expected_mode: orientación esperada aproximada
      ("positive", "mixed", "insufficient")
    """
    return [
        {
            "id": "creatine_strength_general",
            "q": "Does creatine increase muscle strength?",
            "expected_mode": "positive",
        },
        {
            "id": "creatine_strength_mixed",
            "q": "Is evidence on creatine and strength mixed?",
            "expected_mode": "mixed",
        },
        {
            "id": "creatine_older_adults",
            "q": "Does creatine improve strength in older adults?",
            "expected_mode": "positive",
        },
        {
            "id": "creatine_inconclusive",
            "q": "Is the evidence on creatine inconclusive?",
            "expected_mode": "mixed",
        },
        {
            "id": "creatine_performance",
            "q": "Does creatine improve high-intensity exercise performance?",
            "expected_mode": "positive",
        },
        {
            "id": "creatine_null_effect",
            "q": "Is there no significant effect of creatine on strength?",
            "expected_mode": "mixed",
        },
    ]
