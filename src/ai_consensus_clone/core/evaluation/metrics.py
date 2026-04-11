from __future__ import annotations

from typing import Dict, Any, List

from ai_consensus_clone.core.evaluation.datasets import get_eval_queries
from ai_consensus_clone.core.reasoning.answer import AnswerService
from ai_consensus_clone.core.retrieval.bm25 import BM25Search


def infer_mode_from_answer(conclusion: str) -> str:
    """
    Heurística simple para inferir el modo de la respuesta a partir de la conclusión.
    Prioriza mixed antes que positive para evitar falsos positivos cuando una
    conclusión diga cosas como "mixta o limitada" y también mencione "efecto positivo".
    """
    c = (conclusion or "").lower()

    if any(x in c for x in ("mixta", "mixto", "inconcluyente", "no concluyente", "limitada")):
        return "mixed"

    if any(x in c for x in ("efecto positivo", "apunta a un efecto positivo", "sugiere un efecto positivo")):
        return "positive"

    if any(x in c for x in ("no es suficiente", "no permite", "no se han encontrado")):
        return "insufficient"

    return "unknown"


def run_eval(search: BM25Search, k: int = 5) -> List[Dict[str, Any]]:
    svc = AnswerService(search=search)
    queries = get_eval_queries()

    results: List[Dict[str, Any]] = []

    for item in queries:
        q = item["q"]
        expected_mode = item["expected_mode"]

        answer = svc.answer(q, k=k)

        predicted_mode = infer_mode_from_answer(answer.get("conclusion", ""))

        results.append(
            {
                "id": item["id"],
                "q": q,
                "expected_mode": expected_mode,
                "predicted_mode": predicted_mode,
                "confidence": answer.get("confidence"),
                "n_citations": len(answer.get("citations", [])),
                "n_evidences": len(answer.get("evidences", [])),
                "conclusion": answer.get("conclusion"),
                "ok": predicted_mode == expected_mode,
            }
        )

    return results


def summarize_eval(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    ok = sum(1 for r in results if r["ok"])
    accuracy = ok / total if total else 0.0

    confidence_counts: Dict[str, int] = {}
    for r in results:
        conf = r.get("confidence", "unknown")
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

    return {
        "total_queries": total,
        "correct_mode_predictions": ok,
        "mode_accuracy": round(accuracy, 3),
        "confidence_distribution": confidence_counts,
    }
