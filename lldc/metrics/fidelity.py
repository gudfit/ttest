from __future__ import annotations
from typing import Tuple, Optional
from Levenshtein import distance as levenshtein_distance
import evaluate
from sacrebleu.metrics import CHRF


def character_level_fidelity(orig: str, recon: str) -> float:
    if len(orig) == len(recon):
        equal = sum(1 for a, b in zip(orig, recon) if a == b)
        return 100.0 * equal / max(1, len(orig))
    d = levenshtein_distance(orig, recon)
    denom = max(len(orig), len(recon), 1)
    return 100.0 * (1.0 - d / denom)


def chrf_score(orig: str, recon: str, order: int = 6) -> float:
    chrf = CHRF(word_order=0, char_order=order)
    return float(chrf.sentence_score(recon, [orig]).score)


def bertscore_f1(
    orig: str,
    recon: str,
    model_type: str = "roberta-large",
    batch_size: Optional[int] = None,
) -> float:
    metric = evaluate.load("bertscore")
    res = metric.compute(
        references=[orig],
        predictions=[recon],
        model_type=model_type,
        batch_size=batch_size or 16,
    )
    return float(res["f1"][0] * 100.0)
