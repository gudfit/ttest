# lldc/metrics/fidelity.py

from __future__ import annotations
from typing import Tuple, Optional, List
from Levenshtein import distance as levenshtein_distance
import evaluate
from sacrebleu.metrics import CHRF
import numpy as np
from sentence_transformers import SentenceTransformer


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


def semantic_span_fidelity(
    orig: str,
    recon: str,
    sbert_model: str = "sentence-transformers/all-mpnet-base-v2",
    span_chars: int = 256,
    stride_chars: int = 128,
) -> float:

    def _spans(s: str) -> List[str]:
        if not s:
            return []
        spans = []
        i = 0
        while i < len(s):
            spans.append(s[i : i + span_chars])
            if i + span_chars >= len(s):
                break
            i += stride_chars
        return spans

    A = _spans(orig)
    B = _spans(recon)
    if not A and not B:
        return 0.0
    L = max(len(A), len(B))
    if len(A) < L and A:
        A += [A[-1]] * (L - len(A))
    if len(B) < L and B:
        B += [B[-1]] * (L - len(B))

    enc = SentenceTransformer(sbert_model)
    emA = (
        enc.encode(
            A, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
        )
        if A
        else np.zeros((L, 768))
    )
    emB = (
        enc.encode(
            B, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
        )
        if B
        else np.zeros((L, 768))
    )
    sims = (emA * emB).sum(axis=1)
    return float(np.mean(sims) * 100.0) if sims.size else 0.0
