# lldc/scripts/channel_analysis.py

from __future__ import annotations
from typing import Any, List, Dict, Tuple, Iterable
from collections import Counter, defaultdict
import math, json, random, logging
import numpy as np
from sentence_transformers import SentenceTransformer
from datasketch import MinHash, MinHashLSH

from lldc.utils.logging import setup_logging
from lldc.metrics.fidelity import character_level_fidelity

from lldc.baselines.kenlm_subsample import (
    subsample_and_reconstruct_kenlm5,
)


def _cfg_get(cfg: Any, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def ngram_topk_coverage(texts: List[str], n: int, top_k: int) -> float:
    grams = []
    for t in texts:
        toks = t.split()
        grams += [
            " ".join(toks[i : i + n]) for i in range(0, max(0, len(toks) - n + 1))
        ]
    cnt = Counter(grams)
    total = sum(cnt.values())
    if total == 0:
        return 0.0
    top = sum(c for _, c in cnt.most_common(top_k))
    return 100.0 * top / total


def semantic_dedup_coverage(
    texts: List[str],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    thr: float = 0.9,
    num_perm: int = 256,
) -> float:
    log = logging.getLogger("lldc")
    if len(texts) < 2:
        return 0.0

    log.info(f"[semantic_dedup] Encoding {len(texts)} texts with {model_name}...")
    enc = SentenceTransformer(model_name)
    embs = enc.encode(
        texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
    )
    log.info("[semantic_dedup] Creating MinHash signatures...")
    minhashes = []
    for vec in embs:
        m = MinHash(num_perm=num_perm)
        for i, val in enumerate(vec):
            if val > 0:
                m.update(f"dim_{i}_pos".encode("utf8"))
            else:
                m.update(f"dim_{i}_neg".encode("utf8"))
        minhashes.append(m)

    lsh = MinHashLSH(threshold=(1 + thr) / 2, num_perm=num_perm)
    for i, m in enumerate(minhashes):
        lsh.insert(f"doc_{i}", m)

    log.info("[semantic_dedup] Querying LSH index for duplicates...")
    duplicate_pairs = set()
    for i, m in enumerate(minhashes):
        result = lsh.query(m)
        for res_key in result:
            j = int(res_key.split("_")[1])
            if i < j:
                cosine_sim = float((embs[i] * embs[j]).sum())
                if cosine_sim >= thr:
                    duplicate_pairs.add((i, j))

    total_possible_pairs = len(texts) * (len(texts) - 1) / 2
    if total_possible_pairs == 0:
        return 0.0
    return 100.0 * len(duplicate_pairs) / max(1, total_possible_pairs)


def run_channel_analysis(
    cfg: Any,
    recon_texts: List[str],
    orig_texts: List[str],
    train_texts: List[str] | None = None,
) -> Dict[str, float]:
    """
    Compute channel statistics on reconstructions and a *standard* n-gram baseline
    using KenLM (5-gram). This replaces the ad-hoc KN5 for Experiment 2B to align
    with the methodology claim.
    """
    n_list = list(
        _cfg_get(
            cfg,
            "deduplication.ngram.n_list",
            _cfg_get(cfg, "experiment.deduplication.ngram.n_list", []),
        )
    )
    top_k = int(
        _cfg_get(
            cfg,
            "deduplication.ngram.top_k",
            _cfg_get(cfg, "experiment.deduplication.ngram.top_k", 10),
        )
    )
    sbert_model = _cfg_get(
        cfg,
        "deduplication.semantic.sbert_model",
        _cfg_get(
            cfg,
            "experiment.deduplication.semantic.sbert_model",
            "sentence-transformers/all-mpnet-base-v2",
        ),
    )
    cov = {}
    for n in n_list:
        cov[f"ngram{n}_top{top_k}_coverage_pct"] = ngram_topk_coverage(
            recon_texts, n, top_k
        )
    cov["semantic_dup_pct"] = semantic_dedup_coverage(recon_texts, sbert_model)
    regen_Ns = list(
        _cfg_get(
            cfg,
            "regen_baseline.subsample_every_N",
            _cfg_get(cfg, "experiment.regen_baseline.subsample_every_N", []),
        )
    )
    workdir = str(
        _cfg_get(
            cfg,
            "regen_baseline.kenlm_workdir",
            _cfg_get(
                cfg,
                "experiment.regen_baseline.kenlm_workdir",
                "artifacts/runs/kenlm_5gram",
            ),
        )
    )
    base_scores: Dict[str, float] = {}
    if train_texts and regen_Ns:
        try:
            outputs = subsample_and_reconstruct_kenlm5(
                test_texts=orig_texts,
                train_texts=train_texts,
                rates=regen_Ns,
                workdir=workdir,
                beam_size=int(_cfg_get(cfg, "regen_baseline.beam_size", 16)),
                cand_per_step=int(_cfg_get(cfg, "regen_baseline.cand_per_step", 200)),
                max_vocab=int(_cfg_get(cfg, "regen_baseline.max_vocab", 50000)),
            )
            for item in outputs:
                N = int(item["rate_N"])
                recons = item["reconstructions"]
                score = float(
                    np.mean(
                        [
                            character_level_fidelity(o, r)
                            for o, r in zip(orig_texts, recons)
                        ]
                    )
                )
                base_scores[f"regen_every_{N}_charF"] = score
        except Exception:
            pass
    return {**cov, **base_scores}
