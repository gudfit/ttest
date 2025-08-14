# lldc/scripts/channel_analysis.py

from __future__ import annotations
from typing import Any, List, Dict, Tuple, Iterable
from collections import Counter, defaultdict
import math, json, random
import numpy as np
from sentence_transformers import SentenceTransformer
from lldc.utils.logging import setup_logging


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
) -> float:
    if len(texts) < 2:
        return 0.0
    enc = SentenceTransformer(model_name)
    idx = list(range(len(texts)))
    pairs = [
        (random.randrange(len(idx)), random.randrange(len(idx)))
        for _ in range(min(2000, len(texts) * 5))
    ]
    embs = enc.encode(
        texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
    )
    dup = 0
    for a, b in pairs:
        if a == b:
            continue
        if float((embs[a] * embs[b]).sum()) >= thr:
            dup += 1
    return 100.0 * dup / max(1, len(pairs))


class KN5:
    def __init__(self, n: int = 5, discount: float = 0.75, vocab_limit: int = 50000):
        self.n = n
        self.D = discount
        self.vocab_limit = vocab_limit
        self.counts = [Counter() for _ in range(n)]
        self.context_counts = [Counter() for _ in range(n)]
        self.vocab: List[str] = []

    def fit(self, texts: Iterable[str]):
        wc = Counter()
        for t in texts:
            toks = t.split()
            wc.update(toks)
            toks = ["<s>"] * (self.n - 1) + toks + ["</s>"]
            for i in range(len(toks)):
                for k in range(1, self.n + 1):
                    if i - k + 1 < 0:
                        break
                    gram = tuple(toks[i - k + 1 : i + 1])
                    self.counts[k - 1][gram] += 1
                    if k > 1:
                        ctx = gram[:-1]
                        self.context_counts[k - 1][ctx] += 1
        self.vocab = [w for w, _ in wc.most_common(self.vocab_limit)]

    def prob(self, hist: Tuple[str, ...], w: str) -> float:
        def kn_prob(k, ctx, w):
            if k == 1:
                cont = sum(1 for (x,), c in self.counts[0].items() if x == w)
                denom = len(self.counts[1]) if self.counts[1] else 1
                return cont / max(1, denom)
            gram = ctx + (w,)
            c = self.counts[k - 1][gram]
            ctx_c = self.context_counts[k - 1][ctx]
            if ctx_c > 0:
                lambda_ctx = (
                    self.D
                    * len(
                        [
                            g
                            for g, cnt in self.counts[k - 1].items()
                            if g[:-1] == ctx and cnt > 0
                        ]
                    )
                ) / ctx_c
                p_cont = kn_prob(k - 1, ctx[1:], w)
                return max(c - self.D, 0) / ctx_c + lambda_ctx * p_cont
            else:
                return kn_prob(k - 1, ctx[1:], w)

        k = min(self.n, len(hist) + 1)
        ctx = hist[-(k - 1) :] if k > 1 else tuple()
        return kn_prob(k, tuple(ctx), w)

    def argmax_next(self, hist: List[str]) -> str:
        best_w, best_p = None, -1.0
        ctx = tuple(hist[-(self.n - 1) :])
        for w in self.vocab:
            p = self.prob(ctx, w)
            if p > best_p:
                best_p, best_w = p, w
        return best_w or (self.vocab[0] if self.vocab else "</s>")


def regen_baseline_everyN_kn(text: str, N: int, kn5: KN5) -> str:
    toks = text.split()
    if not toks:
        return ""
    recon: List[str] = []
    for i in range(len(toks)):
        if i % N == 0:
            recon.append(toks[i])
        else:
            recon.append(kn5.argmax_next(recon))
    return " ".join(recon)


def run_channel_analysis(
    cfg: Any, recon_texts: List[str], orig_texts: List[str], kn5: KN5 | None = None
) -> Dict[str, float]:
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

    from lldc.metrics.fidelity import character_level_fidelity

    base_scores = {}
    for N in regen_Ns:
        if kn5 is None:
            recon_b: List[str] = []
            for t in orig_texts:
                toks = t.split()
                out: List[str] = []
                last_kept: str | None = None
                for i, w in enumerate(toks):
                    if i % N == 0:
                        out.append(w)
                        last_kept = w
                    else:
                        out.append(last_kept if last_kept is not None else w)
                recon_b.append(" ".join(out))
        else:
            recon_b = [regen_baseline_everyN_kn(t, N, kn5) for t in orig_texts]
        base_scores[f"regen_every_{N}_charF"] = float(
            np.mean(
                [character_level_fidelity(o, r) for o, r in zip(orig_texts, recon_b)]
            )
        )
    return {**cov, **base_scores}
