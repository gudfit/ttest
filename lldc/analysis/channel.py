# lldc/analysis/channel.py

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import datasketch, torch
import itertools
from datasketch import MinHash
from collections import Counter
from sentence_transformers import SentenceTransformer

from lldc.baselines.kenlm_subsample import subsample_and_reconstruct_kenlm5
from lldc.metrics.crumpled_paper import Oracle, tcm_pcm_from_texts
from lldc.utils import wandb_log


@dataclass
class ChannelConfig:
    out_dir: Path
    ngram_n: int = 3
    topk: int = 10
    minhash_perm: int = 64
    sbert_model: str = "sentence-transformers/all-mpnet-base-v2"
    subsample_rates: List[int] = (3, 5, 7)
    beam_size: int = 16
    cand_per_step: int = 200
    max_vocab: int = 50000
    scales: List[int] = (1, 2, 4, 8)
    Nc: int = 10000


def _load_recons(paths: List[Path]) -> List[dict]:
    out = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for ln in f:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    continue
    return out


def _ngram_topk_coverage(texts: List[str], n: int, k: int) -> float:

    cnt = Counter()
    for t in texts:
        toks = t.split()
        for i in range(0, max(0, len(toks) - n + 1)):
            cnt[tuple(toks[i : i + n])] += 1
    total = sum(cnt.values())
    if total == 0:
        return 0.0
    top = sum(v for _, v in cnt.most_common(k))
    return float(top) / float(total)


def _semantic_minhash_stats(texts: List[str], perms: int, sbert_model: str) -> dict:
    try:

        enc = SentenceTransformer(sbert_model)
        reps = enc.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        reps = reps / reps.norm(dim=-1, keepdim=True)
        vs = reps.cpu().numpy()
        sigs = []
        for v in vs:
            mh = MinHash(num_perm=perms)
            for x in v.tolist():
                mh.update(bytes(f"{x:.5f}", "utf-8"))
            sigs.append(mh)
        sims = [
            sigs[i].jaccard(sigs[j])
            for i, j in itertools.combinations(range(len(sigs)), 2)
        ]
        return {"minhash_mean": float(np.mean(sims) if sims else 0.0)}
    except Exception as e:
        return {"error": f"minhash_unavailable: {e}"}


def run_channel(
    test_texts: List[str],
    recon_texts: List[str],
    payload_bits_total: int,
    static_bits: int,
    base_chars: int,
    cfg: ChannelConfig,
) -> Path:
    run = wandb_log.start(cfg, run_name="E2B-channel", tags=["e2b", "channel"])
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    cov_topk = _ngram_topk_coverage(recon_texts, n=cfg.ngram_n, k=cfg.topk)
    sem = _semantic_minhash_stats(
        recon_texts, perms=cfg.minhash_perm, sbert_model=cfg.sbert_model
    )
    sem_jaccard = sem.get("minhash_mean", 0.0)
    wandb_log.log(
        {
            "e2b/ngram_topk_cov": float(cov_topk),
            "e2b/semantic_minhash_jaccard": float(sem_jaccard),
        }
    )

    bpc_pm = payload_bits_total / max(1, base_chars)

    base = subsample_and_reconstruct_kenlm5(
        test_texts=test_texts,
        train_texts=test_texts,
        rates=list(cfg.subsample_rates),
        workdir=cfg.out_dir / "kenlm5",
        beam_size=cfg.beam_size,
        cand_per_step=cfg.cand_per_step,
        max_vocab=cfg.max_vocab,
    )
    baseline_eval = []
    for rec in base:
        fids, chars = [], 0
        for o, r in zip(test_texts, rec["reconstructions"]):
            match = sum(1 for a, b in zip(o, r) if a == b)
            denom = max(len(o), len(r), 1)
            fids.append(match / denom)
            chars += max(1, len(o))
        bits = sum(len(p.encode("utf-8")) for p in rec["subsamples_payload"]) * 8
        bpc = bits / max(1, chars)
        baseline_eval.append(
            {
                "rate_N": rec["rate_N"],
                "fidelity_mean": float(np.mean(fids) if fids else 0.0),
                "bpc": bpc,
            }
        )

    oracle = Oracle("gpt2-large")
    tcms, pcms = [], []
    for o, r in zip(test_texts[:64], recon_texts[:64]):
        m = tcm_pcm_from_texts(o, r, oracle)
        tcms.append(m["tcm_bits"])
        pcms.append(m["pcm_bits"])

    def amortised_bpc(
        bpc_runtime: float,
        static_bits: int,
        base_chars: int,
        copies: int = 1,
        scale: int = 1,
    ) -> float:
        denom = max(1, base_chars * copies * scale)
        return static_bits / denom + bpc_runtime

    fids_main = []
    for o, r in zip(test_texts, recon_texts):
        match = sum(1 for a, b in zip(o, r) if a == b)
        denom = max(len(o), len(r), 1)
        fids_main.append(match / denom)
    fid = float(np.mean(fids_main) if fids_main else 0.0)

    scales = list(cfg.scales)
    scale_records = []
    for s in scales:
        am_bpc = amortised_bpc(bpc_pm, static_bits, base_chars, copies=cfg.Nc, scale=s)
        scale_records.append({"scale": s, "amortised_bpc": am_bpc})
        wandb_log.log(
            {
                "e2b/scale": int(s),
                "e2b/amortised_bpc": float(am_bpc),
                "e2b/fid": float(fid),
            }
        )

    out = {
        "dedup": {
            "ngram_topk_coverage": cov_topk,
            "semantic_minhash": sem,
            "pm_bpc": bpc_pm,
        },
        "baseline_subsample_5gram_viterbi": baseline_eval,
        "scalability": scale_records,
        "TCM_mean": float(np.mean(tcms) if tcms else 0.0),
        "PCM_max": float(np.max(pcms) if pcms else 0.0),
    }
    out_path = cfg.out_dir / "E2B_channel.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    wandb_log.finish()
    return out_path
