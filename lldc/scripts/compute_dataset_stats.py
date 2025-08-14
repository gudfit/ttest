# lldc/scripts/compute_dataset_stats.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Dict, Optional

import json
import math
import itertools
import hydra
from omegaconf import OmegaConf
from datasets import load_dataset

from lldc.utils.paths import Paths
from lldc.utils.logging import setup_logging

from lldc.metrics.entropy_mi import (
    unigram_entropy_bits_per_symbol,
    avg_token_length_bytes,
    entropy_per_byte,
    mutual_information_adjacent,
)

from lldc.baselines.kenlm_subsample import kenlm_ngram_bpc


@dataclass
class _StatsCfg:
    compute_unigram_entropy: bool = True
    compute_ngram_entropy: Iterable[int] = ()
    compute_mi: bool = False


def _read_stats_cfg(cfg: Any) -> _StatsCfg:
    d = getattr(getattr(cfg, "data", None), "stats", None) or getattr(
        cfg, "stats", None
    )
    if d is None:
        return _StatsCfg()
    return _StatsCfg(
        compute_unigram_entropy=bool(getattr(d, "compute_unigram_entropy", True)),
        compute_ngram_entropy=list(getattr(d, "compute_ngram_entropy", []) or []),
        compute_mi=bool(getattr(d, "compute_mi", False)),
    )


def _word_tokens(texts: Iterable[str]) -> List[str]:
    toks: List[str] = []
    for t in texts:
        toks.extend((t or "").split())
    return toks


def _safe_list(ds_split, text_field: str, limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    n_limit = limit or 50_000

    if hasattr(ds_split, "take"):
        for ex in ds_split.take(n_limit):
            out.append(ex.get(text_field, "") or "")
        return out

    try:
        rng = (
            range(len(ds_split)) if limit is None else range(min(limit, len(ds_split)))
        )
        for i in rng:
            out.append(ds_split[i].get(text_field, "") or "")
        return out
    except Exception:
        it = iter(ds_split)
        for ex in itertools.islice(it, n_limit):
            if isinstance(ex, dict):
                out.append(ex.get(text_field, "") or "")
            else:
                out.append(str(ex) or "")
        return out


def main():
    @hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
    def _run(cfg: Any) -> None:
        log = setup_logging()
        paths = Paths().ensure()
        ds_name = cfg.data.source.hf_dataset
        ds_cfg = getattr(cfg.data.source, "hf_config", None)
        tf = cfg.data.processing.text_field
        split_map = cfg.data.source.split_map
        streaming = bool(getattr(cfg.data.source, "streaming", False))

        log.info(
            f"[dataset-stats] Loading {ds_name}:{ds_cfg} (streaming={streaming}) ..."
        )
        ds = load_dataset(ds_name, ds_cfg, streaming=streaming)

        train_split = ds[split_map.train]
        test_split = ds[split_map.test]

        s = _read_stats_cfg(cfg)

        max_train = getattr(getattr(cfg.data, "limits", {}), "max_train_samples", None)
        max_test = getattr(getattr(cfg.data, "limits", {}), "max_eval_samples", None)

        train_texts = _safe_list(train_split, tf, limit=max_train)
        test_texts = _safe_list(test_split, tf, limit=max_test)

        out: Dict[str, Any] = {
            "dataset": {
                "name": ds_name,
                "config": ds_cfg,
                "num_train_samples_used": len(train_texts),
                "num_test_samples_used": len(test_texts),
            }
        }

        if s.compute_unigram_entropy:
            tokens = _word_tokens(train_texts)
            H_uni = unigram_entropy_bits_per_symbol(tokens)
            avg_bytes = avg_token_length_bytes(tokens)
            H_byte = entropy_per_byte(H_uni, avg_bytes)
            out["unigram"] = {
                "entropy_bits_per_token": float(H_uni),
                "avg_token_len_bytes": float(avg_bytes),
                "entropy_bits_per_byte": float(H_byte),
            }

        if s.compute_ngram_entropy:
            out["ngram"] = {}
            for n in s.compute_ngram_entropy:
                try:
                    bpc_n = kenlm_ngram_bpc(
                        train_texts,
                        test_texts,
                        workdir=str(paths.artifacts / "runs" / f"kenlm_{n}gram_stats"),
                        order=int(n),
                    )
                    out["ngram"][f"order_{int(n)}"] = {"bpc": float(bpc_n)}
                except Exception as e:
                    out["ngram"][f"order_{int(n)}"] = {"error": str(e)}

        if s.compute_mi:
            tokens_test = _word_tokens(test_texts)
            out["mutual_information"] = {
                "adjacent_token_bits": float(mutual_information_adjacent(tokens_test))
            }

        H_inf_ref = getattr(
            getattr(cfg.data, "stats", {}), "h_infty_bits_per_char", None
        )
        if H_inf_ref is not None and "ngram" in out and out["ngram"]:
            try:
                bpcs = [
                    v["bpc"]
                    for v in out["ngram"].values()
                    if isinstance(v, dict) and "bpc" in v
                ]
                if bpcs:
                    out["ngram"]["gap_to_published_H_infty_bits_per_char"] = float(
                        min(bpcs) - float(H_inf_ref)
                    )
            except Exception:
                pass

        stats_dir = paths.results / "dataset_stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        json_path = stats_dir / f"{cfg.data.name}.json"
        json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

        tsv_path = stats_dir / f"{cfg.data.name}.tsv"
        rows: List[str] = []
        rows.append("key\tvalue")
        if "unigram" in out:
            rows.append(
                f"unigram.entropy_bits_per_token\t{out['unigram']['entropy_bits_per_token']}"
            )
            rows.append(
                f"unigram.avg_token_len_bytes\t{out['unigram']['avg_token_len_bytes']}"
            )
            rows.append(
                f"unigram.entropy_bits_per_byte\t{out['unigram']['entropy_bits_per_byte']}"
            )
        if "ngram" in out:
            for k, v in out["ngram"].items():
                if isinstance(v, dict) and "bpc" in v:
                    rows.append(f"ngram.{k}.bpc\t{v['bpc']}")
            gap = out["ngram"].get("gap_to_published_H_infty_bits_per_char")
            if gap is not None:
                rows.append(f"ngram.gap_to_published_Hinfty\t{gap}")
        if "mutual_information" in out:
            rows.append(
                f"mi.adjacent_token_bits\t{out['mutual_information']['adjacent_token_bits']}"
            )
        tsv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        log.info(f"[dataset-stats] Wrote {json_path} and {tsv_path}")

    _run()


if __name__ == "__main__":
    main()
