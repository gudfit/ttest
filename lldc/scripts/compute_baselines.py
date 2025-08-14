# lldc/scripts/compute_baselines.py

import hydra
from __future__ import annotations
from typing import Any, Dict, List
import json, os, subprocess, tempfile, time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from lldc.utils.paths import Paths
from lldc.utils.logging import setup_logging
from lldc.baselines.kenlm_subsample import kenlm_ngram_bpc


def _write_corpus(txts: List[str], path: Path) -> None:
    path.write_text("\n".join(txts), encoding="utf-8")


def _try_run(cmd: List[str]) -> tuple[bool, float, str]:
    try:
        t0 = time.perf_counter()
        out = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        dt = (time.perf_counter() - t0) * 1000.0
        return True, dt, out.stdout.decode("utf-8", errors="ignore")
    except Exception as e:
        return False, 0.0, str(e)


def main():
    @hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
    def _run(cfg: Any) -> None:
        log = setup_logging()
        paths = Paths().ensure()
        out_dir = paths.results
        out_dir.mkdir(parents=True, exist_ok=True)

        ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
        text_field = cfg.data.processing.text_field
        train_texts = [ex[text_field] for ex in ds[cfg.data.source.split_map.train]]
        test_texts = [ex[text_field] for ex in ds[cfg.data.source.split_map.test]]
        total_chars = sum(len(t) for t in test_texts)

        results: Dict[str, Dict] = {}

        try:
            bpc8 = kenlm_ngram_bpc(
                train_texts, test_texts, workdir="artifacts/runs/kenlm_8gram", order=8
            )
            results["kenlm_8gram"] = {"bpc": float(bpc8), "status": "ok"}
        except Exception as e:
            results["kenlm_8gram"] = {"status": f"error: {e}"}

        try:
            with tempfile.TemporaryDirectory() as td:
                raw = Path(td) / "test.txt"
                _write_corpus(test_texts, raw)
                comp = Path(td) / "test.zst"
                ok, _, _ = _try_run(
                    ["zstd", "-f", "-22", "-q", str(raw), "-o", str(comp)]
                )
                if not ok:
                    raise RuntimeError("zstd not available")
                out = Path(td) / "out.txt"
                ok2, dec_ms, _ = _try_run(
                    ["zstd", "-d", "-q", "-f", str(comp), "-o", str(out)]
                )
                if not ok2:
                    raise RuntimeError("zstd decode failed")
                comp_bits = comp.stat().st_size * 8
                results["zstd_22"] = {
                    "bpc": float(comp_bits / max(1, total_chars)),
                    "cpu_decode_ms": float(dec_ms),
                    "status": "ok",
                }
        except Exception as e:
            results["zstd_22"] = {"status": f"unavailable or error: {e}"}

        try:
            with tempfile.TemporaryDirectory() as td:
                raw = Path(td) / "test.txt"
                _write_corpus(test_texts, raw)
                comp = Path(td) / "test.cmix"
                ok, _, _ = _try_run(["cmix", str(raw), str(comp)])
                if not ok:
                    raise RuntimeError("cmix not available")
                out = Path(td) / "out.txt"
                ok2, dec_ms, _ = _try_run(["cmix", "-d", str(comp), str(out)])
                if not ok2:
                    raise RuntimeError("cmix decode failed")
                comp_bits = comp.stat().st_size * 8
                results["cmix"] = {
                    "bpc": float(comp_bits / max(1, total_chars)),
                    "cpu_decode_ms": float(dec_ms),
                    "status": "ok",
                }
        except Exception as e:
            results["cmix"] = {"status": f"unavailable or error: {e}"}

        try:
            import deepzip

            # NOTE: This is a placeholder; actual DeepZip usage depends on package API.
            # We record unavailability gracefully if import fails.
            results["deepzip"] = {"status": "ok_but_not_implemented_in_this_stub"}
        except Exception as e:
            results["deepzip"] = {"status": f"unavailable: {e}"}

        (out_dir / "baselines.json").write_text(json.dumps(results, indent=2))
        log.info(f"[baselines] Wrote results -> {out_dir / 'baselines.json'}")

    _run()


if __name__ == "__main__":
    main()
