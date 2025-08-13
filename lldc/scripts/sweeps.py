# lldc/scripts/sweeps.py
from __future__ import annotations
from typing import Any, List
from pathlib import Path
import importlib, json

from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.data.bootstrap import ensure_data
from lldc.analysis.channel import run_channel, ChannelConfig


def _run_module(mod: str, argv: List[str]) -> None:
    m = importlib.import_module(mod)
    if hasattr(m, "main"):
        m.main(*argv)


def _load_recons(jsonl_paths: List[Path]) -> list[dict]:
    out = []
    for p in jsonl_paths:
        with p.open("r", encoding="utf-8") as f:
            for ln in f:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    continue
    return out


def main():
    import hydra

    @hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
    def _run(cfg: Any) -> None:
        log = setup_logging()
        paths = Paths().ensure()

        ensure_data(paths.root)

        exp = cfg.experiment.name
        mg = getattr(cfg.experiment, "model_groups", {})

        if exp == "e1a_wiki103":
            for m in mg.get("mlm", []):
                log.info(f"[sweeps:e1a] Stage1 specialise {m}")
                _run_module("lldc.scripts.stage1_specialise", [f"model={m}"])

            for m in mg.get("mlm", []):
                log.info(f"[sweeps:e1a] Stage2 PM {m}")
                _run_module(
                    "lldc.scripts.stage2_compress_pm", [f"model={m}", "dump_recon=true"]
                )

            for m in mg.get("ar", []):
                log.info(f"[sweeps:e1a] Stage2 VQ {m}")
                _run_module(
                    "lldc.scripts.stage2_compress_vq", [f"model={m}", "dump_recon=true"]
                )

            _run_module("lldc.scripts.evaluate_all", [])

        elif exp == "e2a_pruning":
            levels = cfg.experiment.pruning.schedule.levels
            for m in mg.get("mlm", []) + mg.get("ar", []):
                arch = "mlm" if m in mg.get("mlm", []) else "ar"
                for lvl in levels:
                    log.info(f"[sweeps:e2a] Prune & recover {m} at level={lvl}")
                    _run_module(
                        "lldc.scripts.prune_and_recover",
                        [f"model={m}", f"prune_level={lvl}", "recover_epochs=auto"],
                    )
                    ckpt = f"artifacts/checkpoints/{m}_pruned_{lvl}"
                    if arch == "mlm":
                        _run_module(
                            "lldc.scripts.stage2_compress_pm",
                            [f"model={m}", f"model_ckpt={ckpt}", "dump_recon=true"],
                        )
                    else:
                        _run_module(
                            "lldc.scripts.stage2_compress_vq",
                            [f"model={m}", f"model_ckpt={ckpt}", "dump_recon=true"],
                        )
            _run_module("lldc.scripts.evaluate_all", [])

        elif exp == "e2b_channel":
            payload_root = paths.payloads
            pm_paths = list(payload_root.glob("pm_*/recons.jsonl"))
            vq_paths = list(payload_root.glob("vq_*/recons.jsonl"))
            payloads_exist = bool(pm_paths or vq_paths)

            if not payloads_exist:
                log.info(
                    "[sweeps:e2b] No recon dumps found — generating via Stage2 now."
                )
                for m in mg.get("mlm", []):
                    _run_module(
                        "lldc.scripts.stage2_compress_pm",
                        [f"model={m}", "dump_recon=true"],
                    )
                for m in mg.get("ar", []):
                    _run_module(
                        "lldc.scripts.stage2_compress_vq",
                        [f"model={m}", "dump_recon=true"],
                    )

            pm_paths = list(payload_root.glob("pm_*/recons.jsonl"))
            vq_paths = list(payload_root.glob("vq_*/recons.jsonl"))
            docs = _load_recons(pm_paths) + _load_recons(vq_paths)

            orig = [d.get("original", "") for d in docs]
            recon = [d.get("reconstruction", "") for d in docs]
            payload_bits = sum(int(d.get("payload_bits", 0)) for d in docs)
            base_chars = sum(
                max(1, int(d.get("orig_chars", len(d.get("original", "")))))
                for d in docs
            )

            static_report = Path("artifacts/results/static_size.json")
            static_bits = (
                int(json.loads(static_report.read_text()).get("static_bits_total", 0))
                if static_report.exists()
                else 0
            )

            out_path = run_channel(
                test_texts=orig,
                recon_texts=recon,
                payload_bits_total=payload_bits,
                static_bits=static_bits,
                base_chars=base_chars,
                cfg=ChannelConfig(out_dir=paths.results / "e2b"),
            )
            log.info(f"[sweeps:e2b] Channel results -> {out_path}")
            _run_module("lldc.scripts.evaluate_all", [])
        else:
            log.error(f"Unknown experiment={exp}")

    _run()


if __name__ == "__main__":
    main()
