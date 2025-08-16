# lldc/scripts/sweeps.py

from __future__ import annotations
import hydra
from typing import Any, List
from pathlib import Path
from importlib import import_module
import json
import shlex
import subprocess
import os
from typing import Sequence
import sys
from hydra.core.global_hydra import GlobalHydra
from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.data.bootstrap import ensure_data
from lldc.utils.seed import DEFAULT_SEEDS
from datasets import load_dataset
from lldc.scripts.channel_analysis import run_channel_analysis
def _extract_hydra_overrides_from_argv(argv: List[str]) -> List[str]:
    forwarded: List[str] = []
    sweeps_only = {"--with-dataset-stats", "--with-latency-flops"}
    subcommands = {"stage2", "evaluate", "eval", "train", "grid", "sweep"}
    hydra_runtime = {"-m", "--multirun", "--cfg", "--info", "-p", "--package"}
    for tok in argv[1:]:
        if tok in sweeps_only or tok in subcommands or tok in hydra_runtime:
            continue
        forwarded.append(tok)
    return forwarded
def _run_cmd(cmd: List[str]) -> None:
    print(f"[sweeps] $ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True)
def run_post_runners(cfg: Any) -> None:
    hydra_overrides = _extract_hydra_overrides_from_argv(sys.argv)
    argv_tokens = set(sys.argv[1:])
    with_dataset_stats = bool(getattr(cfg.experiment, "with_dataset_stats", False)) or (
        "--with-dataset-stats" in argv_tokens
    )
    with_latency_flops = bool(getattr(cfg.experiment, "with_latency_flops", False)) or (
        "--with-latency-flops" in argv_tokens
    )
    if with_dataset_stats:
        _run_cmd(
            [
                sys.executable,
                "-m",
                "lldc.scripts.compute_dataset_stats",
                *hydra_overrides,
            ]
        )
    if with_latency_flops:
        _run_cmd(
            [
                sys.executable,
                "-m",
                "lldc.scripts.measure_latency_flops",
                *hydra_overrides,
            ]
        )
def _run_module(modname: str, argv: Sequence[str] = ()) -> None:
    cmd = [sys.executable, "-m", str(modname)]
    cmd.extend(str(a) for a in argv)
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        cmd,
        check=True,
        cwd=str(repo_root),
        env=env,
    )
def _load_recons(jsonl_paths: List[Path]) -> list[dict]:
    out: list[dict] = []
    for p in jsonl_paths:
        with p.open("r", encoding="utf-8") as f:
            for i, ln in enumerate(f, 1):
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
    return out
def _best_ready_ckpt(paths: Paths, model_name: str, experiment_name: str) -> Path | None:
    exp_root = paths.checkpoints / experiment_name
    if not exp_root.is_dir():
        return None
    cands: list[tuple[float, float, Path]] = []
    for p in exp_root.glob(f"{model_name}_pruned_*"):
        if not p.is_dir():
            continue
        if not (p / "READY").exists():
            continue
        cfg_ok = (p / "config.json").exists() or (p / "adapter_config.json").exists()
        weights_ok = any(
            (p / fn).exists()
            for fn in ("model.safetensors", "pytorch_model.bin", "adapter_model.bin")
        )
        if not (cfg_ok and weights_ok):
            continue
        level = -1.0
        try:
            tail = p.name.split("_pruned_")[-1]
            level = float(tail.split("_seed")[0])
        except Exception:
            level = -1.0
        try:
            mtime = p.stat().st_mtime
        except Exception:
            mtime = 0.0
        cands.append((level, mtime, p))
    if not cands:
        return None
    cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return cands[0][2]
def main() -> None:
    @hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
    def _run(cfg: Any) -> None:
        log = setup_logging()
        paths = Paths().ensure()
        ensure_data(paths.root)
        hydra_overrides = _extract_hydra_overrides_from_argv(sys.argv)
        exp = cfg.experiment.name
        mg = getattr(cfg.experiment, "model_groups", {})
        num_seeds = 2
        seed_list = list(DEFAULT_SEEDS)[:num_seeds]
        if exp == "e1a_wiki103":
            for seed in seed_list:
                for m in mg.get("mlm", []):
                    log.info(f"[sweeps:e1a] Stage1 specialise {m} (seed={seed})")
                    _run_module(
                        "lldc.scripts.stage1_specialise",
                        [f"model={m}", f"+seed={seed}", *hydra_overrides],
                    )
            for seed in seed_list:
                for m in mg.get("mlm", []):
                    ckpt = _best_ready_ckpt(paths, m, experiment_name=exp)
                    if ckpt is not None:
                        log.info(f"[sweeps:e1a] Stage2 PM {m} (seed={seed}) using ckpt={ckpt}")
                        args = [
                            f"model={m}",
                            f"+seed={seed}",
                            f"+model_ckpt={ckpt}",
                            "+dump_recon=true",
                            *hydra_overrides,
                        ]
                    else:
                        log.warning(f"[sweeps:e1a] No local checkpoint found for {m}; using hub weights.")
                        args = [
                            f"model={m}",
                            f"+seed={seed}",
                            "+dump_recon=true",
                            *hydra_overrides,
                        ]
                    _run_module("lldc.scripts.stage2_compress_pm", args)
            for seed in seed_list:
                for m in mg.get("ar", []):
                    log.info(f"[sweeps:e1a] Stage2 VQ {m} (seed={seed})")
                    _run_module(
                        "lldc.scripts.stage2_compress_vq",
                        [
                            f"model={m}",
                            f"+seed={seed}",
                            "+dump_recon=true",
                            *hydra_overrides,
                        ],
                    )
            _run_module("lldc.scripts.compute_baselines", [*hydra_overrides])
            _run_module("lldc.scripts.evaluate_all", [*hydra_overrides])
            run_post_runners(cfg)
        elif exp == "e2a_pruning":
            levels = cfg.experiment.pruning.schedule.levels
            for seed in seed_list:
                for m in mg.get("mlm", []) + mg.get("ar", []):
                    arch = "mlm" if m in mg.get("mlm", []) else "ar"
                    for lvl in levels:
                        log.info(
                            f"[sweeps:e2a] Prune & recover {m} at level={lvl} (seed={seed})"
                        )
                        _run_module(
                            "lldc.scripts.prune_and_recover",
                            [
                                f"model={m}",
                                f"+prune_level={lvl}",
                                f"+seed={seed}",
                                *hydra_overrides,
                            ],
                        )
                        ckpt_path = _best_ready_ckpt(paths, m, experiment_name=exp)
                        if ckpt_path is None:
                            exp_name = cfg.experiment.name
                            ckpt_guess = paths.checkpoints / exp_name / f"{m}_pruned_{lvl}_seed{seed}"
                            ckpt_path = ckpt_guess if ckpt_guess.exists() else None
                        if arch == "mlm":
                            args = [
                                f"model={m}",
                                f"+seed={seed}",
                                "+dump_recon=true",
                            ]
                            if ckpt_path is not None:
                                log.info(f"[sweeps:e2a] Stage2 PM {m} level={lvl} using ckpt={ckpt_path}")
                                args.append(f"+model_ckpt={ckpt_path}")
                            else:
                                log.warning(f"[sweeps:e2a] Could not locate pruned ckpt for {m}@{lvl}; using hub.")
                            _run_module("lldc.scripts.stage2_compress_pm", [*args, *hydra_overrides])
                        else:
                            args = [
                                f"model={m}",
                                f"+seed={seed}",
                                "+dump_recon=true",
                            ]
                            if ckpt_path is not None:
                                log.info(f"[sweeps:e2a] Stage2 VQ {m} level={lvl} using ckpt={ckpt_path}")
                                args.append(f"+model_ckpt={ckpt_path}")
                            else:
                                log.warning(f"[sweeps:e2a] Could not locate pruned ckpt for {m}@{lvl}; using hub.")
                            _run_module("lldc.scripts.stage2_compress_vq", [*args, *hydra_overrides])
            _run_module("lldc.scripts.compute_baselines", [*hydra_overrides])
            _run_module("lldc.scripts.evaluate_all", [*hydra_overrides])
            run_post_runners(cfg)
        elif exp == "e2b_channel":
            payload_root = paths.payloads
            pm_paths = list(payload_root.glob("pm_*/recons.jsonl"))
            vq_paths = list(payload_root.glob("vq_*/recons.jsonl"))
            payloads_exist = bool(pm_paths or vq_paths)
            if not payloads_exist:
                log.info(
                    "[sweeps:e2b] No recon dumps found — generating via Stage2 now."
                )
                for seed in seed_list:
                    for m in mg.get("mlm", []):
                        _run_module(
                            "lldc.scripts.stage2_compress_pm",
                            [
                                f"model={m}",
                                "+dump_recon=true",
                                f"+seed={seed}",
                                *hydra_overrides,
                            ],
                        )
                    for m in mg.get("ar", []):
                        _run_module(
                            "lldc.scripts.stage2_compress_vq",
                            [
                                f"model={m}",
                                "+dump_recon=true",
                                f"+seed={seed}",
                                *hydra_overrides,
                            ],
                        )
            pm_paths = list(payload_root.glob("pm_*/recons.jsonl"))
            vq_paths = list(payload_root.glob("vq_*/recons.jsonl"))
            docs = _load_recons(pm_paths) + _load_recons(vq_paths)
            orig = [d.get("original", "") for d in docs]
            recon = [d.get("reconstruction", "") for d in docs]
            ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
            text_field = cfg.data.processing.text_field
            train_split = ds[cfg.data.source.split_map.train].select(range(2000))
            train_texts = [ex[text_field] for ex in train_split]
            stats = run_channel_analysis(
                cfg,
                recon_texts=recon,
                orig_texts=orig,
                train_texts=train_texts,
            )
            out_path = paths.rd_curves / "channel_stats.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            log.info(f"[sweeps:e2b] Channel stats -> {out_path}")
            _run_module("lldc.scripts.evaluate_all", [*hydra_overrides])
            run_post_runners(cfg)
        else:
            log.error(f"Unknown experiment={exp}")
            sys.exit(2)
    _run()
if __name__ == "__main__":
    main()
