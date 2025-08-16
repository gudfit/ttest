# lldc/scripts/stage1_specialise.py

from __future__ import annotations
from typing import Any
from pathlib import Path
import hydra
import torch
import inspect
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from lldc.utils.logging import setup_logging
from lldc.utils.hydra_utils import resolve_auto
from lldc.utils.determinism import set_determinism
from lldc.utils.paths import Paths
from lldc.models.specialization import structured_prune


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


def _is_valid_hf_dir(p: Path) -> bool:
    try:
        if not p.exists() or not p.is_dir():
            return False
        has_ready = (p / "READY").exists()
        has_cfg = (p / "config.json").exists()
        has_weights = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
        return has_ready and has_cfg and has_weights
    except Exception:
        return False


def _cfg_arch(cfg: Any) -> str:
    a = cfg.model.arch
    if a == "auto":
        name = cfg.model.pretrained_name.lower()
        if "gpt2" in name:
            return "ar"
        else:
            return "mlm"
    return a


def _make_training_args(output_dir: str, epochs: int) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    params = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 2,
        "learning_rate": 5e-5,
        "num_train_epochs": int(epochs),
        "report_to": [],
    }
    if "bf16" in sig.parameters:
        params["bf16"] = torch.cuda.is_available()
    elif "fp16" in sig.parameters:
        params["fp16"] = torch.cuda.is_available()

    if "evaluation_strategy" in sig.parameters:
        params["evaluation_strategy"] = "no"
    elif "eval_strategy" in sig.parameters:
        params["eval_strategy"] = "no"

    if "save_strategy" in sig.parameters:
        params["save_strategy"] = "no"
    if "save_total_limit" in sig.parameters:
        params["save_total_limit"] = 1
    if "logging_steps" in sig.parameters:
        params["logging_steps"] = 50

    return TrainingArguments(**params)


@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg: Any) -> None:
    log = setup_logging()
    cfg = resolve_auto(cfg)
    paths = Paths().ensure()

    torch.use_deterministic_algorithms(True, warn_only=False)
    set_determinism(getattr(cfg, "seed", 13))

    model_name = cfg.model.pretrained_name
    arch = _cfg_arch(cfg)
    level = _cfg_get(cfg, "prune_level", 0.0)
    seed = getattr(cfg, "seed", None)
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    exp_name = str(getattr(getattr(cfg, "experiment", None), "name", "default"))
    outdir = paths.checkpoints / exp_name / f"{model_name}_pruned_{level}{seed_suffix}"
    skip_if_ready = bool(_cfg_get(cfg, "stage1.train.skip_if_ready", True))
    force = bool(_cfg_get(cfg, "stage1.train.force", False))
    if skip_if_ready and not force and _is_valid_hf_dir(outdir):
        log.info(f"[prune] READY checkpoint found — skipping Stage1 retrain: {outdir}")
        return

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None and tok.eos_token:
        tok.pad_token = tok.eos_token

    if arch == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    drop_heads = bool(
        _cfg_get(
            cfg,
            "pruning.structured.drop_attention_heads",
            _cfg_get(
                cfg, "experiment.pruning.structured.drop_attention_heads", True
            ),
        )
    )

    drop_ffn = bool(
        _cfg_get(
            cfg,
            "pruning.structured.drop_ffn_channels",
            _cfg_get(cfg, "experiment.pruning.structured.drop_ffn_channels", True),
        )
    )

    dropped = structured_prune(
        model, level=level, drop_heads=drop_heads, drop_ffn=drop_ffn
    )
    log.info(f"[prune] {model_name} @ level={level} → dropped={dropped}")

    ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
    text_field = cfg.data.processing.text_field

    def tok_fn(b):
        return tok(
            b[text_field],
            truncation=True,
            max_length=cfg.data.processing.max_length,
        )

    ds = ds.map(tok_fn, batched=True, remove_columns=[text_field])
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=(arch == "mlm"))

    recovery_epochs = _cfg_get(
        cfg,
        "stage1.train.epochs",
        _cfg_get(cfg, "experiment.stage1.train.epochs", 'auto'),
    )
    if recovery_epochs == 'auto':
        recovery_epochs = 3

    args = _make_training_args(
        output_dir=str(outdir),
        epochs=int(recovery_epochs),
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=ds["train"].select(range(min(10000, len(ds["train"])))),
    )
    trainer.train()

    outdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    (outdir / "READY").write_text("ok\n")
    log.info(f"[prune] Saved pruned+recovered checkpoint → {outdir}")

if __name__ == "__main__":
    main()

