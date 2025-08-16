# lldc/scripts/prune_and_recover.py

from __future__ import annotations
from typing import Any, Optional
from pathlib import Path
import math
import inspect
import os
import json
import hydra
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    GPT2LMHeadModel,
    BertForMaskedLM,
    RobertaForMaskedLM,
)
from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.utils.determinism import set_determinism
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

def _cfg_get_first(cfg: Any, candidates: list[str], default=None):
    for path in candidates:
        val = _cfg_get(cfg, path, None)
        if val is not None:
            return val
    return default

def _is_auto(x: Any) -> bool:
    return isinstance(x, str) and x.strip().lower() == "auto"

def _cfg_arch(name: str) -> str:
    n = name.lower()
    return "ar" if "gpt2" in n else "mlm"

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

def _make_training_args(
    output_dir: str,
    epochs: int,
    lr: float,
    scheduler: str = "cosine",
    warmup_ratio: float = 0.06,
) -> TrainingArguments:
    import inspect as _inspect
    sig = _inspect.signature(TrainingArguments.__init__)
    params = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 2,
        "learning_rate": float(lr),
        "num_train_epochs": int(epochs),
        "report_to": [],
        "warmup_ratio": float(warmup_ratio) if "warmup_ratio" in sig.parameters else None,
        "lr_scheduler_type": scheduler if "lr_scheduler_type" in sig.parameters else None,
    }
    params = {k: v for k, v in params.items() if v is not None}
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

def _resolve_recovery_hparams(
    cfg: Any,
    prune_level: float,
    arch: str,
    n_train_examples: int,
    per_device_batch_size: int = 2,
) -> dict:
    epochs_cfg = _cfg_get_first(
        cfg,
        [
            "recovery.finetune_epochs",
            "e2a.recovery.finetune_epochs",
            "experiment.recovery.finetune_epochs",
            "experiment.e2a.recovery.finetune_epochs",
        ],
        None,
    )
    lr_cfg = _cfg_get_first(
        cfg,
        [
            "recovery.lr",
            "e2a.recovery.lr",
            "experiment.recovery.lr",
            "experiment.e2a.recovery.lr",
        ],
        None,
    )
    scheduler_cfg = _cfg_get_first(
        cfg,
        [
            "recovery.scheduler",
            "e2a.recovery.scheduler",
            "experiment.recovery.scheduler",
            "experiment.e2a.recovery.scheduler",
        ],
        None,
    )
    warmup_cfg = _cfg_get_first(
        cfg,
        [
            "recovery.warmup_ratio",
            "e2a.recovery.warmup_ratio",
            "experiment.recovery.warmup_ratio",
            "experiment.e2a.recovery.warmup_ratio",
        ],
        None,
    )
    level = float(max(0.0, min(1.0, prune_level)))
    severity = level / max(1e-6, (1.0 - level))
    base_updates = 600 if arch == "mlm" else 900
    target_updates = int(round(base_updates * (1.0 + 1.5 * severity)))
    target_updates = max(200, min(target_updates, 4000))
    steps_per_epoch = max(1, math.ceil(n_train_examples / max(1, per_device_batch_size)))
    if epochs_cfg is None or _is_auto(epochs_cfg):
        epochs = max(1, min(12, math.ceil(target_updates / steps_per_epoch)))
    else:
        epochs = int(epochs_cfg)
    if lr_cfg is None or _is_auto(lr_cfg):
        base_lr = 5e-5 if arch == "mlm" else 2e-5
        lr = base_lr * (1.0 + 0.5 * min(severity, 2.0))
        lr = float(min(lr, 2e-4 if arch == "mlm" else 1e-4))
    else:
        lr = float(lr_cfg)
    scheduler = "cosine" if (scheduler_cfg is None or _is_auto(scheduler_cfg)) else str(scheduler_cfg)
    if warmup_cfg is None or _is_auto(warmup_cfg):
        warmup_ratio = float(min(0.20, 0.06 + 0.10 * level))
    else:
        warmup_ratio = float(warmup_cfg)
    return {
        "epochs": int(epochs),
        "lr": float(lr),
        "scheduler": scheduler,
        "warmup_ratio": float(warmup_ratio),
        "steps_per_epoch": int(steps_per_epoch),
        "target_updates": int(target_updates),
        "severity": float(severity),
    }

def _greatest_divisor_leq(n: int, cap: int) -> int:
    m = min(cap, n)
    for d in range(m, 0, -1):
        if n % d == 0:
            return d
    return 1

def _update_config_after_pruning(model: torch.nn.Module, model_name: str, arch: str, log):
    """
    Inspects the pruned model and updates its config object with the new layer dimensions.
    This is CRITICAL for correctly reloading the pruned model later.
    """
    if not hasattr(model, "config"):
        log.warning("[prune] Model has no 'config' attribute to update.")
        return

    try:
        if arch == "ar" and isinstance(model, (GPT2LMHeadModel)):
            first_block = model.transformer.h[0]
            
            new_n_head = first_block.attn.num_heads
            if model.config.n_head != new_n_head:
                log.info(f"[prune] Updating config: n_head {model.config.n_head} -> {new_n_head}")
                model.config.n_head = new_n_head

            if hasattr(first_block.mlp, 'c_fc'): 
                new_n_inner = first_block.mlp.c_fc.weight.shape[-1]
            else: 
                new_n_inner = first_block.mlp.fc_in.out_features
                
            if hasattr(model.config, "n_inner") and model.config.n_inner != new_n_inner:
                log.info(f"[prune] Updating config: n_inner {model.config.n_inner} -> {new_n_inner}")
                model.config.n_inner = new_n_inner
            elif not hasattr(model.config, "n_inner"):
                 log.info(f"[prune] Setting config: n_inner -> {new_n_inner}")
                 model.config.n_inner = new_n_inner

        elif arch == "mlm" and isinstance(model, (BertForMaskedLM, RobertaForMaskedLM)):
            first_layer = model.bert.encoder.layer[0]

            new_n_head = first_layer.attention.self.num_attention_heads
            if model.config.num_attention_heads != new_n_head:
                log.info(f"[prune] Updating config: num_attention_heads {model.config.num_attention_heads} -> {new_n_head}")
                model.config.num_attention_heads = new_n_head

            new_intermediate_size = first_layer.intermediate.dense.out_features
            if model.config.intermediate_size != new_intermediate_size:
                log.info(f"[prune] Updating config: intermediate_size {model.config.intermediate_size} -> {new_intermediate_size}")
                model.config.intermediate_size = new_intermediate_size
        else:
            log.warning(f"[prune] Config update logic not implemented for model type: {type(model).__name__}")

    except Exception as e:
        log.error(f"[prune] Failed to update model config after pruning. This will likely cause loading errors. Error: {e}", exc_info=True)


@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg: Any) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    log = setup_logging()
    paths = Paths().ensure()
    torch.use_deterministic_algorithms(True, warn_only=True)
    set_determinism(getattr(cfg, "seed", 13))
    model_name = cfg.model.pretrained_name
    arch = _cfg_arch(model_name)
    level = float(getattr(cfg, "prune_level", 0.0))
    seed = getattr(cfg, "seed", None)
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    exp_name = str(getattr(getattr(cfg, "experiment", None), "name", "default"))
    outdir = paths.checkpoints / exp_name / f"{model_name}_pruned_{level}{seed_suffix}"
    skip_if_ready = bool(
        _cfg_get_first(
            cfg,
            [
                "recovery.skip_if_ready",
                "e2a.recovery.skip_if_ready",
                "experiment.recovery.skip_if_ready",
                "experiment.e2a.recovery.skip_if_ready",
            ],
            True,
        )
    )
    force_retrain = bool(
        _cfg_get_first(
            cfg,
            [
                "recovery.force",
                "recovery.force_retrain",
                "e2a.recovery.force",
                "experiment.recovery.force",
                "experiment.e2a.recovery.force",
            ],
            False,
        )
    )
    if skip_if_ready and not force_retrain and _is_valid_hf_dir(outdir):
        log.info(f"[prune] Checkpoint already READY — skipping retrain: {outdir}")
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
            _cfg_get(cfg, "experiment.pruning.structured.drop_attention_heads", True),
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
    
    _update_config_after_pruning(model, model_name, arch, log)

    ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
    text_field = cfg.data.processing.text_field
    def tok_fn(b):
        return tok(
            b[text_field],
            truncation=True,
            max_length=cfg.data.processing.max_length,
        )
    ds = ds.map(tok_fn, batched=True, remove_columns=[text_field])
    ds = ds.filter(lambda ex: len(ex["input_ids"]) > 0)
    train_split = ds["train"]
    if len(train_split) == 0:
        log.error("[prune] No valid training examples left after tokenization and filtering. Cannot recover.")
        raise RuntimeError("Empty training dataset for recovery.")
    train_slice_size = min(4000, len(train_split))
    train_dataset = train_split.select(range(train_slice_size))
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=(arch == "mlm"))
    resolved = _resolve_recovery_hparams(
        cfg=cfg,
        prune_level=level,
        arch=arch,
        n_train_examples=train_slice_size,
        per_device_batch_size=2,
    )
    log.info(
        "[recover/auto] severity=%.3f, target_updates=%d, steps/epoch=%d → epochs=%d, lr=%.2e, warmup_ratio=%.3f, scheduler=%s",
        resolved["severity"],
        resolved["target_updates"],
        resolved["steps_per_epoch"],
        resolved["epochs"],
        resolved["lr"],
        resolved["warmup_ratio"],
        resolved["scheduler"],
    )
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "recovery_hparams.json").write_text(json.dumps(resolved, indent=2))
    args = _make_training_args(
        output_dir=str(outdir),
        epochs=resolved["epochs"],
        lr=resolved["lr"],
        scheduler=resolved["scheduler"],
        warmup_ratio=resolved["warmup_ratio"],
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    (outdir / "READY").write_text("ok\n")
    log.info(f"[prune] Saved pruned+recovered checkpoint → {outdir}")

if __name__ == "__main__":
    main()
