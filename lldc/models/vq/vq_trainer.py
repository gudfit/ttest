# lldc/models/vq/vq_trainer.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import inspect
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from lldc.models.vq.vq_bottleneck import VQBottleneckWrapper
import json
from pathlib import Path


def _select_memory_appropriate_model(
    model_name: str, task_gb_requirement: float = 15.0
) -> str:
    if not torch.cuda.is_available():
        return model_name

    FALLBACK_CHAINS = {
        "gpt2": ["gpt2", "gpt2-medium", "gpt2-large"],
    }
    MODEL_MEMORY_REQ_GB = {
        "gpt2-large": 14.5,
        "gpt2-medium": 7.0,
        "gpt2": 3.5,
    }

    family = None
    for k in FALLBACK_CHAINS:
        if k in model_name:
            family = k
            break
    if not family:
        return model_name

    available_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    chain = FALLBACK_CHAINS[family]
    try:
        start_index = chain.index(model_name)
    except ValueError:
        return model_name

    for i in range(start_index, -1, -1):
        cand = chain[i]
        need = MODEL_MEMORY_REQ_GB.get(cand, float("inf"))
        if need <= available_gb:
            if cand != model_name:
                logging.getLogger("lldc").warning(
                    f"[VQ] '{model_name}' needs ~{MODEL_MEMORY_REQ_GB.get(model_name, 'N/A')}GB VRAM; "
                    f"available={available_gb:.1f}GB. Falling back to '{cand}'."
                )
            return cand

    smallest = chain[0]
    logging.getLogger("lldc").error(
        f"[VQ] Even smallest '{smallest}' needs ~{MODEL_MEMORY_REQ_GB.get(smallest, 'N/A')}GB; "
        f"available={available_gb:.1f}GB. Aborting."
    )
    raise MemoryError("Insufficient GPU memory for any model in the family.")


def save_vq_wrapper(model: VQBottleneckWrapper, out_dir: Path, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    torch.save(model.state_dict(), out_dir / "model.pt")


def load_vq_wrapper(
    base_model_name: str,
    layer_after: int,
    codebook_size: int,
    beta: float,
    ckpt_dir: Path,
) -> VQBottleneckWrapper | None:
    try:
        if not (ckpt_dir / "model.pt").exists():
            return None
        base = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = VQBottleneckWrapper(
            base, layer_after=layer_after, codebook_size=codebook_size, beta=beta
        )
        sd = torch.load(ckpt_dir / "model.pt", map_location="cpu")
        model.load_state_dict(sd, strict=True)
        model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        return model
    except Exception:
        return None


class IndexLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 512, layers: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.LongTensor):
        h = self.embed(x)
        y, _ = self.gru(h)
        logits = self.out(y)
        return logits


def _make_training_args(output_dir: str, lr: float, epochs: int) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    params = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "learning_rate": lr,
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


def train_vq_joint(
    base_model_name: str,
    dataset_name: str,
    dataset_config: str,
    text_field: str,
    max_length: int,
    layer_after: int,
    codebook_size: int,
    lr: float = 5e-5,
    epochs: int = 2,
    beta: float = 0.25,
) -> Tuple[VQBottleneckWrapper, AutoTokenizer]:
    torch.use_deterministic_algorithms(True, warn_only=False)
    safe_name = _select_memory_appropriate_model(base_model_name)

    tok = AutoTokenizer.from_pretrained(safe_name)
    if tok.pad_token is None and tok.eos_token:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(safe_name)
    model = VQBottleneckWrapper(
        base, layer_after=layer_after, codebook_size=codebook_size, beta=beta
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    ds = load_dataset(dataset_name, dataset_config)

    def tok_fn(b):
        return tok(b[text_field], truncation=True, max_length=max_length)

    ds = ds.map(tok_fn, batched=True, remove_columns=[text_field])
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    args = _make_training_args(
        output_dir=f"artifacts/checkpoints/vq_{safe_name}_K{codebook_size}",
        lr=lr,
        epochs=epochs,
    )

    def compute_loss(model, inputs, return_outputs=False):
        labels = inputs["input_ids"]
        outputs = model(input_ids=labels, labels=labels)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=ds["train"].select(range(min(10000, len(ds["train"])))),
        tokenizer=tok,
        compute_loss_func=compute_loss,
    )
    trainer.train()
    model.eval()
    return model, tok


@torch.no_grad()
def encode_indices(
    model: VQBottleneckWrapper, input_ids: torch.LongTensor
) -> torch.LongTensor:
    out = model(input_ids=input_ids)
    return out["indices"]


def _seq_ce_bits(lm: "IndexLM", seq: List[int]) -> float:
    if len(seq) < 2:
        return 0.0
    device = next(lm.parameters()).device
    x = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(seq[1:], dtype=torch.long, device=device).unsqueeze(0)
    logits = lm(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    return float((loss / torch.log(torch.tensor(2.0, device=device))).sum().item())


def train_index_lm(
    indices: List[List[int]],
    K: int,
    hidden: int = 512,
    layers: int = 1,
    epochs: int = 2,
    val_fraction: float = 0.1,
    early_stop: bool = True,
    patience: int = 3,
) -> IndexLM:
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm = IndexLM(vocab_size=K, hidden_size=hidden, layers=layers).to(device)
    opt = torch.optim.Adam(lm.parameters(), lr=1e-3)

    # Flatten into sequences
    data = [torch.tensor(seq, dtype=torch.long, device=device) for seq in indices if len(seq) >= 2]
    if not data:
        return lm.eval()

    split = max(1, int(len(data) * (1 - val_fraction)))
    train, val = data[:split], data[split:]

    best_val = float("inf")
    stale = 0

    for ep in range(int(epochs)):
        lm.train()
        tot = 0.0
        for seq in train:
            x = seq[:-1].unsqueeze(0)
            y = seq[1:].unsqueeze(0)
            logits = lm(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
        lm.eval()
        with torch.no_grad():
            vloss = 0.0
            for seq in val or []:
                x = seq[:-1].unsqueeze(0)
                y = seq[1:].unsqueeze(0)
                logits = lm(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                vloss += float(loss.item())
        if val:
            if vloss < best_val - 1e-6:
                best_val = vloss
                stale = 0
            else:
                stale += 1
                if early_stop and stale >= patience:
                    break

    return lm.eval()


def cross_entropy_bits_index_stream(lm: "IndexLM", seq: List[int]) -> float:
    if len(seq) < 2:
        return 0.0
    device = next(lm.parameters()).device
    x = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(seq[1:], dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = lm(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        bits = float(loss.item()) / float(torch.log(torch.tensor(2.0, device=device)))
    return bits

