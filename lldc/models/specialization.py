from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal
import torch
import torch.nn as nn

Saliency = Literal["magnitude"]


def _collect_ffn_channels(module: nn.Module) -> torch.Tensor:
    # works for BERT/RoBERTa style FFN: intermediate.dense -> output.dense
    inter = getattr(module, "intermediate", None)
    out = getattr(module, "output", None)
    if inter is None or out is None:
        return torch.empty(0)
    dense_i = getattr(inter, "dense", None)
    dense_o = getattr(out, "dense", None)
    if dense_i is None or dense_o is None:
        return torch.empty(0)
    # saliency per hidden channel (magnitude of outgoing weights)
    w = dense_i.weight.detach()  # [d_inter, d_model]
    sal = w.abs().mean(dim=1)  # [d_inter]
    return sal


def prune_ffn_block(module: nn.Module, keep_mask: torch.Tensor):
    inter = module.intermediate.dense
    out = module.output.dense
    # inter: [d_inter, d_model], out: [d_model, d_inter]
    with torch.no_grad():
        inter.weight = nn.Parameter(inter.weight[keep_mask, :].clone())
        inter.bias = nn.Parameter(inter.bias[keep_mask].clone())
        out.weight = nn.Parameter(out.weight[:, keep_mask].clone())
        # out.bias unchanged
    # update dims in config if present
    if hasattr(module, "config"):
        module.config.intermediate_size = int(keep_mask.sum().item())


def _num_heads_and_size(attn_module: nn.Module) -> Tuple[int, int]:
    cfg = getattr(
        attn_module,
        "config",
        getattr(attn_module, "self", getattr(attn_module, "num_heads", None)),
    )
    try:
        n_heads = attn_module.num_attention_heads
        head_dim = attn_module.attention_head_size
    except Exception:
        # GPT2
        n_heads = attn_module.num_heads
        head_dim = attn_module.head_dim
    return int(n_heads), int(head_dim)


def head_saliency(attn_module: nn.Module) -> torch.Tensor:
    # magnitude of q,k,v projections per head (sum of abs)
    wq = getattr(attn_module, "q_proj", None) or getattr(attn_module, "c_attn", None)
    if wq is None:
        # BERT-like: query,key,value separate
        q = attn_module.self.query.weight.data
        k = attn_module.self.key.weight.data
        v = attn_module.self.value.weight.data
        H = attn_module.num_attention_heads
        D = q.size(0) // H
        sal = []
        for h in range(H):
            s = (
                q[h * D : (h + 1) * D, :].abs().mean()
                + k[h * D : (h + 1) * D, :].abs().mean()
                + v[h * D : (h + 1) * D, :].abs().mean()
            )
            sal.append(s.item())
        return torch.tensor(sal)
    else:
        # GPT2 fused c_attn: split into q,k,v along last dim
        W = attn_module.c_attn.weight.data  # [d_model, 3*d_model]
        d = W.size(1) // 3
        q, k, v = W[:, :d], W[:, d : 2 * d], W[:, 2 * d :]
        H = attn_module.num_heads
        D = d // H
        sal = []
        for h in range(H):
            sl = (
                q[:, h * D : (h + 1) * D].abs().mean()
                + k[:, h * D : (h + 1) * D].abs().mean()
                + v[:, h * D : (h + 1) * D].abs().mean()
            )
            sal.append(float(sl))
        return torch.tensor(sal)


def structured_prune(
    model: nn.Module, level: float, drop_heads: bool = True, drop_ffn: bool = True
) -> Dict[str, int]:
    """
    level: fraction to drop from each structure type.
    Returns counts dropped.
    """
    dropped = {"heads": 0, "ffn_channels": 0}
    # Iterate encoder/decoder blocks depending on arch
    blocks = []
    if hasattr(model, "encoder"):  # BERT/RoBERTa
        blocks = list(model.encoder.layer)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):  # GPT2
        blocks = list(model.transformer.h)
    else:
        return dropped

    for blk in blocks:
        if drop_heads:
            # Derive saliency per head and drop lowest-magnitude fraction
            try:
                attn = blk.attention if hasattr(blk, "attention") else blk.attn
            except Exception:
                attn = getattr(blk, "attn", None)
            if attn is not None:
                sal = head_saliency(attn)
                H = sal.numel()
                k_drop = int(level * H)
                if k_drop > 0 and k_drop < H:
                    thr = torch.topk(sal, k_drop, largest=False).values.max()
                    # head_mask keeps 1 for kept heads; attach to module for use at forward
                    mask = (sal > thr).float().to(next(model.parameters()).device)
                    blk.register_buffer("head_mask", mask)
                    dropped["heads"] += int((sal <= thr).sum().item())
        if drop_ffn and hasattr(blk, "intermediate"):
            sal = _collect_ffn_channels(blk)
            if sal.numel() > 0:
                k_drop = int(level * sal.numel())
                if k_drop > 0 and k_drop < sal.numel():
                    thr = torch.topk(sal, k_drop, largest=False).values.max()
                    keep = sal > thr
                    prune_ffn_block(blk, keep)
                    dropped["ffn_channels"] += int(k_drop)
    return dropped
