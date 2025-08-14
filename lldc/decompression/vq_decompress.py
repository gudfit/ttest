# lldc/decompression/vq_decompress.py

from __future__ import annotations
from typing import List, Optional
import math
import torch
from torch import nn
from lldc.models.vq.vq_bottleneck import VQBottleneckWrapper


@torch.no_grad()
def reconstruct_tokens_from_indices(
    model: VQBottleneckWrapper,
    indices: List[int],
    start_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    max_new_tokens: int = 0,
    margin_factor: float = 1.05,
) -> List[int]:
    device = next(model.parameters()).device
    lm = model.lm
    cfg = getattr(lm, "config", None)
    if eos_token_id is None and cfg is not None:
        eos_token_id = getattr(cfg, "eos_token_id", None)
    if start_token_id is None:
        start_token_id = eos_token_id if eos_token_id is not None else 0

    transformer = (
        lm.transformer if hasattr(lm, "transformer") else lm.base_model.transformer
    )
    h: nn.ModuleList = transformer.h
    ln_f = transformer.ln_f
    head = lm.lm_head

    codebook = model.vq.codebook
    D = codebook.size(1)
    L0 = len(indices)
    target_len = int(math.ceil(L0 * margin_factor))
    if max_new_tokens and target_len > L0 + max_new_tokens:
        target_len = L0 + max_new_tokens

    xq_list: List[torch.Tensor] = []
    out_tokens: List[int] = [int(start_token_id)]

    for t in range(target_len):
        idx_t = indices[t] if t < L0 else indices[-1]
        z_t = codebook[idx_t].view(1, 1, D).to(device)
        xq_list.append(z_t)
        xq = torch.cat(xq_list, dim=1)
        y = xq
        for i in range(model.layer_after, len(h)):
            y = h[i](y)[0] if isinstance(h[i](y), tuple) else h[i](y)
        logits = head(ln_f(y))[:, -1, :]
        next_id = int(torch.argmax(logits, dim=-1).item())
        out_tokens.append(next_id)
        if eos_token_id is not None and next_id == eos_token_id:
            break

    return out_tokens
