from __future__ import annotations
from typing import List
import torch
from lldc.models.vq.vq_bottleneck import VQBottleneckWrapper


@torch.no_grad()
def reconstruct_tokens_from_indices(
    model: VQBottleneckWrapper,
    indices: List[int],
    input_length: int,
    eos_token_id: int | None,
    maxlen_margin: float = 1.05,
    device: str = "cuda",
) -> List[int]:
    """
    Feed quantised z (via wrapper) and decode greedily.
    Stop on EOS or when length exceeds ceil(input_length * maxlen_margin).
    """
    model.to(device)
    model.eval()
    max_new = int((input_length * maxlen_margin) - input_length + 0.5)
    max_out = input_length + max(max_new, 0)

    z = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
    out_ids = model.decode_from_indices(z, max_out=max_out, eos_token_id=eos_token_id)
    if eos_token_id is not None:
        # truncate after first EOS (inclusive)
        try:
            k = (out_ids[0] == eos_token_id).nonzero(as_tuple=True)[0][0].item()
            out_ids = out_ids[:, : k + 1]
        except Exception:
            pass
    return out_ids[0].tolist()
