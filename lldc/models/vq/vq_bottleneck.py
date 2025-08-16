# lldc/models/vq/vq_bottleneck.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class VQOutput:
    z_q: torch.Tensor
    indices: torch.LongTensor
    commit_loss: torch.Tensor
    codebook_loss: torch.Tensor

class VectorQuantiser(nn.Module):
    def __init__(self, codebook_size: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.beta = beta
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim) * 0.02)
    def forward(self, h: torch.Tensor) -> VQOutput:
        B, T, D = h.shape
        assert D == self.dim, f"dim mismatch {D}!={self.dim}"
        flat = h.reshape(-1, D)
        h2 = (flat**2).sum(-1, keepdim=True)
        c2 = (self.codebook**2).sum(-1)
        dist = h2 + c2 - 2.0 * flat @ self.codebook.t()
        indices = torch.argmin(dist, dim=-1)
        z = self.codebook[indices]
        z = z.view(B, T, D)
        commit_loss = F.mse_loss(h.detach(), z)
        codebook_loss = F.mse_loss(h, z.detach())
        z_q = h + (z - h).detach()
        return VQOutput(
            z_q=z_q,
            indices=indices.view(B, T),
            commit_loss=commit_loss,
            codebook_loss=codebook_loss,
        )

class VQBottleneckWrapper(nn.Module):
    def __init__(
        self, lm: nn.Module, layer_after: int, codebook_size: int, beta: float = 0.25
    ):
        super().__init__()
        self.lm = lm
        self.layer_after = layer_after
        try:
            D = lm.config.n_embd
        except Exception:
            D = lm.base_model.model.config.n_embd
        self.vq = VectorQuantiser(codebook_size=codebook_size, dim=D, beta=beta)
    def forward(
        self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None, labels: torch.LongTensor | None = None
    ):
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
            
        model = self.lm
        transformer = (
            model.transformer
            if hasattr(model, "transformer")
            else model.base_model.transformer
        )
        
        extended_attention_mask: torch.Tensor | None = None
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=next(model.parameters()).dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(extended_attention_mask.dtype).min
            
        wte, wpe, drop, h = (
            transformer.wte,
            transformer.wpe,
            transformer.drop,
            transformer.h,
        )
        device = input_ids.device
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=device).unsqueeze(0)
        x = wte(input_ids) + wpe(pos)
        x = drop(x)
        for i in range(self.layer_after):
            layer_outputs = h[i](x, attention_mask=extended_attention_mask)
            x = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        vqo = self.vq(x)
        xq = vqo.z_q
        for i in range(self.layer_after, len(h)):
            layer_outputs = h[i](xq, attention_mask=extended_attention_mask)
            xq = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        ln_f = transformer.ln_f
        logits = model.lm_head(ln_f(xq))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            loss = ce + self.vq.beta * vqo.commit_loss + vqo.codebook_loss
        return {
            "logits": logits,
            "loss": loss,
            "vq_commit": vqo.commit_loss,
            "vq_codebook": vqo.codebook_loss,
            "indices": vqo.indices,
        }
    @torch.no_grad()
    def decode_from_indices(self, indices: torch.LongTensor) -> torch.LongTensor:
        model = self.lm
        transformer = (
            model.transformer
            if hasattr(model, "transformer")
            else model.base_model.transformer
        )
        h = transformer.h
        xq = self.vq.codebook[indices.view(-1)].view(
            indices.size(0), indices.size(1), -1
        )
        for i in range(self.layer_after, len(h)):
            xq = h[i](xq)[0] if isinstance(h[i](xq), tuple) else h[i](xq)
        ln_f = transformer.ln_f
        logits = model.lm_head(ln_f(xq))
        pred = logits.argmax(dim=-1)
        return pred
