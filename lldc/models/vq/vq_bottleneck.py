from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQOutput:
    z_q: torch.Tensor  # [B,T,D] quantised vectors
    indices: torch.LongTensor  # [B,T] codebook indices
    commit_loss: torch.Tensor  # scalar
    codebook_loss: torch.Tensor  # scalar


class VectorQuantiser(nn.Module):
    """
    Standard VQ layer with EMA codebook updates disabled (static codebook, ST estimator).
    """

    def __init__(self, codebook_size: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.beta = beta
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim) * 0.02)

    def forward(self, h: torch.Tensor) -> VQOutput:
        """
        h: [B,T,D]
        """
        B, T, D = h.shape
        assert D == self.dim, f"dim mismatch {D}!={self.dim}"
        flat = h.reshape(-1, D)  # [BT, D]
        # squared L2 to each codeword
        # ||h||^2 + ||c||^2 - 2 h·c
        h2 = (flat**2).sum(-1, keepdim=True)  # [BT,1]
        c2 = (self.codebook**2).sum(-1)  # [K]
        dist = h2 + c2 - 2.0 * flat @ self.codebook.t()  # [BT,K]
        indices = torch.argmin(dist, dim=-1)  # [BT]
        z = self.codebook[indices]  # [BT,D]
        z = z.view(B, T, D)

        # commitment + codebook losses (straight-through)
        commit_loss = F.mse_loss(h.detach(), z)
        codebook_loss = F.mse_loss(h, z.detach())
        z_q = h + (z - h).detach()  # ST estimator

        return VQOutput(
            z_q=z_q,
            indices=indices.view(B, T),
            commit_loss=commit_loss,
            codebook_loss=codebook_loss,
        )


class VQBottleneckWrapper(nn.Module):
    """
    Wrap a GPT-2 style causal LM; insert VQ after layer `layer_after`.
    Lower blocks act as 'encoder', upper blocks as 'decoder' that consume quantised states.
    """

    def __init__(
        self, lm: nn.Module, layer_after: int, codebook_size: int, beta: float = 0.25
    ):
        super().__init__()
        self.lm = lm
        self.layer_after = layer_after
        # Infer hidden size
        try:
            D = lm.config.n_embd
        except Exception:
            D = lm.base_model.model.config.n_embd  # fallback
        self.vq = VectorQuantiser(codebook_size=codebook_size, dim=D, beta=beta)

    def forward(
        self, input_ids: torch.LongTensor, labels: torch.LongTensor | None = None
    ):
        """
        Run transformer up to layer_after, quantise hidden states, feed quantised states to the rest.
        Returns: logits, total_loss (if labels provided), (commit, codebook losses)
        """
        model = self.lm
        transformer = (
            model.transformer
            if hasattr(model, "transformer")
            else model.base_model.transformer
        )
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

        # pass through lower blocks
        for i in range(self.layer_after):
            x = h[i](x)[0] if isinstance(h[i](x), tuple) else h[i](x)

        # quantise
        vqo = self.vq(x)

        # continue with quantised states
        xq = vqo.z_q
        for i in range(self.layer_after, len(h)):
            xq = h[i](xq)[0] if isinstance(h[i](xq), tuple) else h[i](xq)

        ln_f = transformer.ln_f
        logits = model.lm_head(ln_f(xq))

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            loss = (
                ce + vqo.commit_loss * self.vq.beta + vqo.codebook_loss * self.vq.beta
            )

        return {
            "logits": logits,
            "loss": loss,
            "vq_commit": vqo.commit_loss,
            "vq_codebook": vqo.codebook_loss,
            "indices": vqo.indices,
        }

    @torch.no_grad()
    def decode_from_indices(self, indices: torch.LongTensor) -> torch.LongTensor:
        """
        Reconstruct token ids given codebook indices [B,T].
        Uses quantised states as inputs to the upper stack and reads logits→argmax.
        Returns predicted token ids [B,T] (aligned to positions; first token may need BOS handling downstream).
        """
        model = self.lm
        transformer = (
            model.transformer
            if hasattr(model, "transformer")
            else model.base_model.transformer
        )
        h = transformer.h
        xq = self.vq.codebook[indices.view(-1)].view(
            indices.size(0), indices.size(1), -1
        )  # [B,T,D]
        for i in range(self.layer_after, len(h)):
            xq = h[i](xq)[0] if isinstance(h[i](xq), tuple) else h[i](xq)
        ln_f = transformer.ln_f
        logits = model.lm_head(ln_f(xq))  # [B,T,V]
        pred = logits.argmax(dim=-1)  # [B,T]
        return pred
