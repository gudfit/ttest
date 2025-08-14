from __future__ import annotations
from typing import List, Optional
import torch
from lldc.models.vq.vq_bottleneck import VQBottleneckWrapper

from lldc.decompression.vq_decompress import reconstruct_tokens_from_indices
