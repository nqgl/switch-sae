import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from jaxtyping import Float
from torch import Tensor
from fwd_cache import FwdCache


@dataclass
class SwitcherConfig:
    d_in: int
    num_experts: int
    alpha: float
    capacity_factor: float
    k: int = 1
    dtype: str = "float32"


class Switcher(nn.Module):
    def __init__(self, cfg: SwitcherConfig, logit_generator: nn.Module = None):
        super().__init__()
        self.logit_generator = logit_generator or nn.Linear(cfg.d_in, cfg.num_experts)
        self.cfg = cfg
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Float[Tensor, "batch d_in"], cache: FwdCache):
        T = x.shape[0]
        N = self.cfg.num_experts
        expert_capacity = int(self.cfg.capacity_factor * T / N)
        gate_probs = self.softmax(self.logit_generator(x))  # b N

        probs, expert_indices = torch.topk(
            gate_probs, self.cfg.k, dim=-1
        )  # (b k), (b k@[N])
        experts_active = (
            torch.arange(N, device=x.device).unsqueeze(-1).unsqueeze(-1)
            == expert_indices
        ).any(
            -1
        )  # (N b)
        fi = experts_active.to(self.cfg.dtype).sum(-1) / T
        Pi = gate_probs.sum(0) / T
        loss_per_expert = self.cfg.alpha * N * (fi * Pi)
        assert loss_per_expert.shape == (N,)
        cache.loss = loss_per_expert.sum()
        cache.indices = expert_indices
        out_batch = torch.zeros(
            N,
            expert_capacity,
            self.cfg.k,
            x.shape[1],
        )

        experts_active_mask = torch.zeros(
            N, expert_capacity, self.cfg.k, dtype=torch.bool, device=x.device
        )
        torch.arange(N, device=x.device).unsqueeze(-1).unsqueeze(-1) == expert_indices
        experts_active_clipped = experts_active & (
            experts_active.int().cumsum(-1) <= expert_capacity
        )  # (N b)

        for i in range(N):
            expert_active = (expert_indices == i).any(-1)
            x[expert_active][:expert_capacity]
