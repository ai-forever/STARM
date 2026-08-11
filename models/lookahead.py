from typing import List, Dict, Any

import torch
from torch.optim.optimizer import Optimizer


class LookAheadWrapper:
    def __init__(self, optimizer: Optimizer, k: int = 6, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        self.slow_weights = self._collect_params(clone=True)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def _collect_params(self, clone: bool = False) -> List[torch.Tensor]:
        params = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None or p.requires_grad:
                    params.append(p.data.clone() if clone else p.data)
        return params

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.optimizer.step()
        self.step_count += 1

        if self.step_count % self.k == 0:
            self._lookahead_sync()

    def _lookahead_sync(self):
        fast_params = self._collect_params(clone=False)
        for slow, fast in zip(self.slow_weights, fast_params):
            slow.add_(fast - slow, alpha=self.alpha)
            fast.copy_(slow)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "optimizer_state": self.optimizer.state_dict(),
            "slow_weights": [w.clone() for w in self.slow_weights],
            "step_count": self.step_count,
            "k": self.k,
            "alpha": self.alpha
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.step_count = state_dict["step_count"]
        self.k = state_dict.get("k", self.k)
        self.alpha = state_dict.get("alpha", self.alpha)

        self.optimizer.load_state_dict(state_dict["optimizer_state"])

        saved_slow = state_dict["slow_weights"]
        current_fast = self._collect_params(clone=False)

        self.slow_weights = []
        for saved_w, current_w in zip(saved_slow, current_fast):
            self.slow_weights.append(saved_w.clone())
            current_w.copy_(saved_w)
