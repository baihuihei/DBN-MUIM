import torch
import torch.nn as nn
from typing import List, Tuple
from utils.utils import get_cosine

def cosine_simtier_list(n_dim, steps, scope_list, eps_list, dim_list,
                        simtier_eps_lists=None):
    return CosineSimTierList(n_dim, steps, scope_list, eps_list, dim_list,
                             simtier_eps_lists=simtier_eps_lists)

class CosineSimTierList(torch.nn.Module):
    def __init__(self, n_dim, steps, scope_list, eps_list, dim_list,
                 simtier_eps_lists=None):
        super().__init__()
        self.n_dim = n_dim
        self.steps = steps
        self.module_lists = []
        if simtier_eps_lists is None:
            simtier_eps_lists = [None] * len(scope_list)
        for scope, eps, dim, eps_sub in zip(scope_list, eps_list, dim_list, simtier_eps_lists):
            module = simtier_level(eps, dim, eps_list=eps_sub)
            self.module_lists.append(module)
            self.add_module(f"{scope}_simtier", module)

    def reset_parameters(self):
        for module in self.module_lists:
            module.reset_parameters()

    def forward(
        self, item, cosine_list, indicator_list
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        res_cosine = []
        res_simtier = []
        for seq, step, indicator, module in zip(
            cosine_list, self.steps, indicator_list, self.module_lists
        ):
            cosine_ = get_cosine(
                item, seq, steps=step, indicator=indicator, n_dim=self.n_dim
            )
            set_ = module(cosine_)
            res_cosine.append(cosine_)
            res_simtier.append(set_)
        return res_cosine, res_simtier
    
def simtier_level(eps=0.1, n_dim=4, eps_list=None):
    return MultiResolutionSimTierLevel(eps, n_dim, eps_list=eps_list)

class SimTierLevel(torch.nn.Module):
    """
    Single-resolution histogram similarity tier encoding (original method).
    """
    def __init__(self, eps=0.1, n_dim=4):
        super().__init__()
        self.eps = eps
        self.n_dim = n_dim
        self.bias = int(1 / eps)
        self.n_bins = 2 * self.bias + 2
        self.register_buffer(
            "equal_ranges",
            torch.reshape(
                torch.arange(0, self.n_bins, 1), [1, self.n_bins, 1]
            ),
            persistent=False,
        )
        self.register_buffer("bias_power", 1/torch.scalar_tensor(self.eps), persistent=False)
        self.emb = torch.nn.Parameter(torch.zeros(1, self.n_bins, self.n_dim))

    def reset_parameters(self):
        torch.nn.init.uniform_(self.emb.data)

    def forward(self, cosine: torch.Tensor):
        cosine_ids = torch.ceil(cosine * self.bias_power).to(torch.int32) + self.bias
        weight = (self.equal_ranges == torch.unsqueeze(cosine_ids, dim=1)).type(
            torch.float32
        )
        times = weight.sum(dim=2, keepdim=True)
        return torch.reshape(
            torch.log(times + 1) * self.emb, [-1, self.n_bins * self.n_dim]
        )


class MultiResolutionSimTierLevel(torch.nn.Module):
    """
    Multi-resolution histogram similarity tier encoding.

    Supports configurable resolution list via eps_list parameter.
    Default: 5 resolutions with eps=[0.1, 0.2, 0.3, 0.5, 0.6]
             bins=[22, 12, 8, 6, 4] = 52 total per scope.

    For ablation studies, pass custom eps_list, e.g.:
        eps_list=[0.1]           → 1 resolution (single), 22 bins
        eps_list=[0.1, 0.3]      → 2 resolutions, 30 bins
        eps_list=[0.1, 0.3, 0.5] → 3 resolutions, 36 bins
    """

    def __init__(self, eps=0.1, n_dim=4, eps_list=None):
        super().__init__()
        self.eps = eps
        self.n_dim = n_dim
        if eps_list is None:
            # Default 5-resolution configuration
            self.eps_list = [eps, eps * 2, eps * 3, eps * 4, eps * 6]
        else:
            self.eps_list = list(eps_list)  # allow custom list from config

        self.resolutions = nn.ModuleList([
            SimTierLevel(eps=e, n_dim=n_dim)
            for e in self.eps_list
        ])

    @property
    def total_bins(self):
        return sum(2 * int(1.0 / e) + 2 for e in self.eps_list)

    def reset_parameters(self):
        for m in self.resolutions:
            m.reset_parameters()

    def forward(self, cosine: torch.Tensor):
        outputs = [m(cosine) for m in self.resolutions]
        return torch.cat(outputs, dim=1)