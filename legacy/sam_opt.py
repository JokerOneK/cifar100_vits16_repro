# Minimal SAM implementation (Sharpness-Aware Minimization)
# Source adapted from https://github.com/davda54/sam with small tweaks.
import torch
from torch.optim import Optimizer

class SAM(Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, "Invalid rho, should be non-negative"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        super().__init__(self.param_groups, defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        rho = self.defaults["rho"]
        adaptive = self.defaults["adaptive"]
        grad_norm = self._grad_norm(adaptive)
        for group in self.param_groups:
            scale = rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: 
                    continue
                self.state[p]["e_w"] = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale
                p.add_(self.state[p]["e_w"])
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: 
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        raise RuntimeError("SAM doesn't support the .step() interface, use first_step and second_step.")

    def _grad_norm(self, adaptive):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(torch.stack([
            ((torch.abs(p) if adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]), p=2)
        return norm
