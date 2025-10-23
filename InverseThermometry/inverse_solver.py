"""
Inverse utilities:
 - Generate synthetic boundary temperature datasets for the forward heat solver
 - Optimize thermal conductivity from boundary measurements via gradient-based fitting
The forward model is differentiable; we keep the inverse code simple and explicit.
"""

import os
import numpy as np
import torch

from heat_solver import HeatSolver, heat_steps
from utils import SimpleSigma, boundary_mask


class InverseSolver:
    def __init__(
        self,
        sigma_module,
        u_b_gt,
        source_func,
        M,
        T=None,
        lr=1e-3,
        alpha=0.1,
        sigma_0=1.0,
        device="cpu",
    ):
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device)

        self.u_b_gt = u_b_gt.to(self.device)
        self.u_b_gt.requires_grad_(False)

        self.M = M
        self.T = T
        self.n_steps = u_b_gt.shape[0] - 1

        self.alpha = alpha
        self.sigma_0 = sigma_0
        self.ls = lr
        self.solver = HeatSolver(self.M, source_func, self.device)

        self.sigma_module = sigma_module.to(self.device)
        self.optimizer = torch.optim.Adam(self.sigma_module.parameters(), lr=lr)

        if isinstance(sigma_0, torch.Tensor):
            self.sigma_0 = sigma_0.to(self.device)
            self.sigma_0.requires_grad_(False)
        else:
            self.sigma_0 = sigma_0

    def solve(self, max_iters=10000, **kwargs):
        for _ in range(max_iters):
            sigma = self.sigma_module()
            _, u_b_history, _ = self.solver(sigma, self.T, self.n_steps, **kwargs)

            loss_data = (
                self.solver.h
                * self.solver.tau
                * (u_b_history - self.u_b_gt).square().sum()
            )
            loss_reg = self.solver.h**2 * (sigma - self.sigma_0).square().sum()
            loss = loss_data + self.alpha * loss_reg

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            yield loss_data.item(), loss_reg.item(), loss.item()

    def get_solution(self):
        return self.sigma_module()


def estimate_conductivity(
    sigma_field,
    u_b_gt,
    source_history,
    sigma_0: float,
    alpha: float,
    h: float,
    tau: float,
    lr: float = 1e-2,
    max_iters: int = 100,
):
    optimizer = torch.optim.Adam(sigma_field.parameters(), lr=lr)
    mask = boundary_mask(source_history.shape[1], source_history.device)

    for _ in range(max_iters):
        sigma = sigma_field()
        u_history = heat_steps(
            torch.zeros_like(sigma),
            sigma,
            source_history,
            h,
            tau,
        )
        u_b_history = u_history[:, mask]

        loss_data = h * tau * (u_b_history - u_b_gt).square().sum()
        loss_reg = h**2 * (sigma - sigma_0).square().sum()
        loss = loss_data + alpha * loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: also calc some metrics

        yield loss_data.item(), loss_reg.item(), loss.item()
