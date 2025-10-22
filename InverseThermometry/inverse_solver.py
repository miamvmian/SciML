"""
Solve the inverse problem
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from heat_solver import HeatSolver


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
