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
        n_steps=None,
        lr=1e-3,
        alpha=0.1,
        sigma_0=1,
        device='cpu'
    ):
        self.u_b_gt = u_b_gt
        self.u_b_gt.requires_grad_(False)

        self.M = M
        self.T = T
        self.n_steps = n_steps

        self.alpha = alpha
        self.sigma_0 = sigma_0
        self.ls = lr
        self.device = device

        self.solver = HeatSolver(self.M, source_func, self.device)

        self.sigma_module = sigma_module
        self.optimizer = torch.optim.Adam(sigma_module.parameters(), lr=lr)

        if isinstance(sigma_0, torch.Tensor):
            sigma_0.requires_grad_(False)
    
    def solve(self, max_iters=10000, tol=1e-3, **kwargs):
        boundary_loss_history = []
        regularization_loss_history = []
        total_loss_history = []
        for i in tqdm(range(max_iters)):
            sigma = self.sigma_module()
            _, u_b_history, _ = self.solver(sigma, self.T, self.n_steps, **kwargs)
            
            loss_data = self.solver.h * self.solver.tau * (u_b_history - self.u_b_gt).square().sum()
            loss_reg = self.solver.h**2 * (sigma - self.sigma_0).square().sum()
            loss = loss_data + self.alpha * loss_reg

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            boundary_loss_history.append(loss_data.item())
            regularization_loss_history.append(loss_reg.item())
            total_loss_history.append(loss.item())
        
            if total_loss_history[-1]/total_loss_history[0] < tol:
                print(f"Converged at iteration {i}, loss: {loss.item():.6f}")
                break
            
            print(f"Iter {i}: Loss = {loss.item():.6f}")

        sigma_est = self.sigma_module()

        return sigma_est, total_loss_history, boundary_loss_history, regularization_loss_history





        