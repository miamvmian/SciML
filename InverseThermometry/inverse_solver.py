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
        M,
        u_b,
        source_func,
        T=1,
        n_steps=None,
        alpha=0.1,
        sigma_0=1,
        device='cpu'
    ):
        self.M = M
        self.u_b = u_b
        self.u_b.requires_grad_(False)

        self.T = T
        self.n_steps = n_steps

        self.alpha = alpha
        self.sigma_0 = sigma_0

        self.device = device

        self.solver = HeatSolver(self.sigma_0, self.M, source_func, self.device)

        if isinstance(sigma_0, torch.Tensor):
            sigma_0.requires_grad_(False)

    
    def solve(self, lr=1e-3, max_iters=10000, tol=1e-3, **kwargs):
        
        loss_history = []

        # idx is for boundary points with ghost cells discarded
        idx = torch.tensor([0, -1], device=self.device)
        mask = torch.zeros((self.M, self.M), dtype=torch.bool, device=self.device)
        mask[idx, :] = True
        mask[:, idx] = True

        optimizer = torch.optim.Adam(self.solver.parameters(), lr=lr)

        for i in tqdm(range(max_iters)):
            _, u = self.solver(self.T, self.n_steps, **kwargs)

            loss_data = self.solver.h * self.solver.tau * (u[:, mask] - self.u_b).square().sum()
            loss_reg = self.solver.h**2 * (self.solver.sigma - self.sigma_0).square().sum()
            loss = loss_data + self.alpha * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if loss.item() < tol:
                print(f"Converged at iteration {i}, loss: {loss.item():.6f}")
                break
            
            print(f"Iter {i}: Loss = {loss.item():.6f}")
    
        sigma_est = self.solver.sigma.detach().cpu().numpy()

        return sigma_est, loss_history





        