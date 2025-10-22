"""
Solve the inverse problem
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import nrmse_score, r2_score
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
        device='cpu',
        max_grad_norm=None
    ):
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device)

        self.u_b_gt = u_b_gt.to(self.device)
        self.u_b_gt.requires_grad_(False)

        self.M = M
        self.T = T
        self.n_steps = n_steps

        self.alpha = alpha
        self.sigma_0 = sigma_0
        self.ls = lr
        self.solver = HeatSolver(self.M, source_func, self.device)

        self.sigma_module = sigma_module.to(self.device)
        self.optimizer = torch.optim.Adam(self.sigma_module.parameters(), lr=lr)
        self.max_grad_norm = max_grad_norm
        self.last_run_logs = {}
        self.sigma_est = None

        if isinstance(sigma_0, torch.Tensor):
            self.sigma_0 = sigma_0.to(self.device)
            self.sigma_0.requires_grad_(False)
        else:
            self.sigma_0 = sigma_0
    
    def solve(
            self,
            max_iters=10000,
            tol=1e-3,
            early_stopping=False,
            patience=10,
            early_stopping_delta=None,
            **kwargs
        ):
        boundary_loss_history = []
        regularization_loss_history = []
        total_loss_history = []
        gradient_norm_history = []
        n_steps_no_improve = 0
        for i in tqdm(range(max_iters)):
            sigma = self.sigma_module()
            _, u_b_history, _ = self.solver(sigma, self.T, self.n_steps, **kwargs)
            
            loss_data = self.solver.h * self.solver.tau * (u_b_history - self.u_b_gt).square().sum()
            loss_reg = self.solver.h**2 * (sigma - self.sigma_0).square().sum()
            loss = loss_data + self.alpha * loss_reg

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                parameters = [param for param in self.sigma_module.parameters() if param.grad is not None]
                if parameters:
                    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm).item()
                else:
                    grad_norm = 0.0
                gradient_norm_history.append(grad_norm)

            self.optimizer.step()

            boundary_loss_history.append(loss_data.item())
            regularization_loss_history.append(loss_reg.item())
            total_loss_history.append(loss.item())
            if len(total_loss_history) > 1:
                improvement = total_loss_history[-2] - total_loss_history[-1]
            else:
                improvement = float('nan')
            
            if early_stopping_delta is not None:
                if improvement < early_stopping_delta:
                    n_steps_no_improve += 1
            elif improvement < 0:
                n_steps_no_improve += 1
        
            nrmse = nrmse_score(u_b_history, self.u_b_gt)
            r2 = r2_score(u_b_history, self.u_b_gt)
            if nrmse < tol:
                print(f"Converged at iteration {i}, loss: {loss.item():.6f}; nrmse={nrmse:.6f}; R2: {r2:.5f}")
                break

            if early_stopping:
                if n_steps_no_improve > patience:
                    print(
                        f"Early stopping at iteration {i}"
                    )
                    print(f"Loss = {loss.item():.6f}; nrmse={nrmse:.6f}; R2: {r2:.5f}")
                    break
        
            print(f"Iter {i}: Loss = {loss.item():.6f}; nrmse={nrmse:.6f}; R2: {r2:.5f}")

        sigma_est = self.sigma_module()
        self.sigma_est = sigma_est

        logs = {
            'total_loss_history': total_loss_history,
            'boundary_loss_history': boundary_loss_history,
            'regularization_loss_history': regularization_loss_history,
            'gradient_norm_history': gradient_norm_history,
        }
        self.last_run_logs = logs

        return sigma_est, logs





        
