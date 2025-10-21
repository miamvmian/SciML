"""
Differentiable Forward Solver for 2D Heat Equation
Implements finite volume method with explicit Euler time-stepping
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Literal


def harmonic_average(sigma, axis):
    """
    Compute harmonic average at cell interfaces.
    
    Args:
        sigma: conductivity field [M, M]
        axis: 0 for x-direction, 1 for y-direction
    
    Returns:
        harmonic average at interfaces [M-1, M] or [M, M-1]
    """
    if axis == 0:  # X-direction
        # Compute σ_{i+½,j} = 2/(1/σ_{i,j} + 1/σ_{i+1,j})
        sigma_left = sigma[:-1, :]   # σ_{i,j}
        sigma_right = sigma[1:, :]   # σ_{i+1,j}
        return 2 / (1/sigma_left + 1/sigma_right)
    
    elif axis == 1:  # Y-direction
        # Compute σ_{i,j+½} = 2/(1/σ_{i,j} + 1/σ_{i,j+1})
        sigma_bottom = sigma[:, :-1]  # σ_{i,j}
        sigma_top = sigma[:, 1:]      # σ_{i,j+1}
        return 2 / (1/sigma_bottom + 1/sigma_top)


def _apply_neumann_bc(u):
    """
    Apply Neumann boundary conditions (zero gradient at boundaries).
    Uses ghost cells approach with symmetric extension.
    
    Args:
        u: temperature field [M, M]
    
    Returns:
        u with boundary conditions applied [M+2, M+2]
    """
    M = u.shape[0]
    u_bc = torch.zeros(M + 2, M + 2, dtype=u.dtype, device=u.device)
    
    # Interior points
    u_bc[1:-1, 1:-1] = u
    
    # Neumann BC: ∂u/∂n = 0
    # Left boundary: u[0, :] = u[1, :]
    u_bc[0, 1:-1] = u[0, :]
    # Right boundary: u[-1, :] = u[-2, :]
    u_bc[-1, 1:-1] = u[-1, :]
    # Bottom boundary: u[:, 0] = u[:, 1]
    u_bc[1:-1, 0] = u[:, 0]
    # Top boundary: u[:, -1] = u[:, -2]
    u_bc[1:-1, -1] = u[:, -1]
    
    # Corner points (average of adjacent boundaries)
    u_bc[0, 0] = (u_bc[0, 1] + u_bc[1, 0]) / 2
    u_bc[0, -1] = (u_bc[0, -2] + u_bc[1, -1]) / 2
    u_bc[-1, 0] = (u_bc[-1, 1] + u_bc[-2, 0]) / 2
    u_bc[-1, -1] = (u_bc[-1, -2] + u_bc[-2, -1]) / 2
    
    return u_bc


def heat_step(u, sigma, f, h, tau):
    """
    Legacy function: Single time step of the heat equation using finite difference method.

    Args:
        u: temperature field [M, M]
        sigma: conductivity field [M, M]
        f: source term [M, M]
        h: grid spacing
        tau: time step
    
    Returns:
        updated temperature field [M, M]
    """
    M = u.shape[0]
    
    # Apply boundary conditions with ghost cells
    u_bc = _apply_neumann_bc(u)  # [M+2, M+2]
    
    # Initialize updated temperature
    u_new = u.clone()
    
    # Use finite difference method: ∂u/∂t = ∇·(σ∇u) + f
    # ∇·(σ∇u) = ∂/∂x(σ ∂u/∂x) + ∂/∂y(σ ∂u/∂y)
    
    for i in range(M):
        for j in range(M):
            # X-direction: ∂/∂x(σ ∂u/∂x)
            # Central difference for ∂u/∂x at cell centers
            if i == 0:  # Left boundary
                du_dx_right = (u_bc[i+2, j+1] - u_bc[i+1, j+1]) / h
                du_dx_left = 0  # Neumann BC: ∂u/∂x = 0
            elif i == M-1:  # Right boundary
                du_dx_right = 0  # Neumann BC: ∂u/∂x = 0
                du_dx_left = (u_bc[i+1, j+1] - u_bc[i, j+1]) / h
            else:  # Interior
                du_dx_right = (u_bc[i+2, j+1] - u_bc[i+1, j+1]) / h
                du_dx_left = (u_bc[i+1, j+1] - u_bc[i, j+1]) / h
            
            # Y-direction: ∂/∂y(σ ∂u/∂y)
            if j == 0:  # Bottom boundary
                du_dy_top = (u_bc[i+1, j+2] - u_bc[i+1, j+1]) / h
                du_dy_bottom = 0  # Neumann BC: ∂u/∂y = 0
            elif j == M-1:  # Top boundary
                du_dy_top = 0  # Neumann BC: ∂u/∂y = 0
                du_dy_bottom = (u_bc[i+1, j+1] - u_bc[i+1, j]) / h
            else:  # Interior
                du_dy_top = (u_bc[i+1, j+2] - u_bc[i+1, j+1]) / h
                du_dy_bottom = (u_bc[i+1, j+1] - u_bc[i+1, j]) / h
            
            # Compute divergence: ∇·(σ∇u) = ∂/∂x(σ ∂u/∂x) + ∂/∂y(σ ∂u/∂y)
            # harmoic_sigma_x = harmonic_average(sigma, axis=0)
            # harmoic_sigma_y = harmonic_average(sigma, axis=0)

            div_sigma_grad_u = (sigma[i, j] * du_dx_right - sigma[i, j] * du_dx_left) / h + \
                              (sigma[i, j] * du_dy_top - sigma[i, j] * du_dy_bottom) / h
            
            # Update temperature: ∂u/∂t = ∇·(σ∇u) + f
            u_new[i, j] = u[i, j] + tau * (div_sigma_grad_u + f[i, j])
    
    return u_new


def harmonic_mean(a, b, eps=1e-12):
    return 2.0 * a * b / (a + b + eps)


def fv_euler_step_neumann(
    u: torch.Tensor,          # [B,H,W]  u^k
    sigma: torch.Tensor,      # [B,H,W]
    f: torch.Tensor,          # [B,H,W]  source at t_k
    h: float,
    tau: float,
):
    """
    One explicit Euler FV step for ∂_t u + div(σ∇u) = f with Neumann BCs.
    """
    assert u.shape == sigma.shape == f.shape and u.ndim in [2,3], "Input tensors must have the same shape [B,H,W] or [H,W]"
    eps = 1e-12

    # neighbors (interior via rolls)
    uR = torch.roll(u, -1, dims=-1)  # right  (j+1)
    uL = torch.roll(u, +1, dims=-1)  # left   (j-1)
    uU = torch.roll(u, -1, dims=-2)  # up     (i+1)
    uD = torch.roll(u, +1, dims=-2)  # down   (i-1)

    sR = torch.roll(sigma, -1, dims=-1)
    sL = torch.roll(sigma, +1, dims=-1)
    sU = torch.roll(sigma, -1, dims=-2)
    sD = torch.roll(sigma, +1, dims=-2)

    # face sigmas (interior harmonic means)
    s_iphalf_j = harmonic_mean(sigma, sU, eps)
    s_imhalf_j = harmonic_mean(sigma, sD, eps)
    s_ijphalf  = harmonic_mean(sigma, sR, eps)
    s_ijmhalf  = harmonic_mean(sigma, sL, eps)

    # ---------- Boundary corrections (replace neighbor diffs on the domain boundary) ----------
    # Start with interior diffs
    dR = (u - uR)       # (i,j) - (i,j+1)
    dL = (u - uL)       # (i,j) - (i,j-1)
    dU = (u - uU)       # (i,j) - (i+1,j)
    dD = (u - uD)       # (i,j) - (i-1,j)

    # Defaults = homogeneous Neumann (mirror): diffs at boundary -> 0
    dL[..., :, 0]  = 0.0   # left edge
    dR[..., :, -1] = 0.0   # right edge
    dD[..., 0, :]  = 0.0   # bottom edge
    dU[..., -1, :] = 0.0   # top edge

    # For boundary faces, use σ_face = cell value (harmonic with itself)
    s_iphalf_j[..., -1, :] = sigma[..., -1, :]
    s_imhalf_j[...,  0, :] = sigma[...,  0, :]
    s_ijphalf[..., :, -1]  = sigma[..., :, -1]
    s_ijmhalf[..., :,  0]  = sigma[..., :,  0]

    # FV sum
    S = (
        s_iphalf_j * dU + s_imhalf_j * dD +
        s_ijphalf  * dR + s_ijmhalf  * dL
    )

    u_next = u + tau * (f - S / (h*h))
    return u_next


def compute_stable_timestep(sigma, h):
    """
    Compute stable time step for explicit Euler scheme.
    
    Args:
        sigma: conductivity field [M, M]
        h: grid spacing
    
    Returns:
        maximum stable time step
    """
    sigma_max = torch.max(sigma)
    # Use more conservative time step for stability
    tau_max = h**2 / (8 * sigma_max)  # More conservative than h^2/(4*sigma)
    return tau_max


class HeatSolver(nn.Module):
    """
    PyTorch module for differentiable heat equation solver.
    """

    def __init__(self, sigma_0, M, source_func, device='cpu'):
        super().__init__()
        self.M = M
        self.device = device

        if isinstance(sigma_0, (int, float)):
            sigma = torch.full((M, M), fill_value=sigma_0, dtype=torch.float32, device=device)
        elif isinstance(sigma_0, torch.Tensor):
            sigma = sigma_0.clone().to(device)
        self.sigma = nn.Parameter(sigma, requires_grad=True)

        self.source_func = source_func

        self.tau = None

        idx = torch.tensor([0, -1], device=self.device)
        self.mask = torch.zeros((self.M, self.M), dtype=torch.bool, device=self.device)
        self.mask[idx, :] = True
        self.mask[:, idx] = True

    def forward(self, T, n_steps=None, print_info=False):
        """
        Solve the 2D heat equation using finite volume method.
        
        Args:
            T: total time
            n_steps: number of time steps (auto-computed if None)
            device: device to run computation on
        
        Returns:
            u: final temperature field [M, M]
            u_b_history: temperature field at each time step [n_steps+1, 4*(M-1)]
        """
        # Grid setup
        self.h = 1.0 / self.M
        x = torch.linspace(0, 1, self.M+1, device=self.device)[:-1] + self.h/2  # Cell centers
        y = torch.linspace(0, 1, self.M+1, device=self.device)[:-1] + self.h/2
        
        # Create coordinate grids
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Initialize temperature field
        u = torch.zeros(self.M, self.M, device=self.device, dtype=torch.float32)
        
        # Compute stable time step
        tau_max = compute_stable_timestep(self.sigma, self.h)
        
        if n_steps is None:
            # Use 90% of maximum stable time step
            self.tau = 0.9 * tau_max
            n_steps = int(T / self.tau) + 1
            if print_info:
                print(f"N time steps {n_steps}")
            self.tau = T / n_steps
        else:
            self.tau = T / n_steps
            if self.tau > tau_max:
                print(f"Warning: Time step {self.tau:.6f} exceeds stability limit {tau_max:.6f}")
        
        if print_info:
            print(f"Grid: {self.M}x{self.M}, Time step: {self.tau:.6f}, Steps: {n_steps}")
        
        # Store solution history
        u_b_history = torch.zeros(n_steps + 1, 4*(self.M - 1), device=self.device)


        u_b_history[0] = u[self.mask].clone()
        
        # Time stepping
        for k in range(n_steps):
            t = k * self.tau
            
            # Compute source term at current time
            f = self.source_func(X, Y, t)
            
            # Single time step
            u = fv_euler_step_neumann(u, self.sigma, f, self.h, self.tau)
            
            # Store solution
            u_b_history[k + 1] = u[self.mask].clone()

        return u, u_b_history


def solve_heat_equation(sigma, verification_source, M, T, device='cpu'):
    """
    Solve the heat equation for verification test case.
    
    Args:
        sigma: conductivity field [M, M]
        verification_source: source term function
        M: grid size
        T: total time
        device: device to run computation on
    Returns:
        u_final: final temperature field [M, M]
        u_b_history: temperature field at each time step [n_steps+1, M, M]
    """

    solver = HeatSolver(sigma, M, verification_source, device)
    u_final, u_b_history = solver(T=T, print_info=True)
    return u_final, u_b_history
    