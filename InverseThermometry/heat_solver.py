"""
Differentiable Forward Solver for 2D Heat Equation
Implements finite volume method with explicit Euler time-stepping
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Literal

from utils import _boundary_mask


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
        sigma_left = sigma[:-1, :]  # σ_{i,j}
        sigma_right = sigma[1:, :]  # σ_{i+1,j}
        return 2 / (1 / sigma_left + 1 / sigma_right)

    elif axis == 1:  # Y-direction
        # Compute σ_{i,j+½} = 2/(1/σ_{i,j} + 1/σ_{i,j+1})
        sigma_bottom = sigma[:, :-1]  # σ_{i,j}
        sigma_top = sigma[:, 1:]  # σ_{i,j+1}
        return 2 / (1 / sigma_bottom + 1 / sigma_top)


def heat_step(u, sigma, f, h, tau):
    """
    Single explicit Euler step with interface-averaged conductivity (finite-volume form).
    Uses harmonic averages at cell faces and zero normal flux at domain boundaries.

    Args:
        u: temperature field [M, M]
        sigma: conductivity field [M, M]
        f: source term [M, M]
        h: grid spacing
        tau: time step

    Returns:
        updated temperature field [M, M]
    """
    # Conductivity at interfaces via harmonic average
    sigma_x = harmonic_average(sigma, axis=0)  # [M-1, M] between i and i+1 at fixed j
    sigma_y = harmonic_average(sigma, axis=1)  # [M, M-1] between j and j+1 at fixed i

    # Differences of u across interfaces (centered jumps)
    du_x = u[1:, :] - u[:-1, :]  # [M-1, M]
    du_y = u[:, 1:] - u[:, :-1]  # [M, M-1]

    # Physical fluxes at interfaces: define q = +σ ∂u/∂n so that div(q) = ∇·(σ∇u)
    flux_x = sigma_x * (du_x / h)  # [M-1, M]
    flux_y = sigma_y * (du_y / h)  # [M, M-1]

    # Neumann BC (zero normal flux): pad interface flux arrays with zeros at boundaries
    flux_x_pad = torch.zeros(u.shape[0] + 1, u.shape[1], dtype=u.dtype, device=u.device)
    flux_x_pad[1:-1, :] = flux_x
    div_x = (flux_x_pad[1:, :] - flux_x_pad[:-1, :]) / h  # [M, M]

    flux_y_pad = torch.zeros(u.shape[0], u.shape[1] + 1, dtype=u.dtype, device=u.device)
    flux_y_pad[:, 1:-1] = flux_y
    div_y = (flux_y_pad[:, 1:] - flux_y_pad[:, :-1]) / h  # [M, M]

    # Explicit Euler update
    return u + tau * (div_x + div_y + f)


def harmonic_mean(a, b, eps=1e-12):
    return 2.0 * a * b / (a + b + eps)


def fv_euler_step_neumann(
    u: torch.Tensor,
    sigma: torch.Tensor,
    f: torch.Tensor,
    h: float,
    tau: float,
):
    """
    One explicit Euler FV step for ∂_t u + div(σ∇u) = f with Neumann BCs.
    """
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
    s_ijphalf = harmonic_mean(sigma, sR, eps)
    s_ijmhalf = harmonic_mean(sigma, sL, eps)

    # ---------- Boundary corrections (replace neighbor diffs on the domain boundary) ----------
    # Start with interior diffs
    dR = u - uR  # (i,j) - (i,j+1)
    dL = u - uL  # (i,j) - (i,j-1)
    dU = u - uU  # (i,j) - (i+1,j)
    dD = u - uD  # (i,j) - (i-1,j)

    # Defaults = homogeneous Neumann (mirror): diffs at boundary -> 0
    dL[..., :, 0] = 0.0  # left edge
    dR[..., :, -1] = 0.0  # right edge
    dD[..., 0, :] = 0.0  # bottom edge
    dU[..., -1, :] = 0.0  # top edge

    # For boundary faces, use σ_face = cell value (harmonic with itself)
    s_iphalf_j[..., -1, :] = sigma[..., -1, :]
    s_imhalf_j[..., 0, :] = sigma[..., 0, :]
    s_ijphalf[..., :, -1] = sigma[..., :, -1]
    s_ijmhalf[..., :, 0] = sigma[..., :, 0]

    # FV sum
    S = s_iphalf_j * dU + s_imhalf_j * dD + s_ijphalf * dR + s_ijmhalf * dL

    u_next = u + tau * (f - S / (h * h))
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
    tau_max = h**2 / (8 * sigma_max)  # More conservative than h^2/(4*sigma)
    return tau_max


class HeatSolver(nn.Module):
    """
    PyTorch module for differentiable heat equation solver.
    """

    def __init__(self, M, source_func, device="cpu"):
        super().__init__()
        self.M = M
        self.device = device
        self.source_func = source_func
        self.tau = None
        self.mask = _boundary_mask(self.M, self.device)

    def forward(self, sigma, T, n_steps=None, max_sigma=None, print_info=False):
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
        x = (
            torch.linspace(0, 1, self.M + 1, device=self.device)[:-1] + self.h / 2
        )  # Cell centers
        y = torch.linspace(0, 1, self.M + 1, device=self.device)[:-1] + self.h / 2

        # Create coordinate grids
        X, Y = torch.meshgrid(x, y, indexing="ij")

        sigma = sigma.to(self.device)

        # Initialize temperature field
        u = torch.zeros(self.M, self.M, device=self.device, dtype=torch.float32)

        # Compute stable time step
        tau_max = compute_stable_timestep(sigma, self.h)

        if n_steps is None and max_sigma is None:
            self.tau = tau_max
            n_steps = int(T / self.tau) + 1
            if print_info:
                print(f"N time steps {n_steps}")
            self.tau = T / n_steps
        elif n_steps is not None:
            self.tau = T / n_steps
            if self.tau > tau_max:
                print(
                    f"Warning: Time step {self.tau:.6f} exceeds stability limit {tau_max:.6f}"
                )
        elif max_sigma is not None:
            self.tau = self.h**2 / (8 * max_sigma)
            n_steps = int(T / self.tau) + 1
            self.tau = T / n_steps

        if print_info:
            print(
                f"Grid: {self.M}x{self.M}, Time step: {self.tau:.6f}, Steps: {n_steps}"
            )

        # Store solution history
        u_history = torch.zeros((n_steps + 1, self.M, self.M), device=self.device)
        u_b_history = torch.zeros((n_steps + 1, 4 * (self.M - 1)), device=self.device)

        u_history[0] = u.clone()
        u_b_history[0] = u[self.mask].clone()

        # Time stepping
        for k in range(n_steps):
            t = k * self.tau

            # Compute source term at current time
            f = self.source_func(X, Y, t)

            # Single time step
            u = heat_step(u, sigma, f, self.h, self.tau)

            # Store solution
            u_history[k + 1] = u.clone()
            u_b_history[k + 1] = u[self.mask].clone()

        return u, u_b_history, u_history


def solve_heat_equation(sigma, source, M, T, device="cpu"):
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

    solver = HeatSolver(M, source, device)
    u_final, u_b_history, u_history = solver(sigma, T, print_info=True)
    return u_final, u_b_history, u_history
