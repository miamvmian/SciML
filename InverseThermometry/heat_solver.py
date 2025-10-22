"""
Differentiable Forward Solver for 2D Heat Equation
Implements finite volume method with explicit Euler time-stepping
"""

import torch
import torch.nn as nn


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
    du_x = u[1:, :] - u[:-1, :]              # [M-1, M]
    du_y = u[:, 1:] - u[:, :-1]              # [M, M-1]

    # Physical fluxes at interfaces: define q = +σ ∂u/∂n so that div(q) = ∇·(σ∇u)
    flux_x = sigma_x * (du_x / h)            # [M-1, M]
    flux_y = sigma_y * (du_y / h)            # [M, M-1]

    # Neumann BC (zero normal flux): pad interface flux arrays with zeros at boundaries
    flux_x_pad = torch.zeros(u.shape[0] + 1, u.shape[1], dtype=u.dtype, device=u.device)
    flux_x_pad[1:-1, :] = flux_x
    div_x = (flux_x_pad[1:, :] - flux_x_pad[:-1, :]) / h  # [M, M]

    flux_y_pad = torch.zeros(u.shape[0], u.shape[1] + 1, dtype=u.dtype, device=u.device)
    flux_y_pad[:, 1:-1] = flux_y
    div_y = (flux_y_pad[:, 1:] - flux_y_pad[:, :-1]) / h  # [M, M]

    # Explicit Euler update
    return u + tau * (div_x + div_y + f)


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


def solve_heat_equation(sigma, source_func, M, T, n_steps=None, device='cpu'):
    """
    Solve the 2D heat equation using finite volume method.
    
    Args:
        sigma: conductivity field [M, M] with requires_grad=True
        source_func: function f(x, y, t) that returns source term
        M: number of grid cells per direction
        T: total time
        n_steps: number of time steps (auto-computed if None)
        device: device to run computation on
    
    Returns:
        u_final: final temperature field [M, M]
        u_history: temperature field at each time step [n_steps+1, M, M]
    """
    # Grid setup
    h = 1.0 / M
    x = torch.linspace(0, 1, M+1, device=device)[:-1] + h/2  # Cell centers
    y = torch.linspace(0, 1, M+1, device=device)[:-1] + h/2
    
    # Create coordinate grids
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Initialize temperature field
    u = torch.zeros(M, M, device=device, dtype=torch.float32)
    
    # Compute stable time step
    tau_max = compute_stable_timestep(sigma, h)
    
    if n_steps is None:
        # Use 90% of maximum stable time step
        tau = 0.9 * tau_max
        n_steps = int(T / tau) + 1
        tau = T / n_steps
    else:
        tau = T / n_steps
        if tau > tau_max:
            print(f"Warning: Time step {tau:.6f} exceeds stability limit {tau_max:.6f}")
    
    print(f"Grid: {M}x{M}, Time step: {tau:.6f}, Steps: {n_steps}")
    
    # Store solution history
    u_history = torch.zeros(n_steps + 1, M, M, device=device)
    u_history[0] = u.clone()
    
    # Time stepping
    for k in range(n_steps):
        t = k * tau
        
        # Compute source term at current time
        f = source_func(X, Y, t)
        
        # Single time step
        u = heat_step(u, sigma, f, h, tau)
        
        # Store solution
        u_history[k + 1] = u.clone()
    
    return u, u_history


class HeatSolver(nn.Module):
    """
    PyTorch module for differentiable heat equation solver.
    """
    
    def __init__(self, M, device='cpu'):
        super().__init__()
        self.M = M
        self.device = device
        self.h = 1.0 / M
        
        # Create coordinate grids
        x = torch.linspace(0, 1, M+1, device=device)[:-1] + self.h/2
        y = torch.linspace(0, 1, M+1, device=device)[:-1] + self.h/2
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        
    def forward(self, sigma, source_func, T, n_steps=None):
        """
        Forward pass of the heat solver.
        
        Args:
            sigma: conductivity field [M, M]
            source_func: function f(x, y, t) that returns source term
            T: total time
            n_steps: number of time steps
        
        Returns:
            u_final: final temperature field [M, M]
        """
        u_final, _ = solve_heat_equation(
            sigma, source_func, self.M, T, n_steps, self.device
        )
        return u_final
