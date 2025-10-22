"""
Utility functions used by the heat solver and the inverse workflow.
Includes reference solutions/sources, error metrics, plotting helpers,
conductivity field generators, and informative printing utilities.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _apply_neumann_bc(u):
    """
    Apply Neumann boundary conditions (zero gradient at boundaries).
    Build a ghost-cell padding with symmetric extension so that
    ∂u/∂n = 0 at the domain boundary.
    
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


def verification_solution(x, y, t):
    """
    Analytical solution for verification test case.
    u(x,y,t) = (1 - exp(-t)) * cos(πx) * cos(πy)
    
    Args:
        x, y: spatial coordinates
        t: time (can be float or tensor)
    
    Returns:
        analytical solution u(x,y,t)
    """
    # Allow scalar t and place it on the right device/dtype
    if isinstance(t, (int, float)):
        t = torch.tensor(t, dtype=x.dtype, device=x.device)
    
    return (1 - torch.exp(-t)) * torch.cos(np.pi * x) * torch.cos(np.pi * y)


def verification_source(x, y, t, sigma=1.0):
    """
    Compute source term f(x,y,t) for the verification test case.
    Derived from the PDE: f = ∂u/∂t - ∇·(σ∇u)
    
    For u(x,y,t) = (1 - exp(-t)) * cos(πx) * cos(πy) and σ = 1:
    f(x,y,t) = exp(-t) * cos(πx) * cos(πy) + 2π²(1 - exp(-t)) * cos(πx) * cos(πy)
    
    Args:
        x, y: spatial coordinates
        t: time (can be float or tensor)
        sigma: conductivity (assumed constant = 1 for verification)
    
    Returns:
        source term f(x,y,t)
    """
    # Allow scalar t
    if isinstance(t, (int, float)):
        t = torch.tensor(t, dtype=x.dtype, device=x.device)
    
    u = verification_solution(x, y, t)
    # ∂u/∂t = exp(-t) * cos(πx) * cos(πy)
    du_dt = torch.exp(-t) * torch.cos(np.pi * x) * torch.cos(np.pi * y)
    
    # ∇²u = -2π² * (1 - exp(-t)) * cos(πx) * cos(πy)
    laplacian_u = -2 * np.pi**2 * u
    
    # f = ∂u/∂t - σ∇²u
    f = du_dt - sigma * laplacian_u
    
    return f


def sinusoidal_source(x, y, t, spatial=True):
    """
    Sinusoidal-in-time source. If spatial=True, modulate by cos(πx)cos(πy)
    to avoid spatial uniformity (which would make the solution independent of σ
    under Neumann BCs). Returns an [M,M] tensor matching x/y.
    """
    if isinstance(t, (int, float)):
        t = torch.tensor(t, dtype=x.dtype, device=x.device)
    if spatial:
        return torch.sin(torch.pi * t) * torch.cos(torch.pi * x) * torch.cos(torch.pi * y)
        
    else:
        return torch.sin(torch.pi * t).expand_as(x)


def compute_l2_error(u_numerical, u_analytical):
    """
    Compute L2 error between numerical and analytical solutions.
    
    Args:
        u_numerical: numerical solution [M, M]
        u_analytical: analytical solution [M, M]
    
    Returns:
        L2 error
    """
    error = u_numerical - u_analytical  # per-cell residual
    l2_error = torch.sqrt(torch.mean(error**2))
    return l2_error


def compute_relative_error(u_numerical, u_analytical):
    """
    Compute relative L2 error.
    
    Args:
        u_numerical: numerical solution [M, M]
        u_analytical: analytical solution [M, M]
    
    Returns:
        relative L2 error
    """
    l2_error = compute_l2_error(u_numerical, u_analytical)
    l2_analytical = torch.sqrt(torch.mean(u_analytical**2))  # ||u||_2
    return l2_error / l2_analytical


def visualize_solution(u, x, y, title="Temperature Field", figsize=(10, 8)):
    """
    Visualize temperature field.
    
    Args:
        u: temperature field [M, M]
        x, y: coordinate grids
        title: plot title
        figsize: figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 2D contour plot at cell centers
    im1 = ax1.contourf(x, y, u, levels=20, cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'{title} - Contour')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='Temperature')
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(x, y, u, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('Temperature')
    ax2.set_title(f'{title} - 3D Surface')
    plt.colorbar(surf, ax=ax2, label='Temperature')
    plt.tight_layout()
    plt.savefig(f"./InverseThermometry/images/solution_{title}.png")
    plt.close()


def visualize_comparison(u_numerical, u_analytical, x, y, title="Solution Comparison"):
    """
    Compare numerical and analytical solutions.
    
    Args:
        u_numerical: numerical solution [M, M]
        u_analytical: analytical solution [M, M]
        x, y: coordinate grids
        title: plot title
    """
    error = u_numerical - u_analytical
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Numerical solution
    im1 = axes[0, 0].contourf(x.detach().numpy(), y.detach().numpy(), u_numerical.detach().numpy(), levels=20, cmap='viridis')
    axes[0, 0].set_title('Numerical Solution')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0], label='Temperature')
    
    # Analytical solution
    im2 = axes[0, 1].contourf(x.detach().numpy(), y.detach().numpy(), u_analytical.detach().numpy(), levels=20, cmap='viridis')
    axes[0, 1].set_title('Analytical Solution')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1], label='Temperature')
    
    # Error
    im3 = axes[1, 0].contourf(x.detach().numpy(), y.detach().numpy(), error.detach().numpy(), levels=20, cmap='RdBu_r')
    axes[1, 0].set_title('Error (Numerical - Analytical)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_aspect('equal')
    plt.colorbar(im3, ax=axes[1, 0], label='Error')
    
    # Cross-section at mid y (slice along fixed column index for 'ij' meshgrid)
    y_idx = y.shape[1] // 2
    axes[1, 1].plot(x[:, y_idx].detach().numpy(), u_numerical[:, y_idx].detach().numpy(), 'b-', label='Numerical', linewidth=2)
    axes[1, 1].plot(x[:, y_idx].detach().numpy(), u_analytical[:, y_idx].detach().numpy(), 'r--', label='Analytical', linewidth=2)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Temperature')
    axes[1, 1].set_title(f'Cross-section at y = {y[0, y_idx]:.2f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"./InverseThermometry/images/comparison_{title}.png")
    plt.close()


def plot_convergence_analysis(h_values, errors, title="Convergence Analysis"):
    """
    Plot convergence analysis.
    
    Args:
        h_values: list of grid spacings
        errors: list of corresponding errors
        title: plot title
    """
    plt.figure(figsize=(10, 6))
    
    plt.loglog(h_values, errors, 'bo-', linewidth=2, markersize=8, label='Numerical Error')
    
    # Reference lines for expected convergence slopes
    h_ref = np.array(h_values)
    plt.loglog(h_ref, h_ref, 'r--', alpha=0.7, label='O(h)')
    plt.loglog(h_ref, h_ref**2, 'g--', alpha=0.7, label='O(h²)')
    
    plt.xlabel('Grid Spacing h')
    plt.ylabel('L2 Error')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"./InverseThermometry/images/convergence_{title}.png")
    plt.close()


def create_conductivity_field(M, pattern='constant', device='cpu'):
    """
    Create different conductivity field patterns for testing.
    
    Args:
        M: grid size
        pattern: 'constant', 'linear', 'sigmoid'
        device: device to create tensor on
    
    Returns:
        conductivity field [M, M]
    """
    h = 1.0 / M  # cell size
    x = torch.linspace(0, 1, M+1, device=device)[:-1] + h/2
    y = torch.linspace(0, 1, M+1, device=device)[:-1] + h/2
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    if pattern == 'constant':
        sigma = torch.ones(M, M, device=device)
    elif pattern == 'linear':
        sigma = 1 + X + Y  # linear ramp across domain
    elif pattern == 'sigmoid':
        # smooth bump centered at the domain center
        sigma = 1 + torch.sigmoid((X - 0.5)**2 + (Y - 0.5)**2)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return sigma


def print_solver_info(M, T, tau, n_steps, sigma_max):
    """
    Print solver information.
    
    Args:
        M: grid size
        T: total time
        tau: time step
        n_steps: number of time steps
        sigma_max: maximum conductivity
    """
    print("=" * 50)
    print("HEAT SOLVER INFORMATION")
    print("=" * 50)
    print(f"Grid size: {M} x {M}")
    print(f"Grid spacing: h = {1.0/M:.4f}")
    print(f"Total time: T = {T}")
    print(f"Time step: τ = {tau:.6f}")
    print(f"Number of steps: {n_steps}")
    print(f"Maximum conductivity: σ_max = {sigma_max:.4f}")
    # Classical constant-coefficient stability bound; solver uses a conservative 1/8 factor
    print(f"Stability limit: τ_max = {1.0/(4*M**2*sigma_max):.6f}")
    print(f"CFL number: {tau * 4 * sigma_max * M**2:.4f}")
    print("=" * 50)
