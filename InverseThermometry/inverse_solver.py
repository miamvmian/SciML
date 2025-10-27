"""
Inverse utilities:
 - Generate synthetic boundary temperature datasets for the forward heat solver
 - Optimize thermal conductivity from boundary measurements via gradient-based fitting
The forward model is differentiable; we keep the inverse code simple and explicit.
"""

import os
import numpy as np
import torch
import torch.nn as nn

# Import strategy
#  - When executed as a module (python -m InverseThermometry.inverse_solver),
#    relative imports work.
#  - When executed as a script (python InverseThermometry/inverse_solver.py),
#    fall back to adding the current dir to sys.path and do absolute imports.
try:
    from .heat_solver import solve_heat_equation, compute_stable_timestep
    from .utils import create_conductivity_field
except ImportError:  # no parent package when run as a script
    import sys
    sys.path.append(os.path.dirname(__file__))
    from heat_solver import solve_heat_equation, compute_stable_timestep
    from utils import create_conductivity_field



def generate_boundary_dataset(M, T, source_func, sigma,sigmax=None, device='cpu', save_path='InverseThermometry/data/boundary_dataset.npz'):
    """
    Generate boundary temperature dataset with 5% uniform multiplicative noise.
    - Conductivity: linear sigma(x,y)=1+x+y
    - Source: f(x,y,t)=sin(pi t)
    - Boundary points: all cell-center boundary nodes (m=4M-4)
    Returns coords [m,2], times [K], u_clean [K,m], d_noisy [K,m]
    """
 
    # Build cell-center coordinates (ij indexing)
    h = 1.0 / M
    x = torch.linspace(0, 1, M+1, device=device)[:-1] + h/2
    y = torch.linspace(0, 1, M+1, device=device)[:-1] + h/2
    X, Y = torch.meshgrid(x, y, indexing='ij')
    # Select all boundary cells (top/bottom/left/right)
    mask = _boundary_mask(M, device=device)
    # Record boundary coordinates in mask traversal order (row-major)
    coords = torch.stack([X[mask], Y[mask]], dim=-1)


    if sigmax is None:
        _, u_hist = solve_heat_equation(sigma, source_func, M, T, n_steps=None, device=device)
    else:
        sigmax = max(torch.max(sigma).item(), sigmax)
        tau_max = compute_stable_timestep(sigmax, h)
        n_steps = int(T / tau_max) + 1
        _, u_hist = solve_heat_equation(sigma, source_func, M, T, n_steps=n_steps, device=device)

    # Extract boundary temperatures across time → shape [K, m]
    U = u_hist[:, mask]
    K = U.shape[0]

    # Apply 5% uniform multiplicative noise, ρ ~ U[-1, 1]
    rho = 2 * torch.rand_like(U) - 1
    # Multiplicative noise matches project spec: d_sk = (1 + 0.05 ρ) u(x_s,y_s,t_k)
    D = (1 + 0.05 * rho) * U
    time_samples = torch.linspace(0, T, K, device=device)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Persist minimal pieces needed for the inverse step
    np.savez(save_path,
             coords=coords.detach().cpu().numpy(),
             time_samples=time_samples.detach().cpu().numpy(),
             u_clean=U.detach().cpu().numpy(),
             d_noisy=D.detach().cpu().numpy(),
             M=M, T=T)

    return coords, time_samples, U, D


def _boundary_mask(M, device):
    # Boundary indicator with True at the domain boundary
    # Order matters for downstream stacking/slicing consistency.
    mask = torch.zeros(M, M, dtype=torch.bool, device=device)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    return mask


def optimize_conductivity_from_dataset(
    dataset_path: str,
    source_func,
    num_iters: int = 200,
    lr: float = 1e-1,
    alpha: float = 1e-1,
    sigma0: float = 2.0,
    sigma_min: float = 0.1,
    sigma_max: float = 3.0,
    device: str = 'cpu',
):
    """
    Inverse solver: fit σ(x,y) to boundary measurements d_noisy.

    Objective (discrete):
      L(σ) = h τ Σ_{k=1..K} Σ_{s∈∂Ω} (u_σ(x_s,y_s,t_k) − d_sk)^2
             + α h^2 Σ_{cells} (σ − σ0)^2

    - h τ scales data term by space-time measure (cell area and dt).
    - α h^2 makes the Tikhonov term grid-independent (units-consistent).
    - σ is parameterized so it stays within [sigma_min, sigma_max].
    Returns: estimated σ tensor and the loss history list.
    """
    # Load dataset produced by generate_boundary_dataset
    data = np.load(dataset_path)
    M = int(data['M'].item()) if np.ndim(data['M']) == 0 else int(data['M'])
    T = float(data['T'].item()) if np.ndim(data['T']) == 0 else float(data['T'])
    time_samples = torch.tensor(data['time_samples'], device=device, dtype=torch.float32)
    D = torch.tensor(data['d_noisy'], device=device, dtype=torch.float32)  # [K, m]

    # Dataset has K time samples (including t=0). Use the same grid for the forward model
    K, m = D.shape
    n_steps = K - 1
    h = 1.0 / M
    tau = T / n_steps

    # Parameterization: σ(θ) = σ_min + (σ_max - σ_min) · sigmoid(θ) keeps σ within [σ_min, σ_max]
    def logit(p: float) -> float:
        return float(np.log(p) - np.log(1.0 - p))

    p0 = (sigma0 - sigma_min) / (sigma_max - sigma_min)
    # Clamp to avoid logit singularities/saturation at 0 or 1
    p0 = float(np.clip(p0, 1e-6, 1.0 - 1e-6))
    init_val = logit(p0)
    # Optimization variable θ (same shape as σ); broadcasting is explicit and clear
    sigma_param = torch.full((M, M), init_val, device=device, dtype=torch.float32, requires_grad=True)
    
    optimizer = torch.optim.Adam([sigma_param], lr=lr)

    mask = _boundary_mask(M, device)
    total_loss_history = []
    data_loss_history = []
    reg_loss_history = []
    it = -1
    try:
        for it in range(num_iters):
            optimizer.zero_grad()
            # Map unconstrained parameters to bounded conductivity
            sigma = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(sigma_param)

            # Forward solve on the same time grid as the dataset
            _, u_hist = solve_heat_equation(sigma, source_func, M, T, n_steps=n_steps, device=device)
            U_pred = u_hist[:, mask]  # [K, m]

            # Data misfit weighted by cell area (h) and timestep (τ)
            data_term = h * tau * torch.sum((U_pred - D) ** 2)
            # Tikhonov regularization that keeps σ near σ0 (dimensionally consistent via h^2)
            reg_term = alpha * (h ** 2) * torch.sum((sigma - sigma0) ** 2)
            loss = data_term + reg_term

            loss.backward()
            # Optional diagnostics: gradient norm should be > 0 and not explode
            # print(f"grad norm: {sigma_param.grad.norm().item():.6e}")
            optimizer.step()

            total_loss_history.append(float(loss.detach().cpu()))
            data_loss_history.append(float(data_term.detach().cpu()))
            reg_loss_history.append(float(reg_term.detach().cpu()))
            if (it + 1) % max(1, num_iters // 10) == 0:
                print(f"Iter {it+1}/{num_iters}  loss={loss.item():.6e}  data={data_term.item():.6e}  reg={reg_term.item():.6e}")
    except KeyboardInterrupt:
        print(f"\nOptimization interrupted at iter {it+1}. Returning current estimate.")

    sigma_est = (sigma_min + (sigma_max - sigma_min) * torch.sigmoid(sigma_param)).detach()
    return sigma_est, U_pred, total_loss_history, data_loss_history, reg_loss_history



def run_training(
    M: int = 10,
    T: float = 1.0,
    device: str = "cpu",
    dataset: str = "InverseThermometry/data/boundary_dataset.npz",
    iters: int = 100,
    lr: float = 1e-1,
    alpha: float = 1e-3,
    sigma0: float = 2.0,
    sigma_min: float = 0.1,
    sigma_max: float = 3.0,
    outdir: str = "InverseThermometry/results",
):
        
    def sinusoidal_source(x, y, t, spatial=True):
        """
        Sinusoidal-in-time source. If spatial=True, modulate by cos(πx)cos(πy)
        to avoid spatial uniformity (which would make the solution independent of σ
        under Neumann BCs). Returns an [M,M] tensor matching x/y.
        """
        if isinstance(t, (int, float)):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)
        if spatial:
            return torch.exp(4*t) * torch.cos(np.pi * x) * torch.cos(np.pi * y)
            
        else:
            return torch.sin(10*torch.pi * t).expand_as(x)

    # Ensure the dataset directory exists
    os.makedirs(os.path.dirname(dataset), exist_ok=True)
    # Generate boundary dataset

    # True conductivity used to create synthetic data
    sigma = create_conductivity_field(M, 'linear', device=device)
    generate_boundary_dataset(M, T, sigma=sigma, source_func=sinusoidal_source, sigmax=sigma_max, device=device, save_path=dataset)

    print("Starting optimization...")
    sigma_est, U_pred,total_loss_history, data_loss_history, reg_loss_history = optimize_conductivity_from_dataset(
        dataset_path=dataset,
        source_func=sinusoidal_source,
        num_iters=iters,
        lr=lr,
        alpha=alpha,
        sigma0=sigma0,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=device,
    )
    
    # Save artifacts for post-analysis/visualization
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "U_pred.npy"), U_pred.detach().numpy())
    np.save(os.path.join(outdir, "sigma_est.npy"), sigma_est.cpu().numpy())
    np.save(os.path.join(outdir, "total_loss_history.npy"), np.array(total_loss_history))
    np.save(os.path.join(outdir, "data_loss_history.npy"), np.array(data_loss_history))
    np.save(os.path.join(outdir, "reg_loss_history.npy"), np.array(reg_loss_history))
    print(f"Saved sigma_est and loss history to {outdir}")


def main():
    # Simple entrypoint with defaults; call run_training(...) directly to customize
    run_training(sigma_max=5.0, alpha=1e-3)


if __name__ == "__main__":
    main()
