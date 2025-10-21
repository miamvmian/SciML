import torch
from heat_solver import HeatSolver
from inverse_solver import InverseSolver
from utils import (create_source_function, create_conductivity_field)


def get_boundary_conditions(sigma, heat_source, T, device='cpu'):
    M = sigma.shape[0]
    solver = HeatSolver(sigma, M, heat_source, device)

    u_final, u_history = solver(T=1.0)

    idx = torch.tensor([0, -1])
    mask = torch.zeros((M, M), dtype=torch.bool)
    mask[idx, :] = True
    mask[:, idx] = True

    u_b = u_history[:, mask]
    return u_b

def main():
    # === Params ===
    M = 10
    T = 1.0
    device = 'cpu'
    n_steps = None  # Auto-compute
    alpha = 0.1
    sigma_0 = 1.0  # Initial guess
    lr = 1e-1
    noise_level = 0.01
    max_iters = 5000
    tol = 1e-4
    pattern = 'constant'
    source_func = create_source_function(pattern=pattern, device=device)
    sigma_gt = create_conductivity_field(2 * M, pattern=pattern, value=1.0, device=device)

    # Generate boundary observations with noise
    u_b_gt = get_boundary_conditions(sigma_gt, source_func, T, device=device)
    u_b = (1 + noise_level * torch.randn_like(u_b_gt)) * u_b_gt

    # Inverse solver
    inverse_solver = InverseSolver(
        M=M,
        u_b=u_b,
        source_func=source_func,
        T=T,
        n_steps=n_steps,
        alpha=alpha,
        sigma_0=sigma_0,
        device=device
    )

    estimated_sigma, loss_history = inverse_solver.solve(
        lr=lr,
        max_iters=max_iters,
        tol=tol,
        print_info=True
    )
    return estimated_sigma, loss_history

if __name__ == "__main__":
    main()
    