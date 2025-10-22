import torch
from heat_solver import HeatSolver
from inverse_solver import InverseSolver
from utils import (create_source_function, create_conductivity_field)


def get_boundary_conditions(sigma, heat_source, T, max_sigma, device='cpu'):
    M = sigma.shape[0]
    solver = HeatSolver(M, heat_source, device)
    _, u_b_history = solver(sigma, T, max_sigma=max_sigma)
    return u_b_history

def main():
    # === Params ===
    M = 10
    T = 1.0
    device = 'cpu'
    max_sigma = 10
    alpha = 0.1
    sigma_0 = 1.0  # Initial guess
    lr = 1e-1
    noise_level = 0.01
    max_iters = 5000
    tol = 1e-4
    pattern = 'constant'
    source_func = create_source_function(pattern=pattern, device=device)
    sigma_gt = create_conductivity_field(M, pattern=pattern, device=device)

    # Generate boundary observations with noise
    u_b_gt = get_boundary_conditions(sigma_gt, source_func, T, max_sigma, device=device)
    u_b = (1 + noise_level * torch.randn_like(u_b_gt)) * u_b_gt

    # Inverse solver
    inverse_solver = InverseSolver(
        M=M,
        u_b_gt=u_b,
        source_func=source_func,
        n_steps=u_b.shape[0] - 1,
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
    