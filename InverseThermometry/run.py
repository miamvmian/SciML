import torch
from heat_solver import HeatSolver
from inverse_solver import InverseSolver
from utils import (
    verification_solution, verification_source, compute_l2_error, 
    compute_relative_error, visualize_solution, visualize_comparison, plot_convergence_analysis,
    print_solver_info, create_conductivity_field
)

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
    M = 10
    T = 1.0
    device = 'cpu'
    source_func = verification_source
    n_steps = None  # Auto-compute
    alpha = 0.1
    sigma_0 = 1.0  # Initial guess
    lr = 1e-1
    noise_level = 0.01
    max_iters = 5000
    tol = 1e-4
    sigma_gt = create_conductivity_field(2 * M, pattern='constant', value=1.0, device=device)

    u_b_gt = get_boundary_conditions(sigma_gt, source_func, T, device=device)
    u_b = (1 + noise_level * torch.randn_like(u_b_gt)) * u_b_gt

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
    