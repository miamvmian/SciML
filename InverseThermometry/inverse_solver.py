import torch

from heat_solver import solve_heat


def estimate_conductivity(
    sigma_field,
    u_b_gt,
    mask,
    source_history,
    sigma_0: float,
    alpha: float,
    h: float,
    tau: float,
    lr: float = 1e-2,
    max_iters: int = 100,
):
    optimizer = torch.optim.Adam(sigma_field.parameters(), lr=lr)

    for _ in range(max_iters):
        sigma = sigma_field()
        u_history = solve_heat(
            torch.zeros_like(sigma),
            sigma,
            source_history,
            h,
            tau,
        )
        u_b_history = u_history[:, mask]

        loss_data = h * tau * (u_b_history - u_b_gt).square().sum()
        loss_reg = h**2 * (sigma - sigma_0).square().sum()
        loss = loss_data + alpha * loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: also calc some metrics

        yield loss_data.item(), loss_reg.item(), loss.item()
