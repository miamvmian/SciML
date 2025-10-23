from IPython.display import display, HTML
from matplotlib.animation import FuncAnimation
import torch
import numpy as np
import matplotlib.pyplot as plt


def optimal_steppings(M, T, max_sigma):
    h = 1.0 / M
    tau = h**2 / (4 * max_sigma)
    iters = int(T / tau) + 1
    tau = T / iters
    return iters, tau, h


def boundary_mask(M, device):
    # Boundary indicator with True at the domain boundary
    # Order matters for downstream stacking/slicing consistency.
    mask = torch.zeros(M, M, dtype=torch.bool, device=device)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    return mask


class SimpleSigma(torch.nn.Module):
    def __init__(self, M, sigma_0):
        super().__init__()
        self.sigma = torch.nn.Parameter(
            torch.randn(M, M) * 0.1 + sigma_0, requires_grad=True
        )

    def forward(self):
        return self.sigma


class SigmoidSigma(torch.nn.Module):
    def __init__(self, M, min_sigma, max_sigma):
        super().__init__()
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.sigma_param = torch.nn.Parameter(torch.rand(M, M))

    def forward(self):
        return (
            torch.sigmoid(self.sigma_param) * (self.max_sigma - self.min_sigma)
            + self.min_sigma
        )


def create_conductivity_field(M=10, pattern="constant", device="cpu"):
    """
    Create different conductivity field patterns for testing.

    Args:
        M: grid size
        pattern: 'constant', 'linear', 'gaussian', 'checkerboard'
        device: device to create tensor on

    Returns:
        conductivity field [M, M]
    """
    h = 1.0 / M
    x = torch.linspace(0, 1, M + 1, device=device)[:-1] + h / 2
    y = torch.linspace(0, 1, M + 1, device=device)[:-1] + h / 2
    X, Y = torch.meshgrid(x, y, indexing="ij")

    if pattern == "constant":
        sigma = torch.ones(M, M, device=device)
    elif pattern == "linear":
        sigma = 1 + X + Y
    elif pattern == "gaussian":
        sigma = 1 + 0.5 * torch.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.1)
    elif pattern == "checkerboard":
        sigma = 1 + 0.5 * torch.sin(4 * np.pi * X) * torch.sin(4 * np.pi * Y)
    elif pattern == "sigmoid":
        # smooth bump centered at the domain center
        sigma = 1 + torch.sigmoid((X - 0.5) ** 2 + (Y - 0.5) ** 2)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return sigma


def sine_source(
    x: torch.Tensor, y: torch.Tensor, t: torch.Tensor | float, omega: float
):
    if isinstance(t, (int, float)):
        t_tensor = torch.tensor([t], dtype=x.dtype, device=x.device)
        sine = torch.sin(omega * t_tensor)
    else:
        t_tensor = t.to(x.device)
        sine = torch.sin(omega * t_tensor.unsqueeze(-1).unsqueeze(-1))
    return sine * torch.ones_like(x, device=x.device)


def sine_gauss_source(
    x: torch.Tensor, y: torch.Tensor, t: torch.Tensor | float, omega: float
):
    if isinstance(t, (int, float)):
        t_tensor = torch.tensor([t], dtype=x.dtype, device=x.device)
        return torch.sin(omega * t_tensor).abs() * torch.exp(
            -0.5 * ((x - 0.5) ** 2 + (y - 0.5) ** 2) * 6
        )
    else:
        t_tensor = t.to(x.device)
        return torch.sin(
            omega * t_tensor.unsqueeze(-1).unsqueeze(-1)
        ).abs() * torch.exp(-0.5 * ((x - 0.5) ** 2 - (y - 0.5) ** 2) * 6)


def sine_sine_source(
    x: torch.Tensor, y: torch.Tensor, t: torch.Tensor | float, omega: float
):
    if isinstance(t, (int, float)):
        t_tensor = torch.tensor([t], dtype=x.dtype, device=x.device)
        return (torch.sin(omega * t_tensor) + 1) * (
            torch.sin(2 * omega * x) * torch.sin(2 * omega * y) + 1
        )
    else:
        t_tensor = t.to(x.device)
        return (torch.sin(omega * t_tensor.unsqueeze(-1).unsqueeze(-1)) + 1) * (
            torch.sin(2 * omega * x) * torch.sin(2 * omega * y) + 1
        )


def sine_cosine_source(
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor | float,
    omega: float,
    device: str | torch.device | None = None,
):
    if isinstance(device, torch.device):
        target_device = device
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = x.device

    x_local = x.to(target_device)
    y_local = y.to(target_device)

    if isinstance(t, (int, float)):
        t_tensor = torch.tensor([t], dtype=x_local.dtype, device=target_device)
        return (
            (torch.sin(omega * t_tensor) + 1)
            * torch.cos(np.pi * x_local)
            * torch.cos(np.pi * y_local)
        )
    else:
        t_tensor = t.to(target_device)
        return (
            (torch.sin(omega * t_tensor.unsqueeze(-1).unsqueeze(-1)) + 1)
            * torch.cos(np.pi * x_local)
            * torch.cos(np.pi * y_local)
        )


def create_source_function(pattern="constant", device="cpu"):
    """
    Create different source term functions for testing.

    Args:
        pattern: 'constant', 'sinusoidal', 'localized'
        device: device to create tensors on

    Returns:
        source function f(x,y,t)
    """

    def constant_source(x, y, t):
        return torch.ones_like(x, device=device)

    def sinusoidal_source(x, y, t):
        return torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.exp(-t)

    def localized_source(x, y, t):
        return torch.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.02) * torch.sin(
            np.pi * t
        )

    if pattern == "constant":
        return constant_source
    elif pattern == "sinusoidal":
        return sinusoidal_source
    elif pattern == "localized":
        return localized_source
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def sinusoidal_source(x, y, t, spatial=True):
    """
    Sinusoidal-in-time source. If spatial=True, modulate by cos(πx)cos(πy)
    to avoid spatial uniformity (which would make the solution independent of σ
    under Neumann BCs). Returns an [M,M] tensor matching x/y.
    """
    if isinstance(t, (int, float)):
        t = torch.tensor(t, dtype=x.dtype, device=x.device)
    if spatial:
        return (
            torch.sin(torch.pi * t) * torch.cos(torch.pi * x) * torch.cos(torch.pi * y)
        )

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


def relative_rmse(pred, real):
    rmse = torch.sqrt(torch.mean(torch.square(real - pred)))
    mean = torch.mean(real)
    return (rmse / mean).item()


def r2_score(pred, real):
    mse = torch.mean(torch.square(real - pred))
    var = torch.var(real, correction=0)
    return (1.0 - mse / var).item()


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


def precompute_source_history(
    source_func, M, n_frames, tau, h, device, return_grid=False
):
    x = torch.linspace(0, 1, M + 1, device=device)[:-1] + h / 2
    y = torch.linspace(0, 1, M + 1, device=device)[:-1] + h / 2

    X, Y = torch.meshgrid(x, y, indexing="ij")

    ts = torch.arange(n_frames, device=device) * tau
    ts = ts[1:]

    with torch.no_grad():
        source_history = torch.stack([source_func(X, Y, t) for t in ts], dim=0)

    if return_grid:
        return source_history, ts, X, Y
    else:
        return source_history


def obtained_vs_true_conductivity(sigma_gt, res_sigma):
    sigma_gt_vis = torch.as_tensor(sigma_gt).detach().cpu()
    res_sigma_vis = torch.as_tensor(res_sigma).detach().cpu()
    fig_sigma, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(
        axes,
        (sigma_gt_vis, res_sigma_vis, torch.abs(res_sigma_vis - sigma_gt_vis)),
        ("Ground truth sigma", "Recovered sigma", "Absolute error"),
    ):
        im = ax.imshow(data, cmap="viridis", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig_sigma.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    display(fig_sigma)
    plt.close(fig_sigma)


def animate_history_for_comparison(u_history, u_gt_history, source_history, tau):
    # pad source history
    source_history = torch.cat(
        [torch.zeros_like(source_history[0]).unsqueeze(0), source_history]
    )

    # --- Original number of frames ---
    n_frames_full = u_history.shape[0]

    # --- Downscale to 100 evenly spaced frames if needed ---
    if n_frames_full > 100:
        idx = torch.linspace(0, n_frames_full - 1, 100).long()
        u_history = u_history[idx]
        u_gt_history = u_gt_history[idx]
        source_history = source_history[idx]
        # Use real times corresponding to selected frames
        ts = idx.to(torch.float32) * tau
    else:
        ts = torch.arange(n_frames_full, device=u_history.device) * tau

    n_frames = u_history.shape[0]

    # --- Convert to NumPy ---
    u_sim_np = u_history.detach().cpu().numpy()
    u_gt_np = u_gt_history.detach().cpu().numpy()
    residual_np = (u_history - u_gt_history).detach().cpu().numpy()
    source_np = source_history.detach().cpu().numpy()
    times_np = ts.detach().cpu().numpy()

    # --- Value limits ---
    vmin = float(min(u_sim_np.min(), u_gt_np.min()))
    vmax = float(max(u_sim_np.max(), u_gt_np.max()))
    res_lim = float(np.max(np.abs(residual_np))) or 1e-12
    source_lim = float(np.max(np.abs(source_np))) or 1e-12
    extent = [0.0, 1.0, 0.0, 1.0]

    # --- Figure setup ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    im_sim = axes[0].imshow(
        u_sim_np[0], cmap="viridis", origin="lower", extent=extent, vmin=vmin, vmax=vmax
    )
    axes[0].set_title("Simulation")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im_sim, ax=axes[0], fraction=0.046, pad=0.04, label="Temperature")

    im_gt = axes[1].imshow(
        u_gt_np[0], cmap="viridis", origin="lower", extent=extent, vmin=vmin, vmax=vmax
    )
    axes[1].set_title("Ground Truth")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im_gt, ax=axes[1], fraction=0.046, pad=0.04, label="Temperature")

    im_res = axes[2].imshow(
        residual_np[0],
        cmap="RdBu_r",
        origin="lower",
        extent=extent,
        vmin=-res_lim,
        vmax=res_lim,
    )
    axes[2].set_title("Residual (Simulation - Truth)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im_res, ax=axes[2], fraction=0.046, pad=0.04, label="Residual")

    im_src = axes[3].imshow(
        source_np[0],
        cmap="RdBu_r",
        origin="lower",
        extent=extent,
        vmin=-source_lim,
        vmax=source_lim,
    )
    axes[3].set_title("Source Term")
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("y")
    fig.colorbar(im_src, ax=axes[3], fraction=0.046, pad=0.04, label="Source")

    time_text = fig.suptitle(f"t = {times_np[0]:.3f}")

    # --- Animation update ---
    def _update(frame):
        im_sim.set_data(u_sim_np[frame])
        im_gt.set_data(u_gt_np[frame])
        im_res.set_data(residual_np[frame])
        im_src.set_data(source_np[frame])
        time_text.set_text(f"t = {times_np[frame]:.3f}")
        return im_sim, im_gt, im_res, im_src

    anim = FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=120,
        blit=False,
    )

    display(HTML(anim.to_jshtml()))
    plt.close(fig)

    rmse_over_time = np.sqrt(np.mean(residual_np**2, axis=(1, 2)))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts.numpy(), rmse_over_time, marker="o", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Residual RMS")
    ax.set_title("Residual RMS over Time")
    ax.grid(True, alpha=0.3)
    plt.show()
