"""
Utility functions for heat equation solver verification and visualization
"""

from IPython.display import display, HTML
from matplotlib.animation import FuncAnimation
import torch
import numpy as np
import matplotlib.pyplot as plt


def relative_rmse(pred, real):
    rmse = torch.sqrt(torch.mean(torch.square(real - pred)))
    mean = torch.mean(real)
    return (rmse / mean).item()


def r2_score(pred, real):
    mse = torch.mean(torch.square(real - pred))
    var = torch.var(real, correction=0)
    return (1.0 - mse / var).item()


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
    # Convert t to tensor if it's a float
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
    # Convert t to tensor if it's a float
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


def compute_l2_error(u_numerical, u_analytical):
    """
    Compute L2 error between numerical and analytical solutions.

    Args:
        u_numerical: numerical solution [M, M]
        u_analytical: analytical solution [M, M]

    Returns:
        L2 error
    """
    error = u_numerical - u_analytical
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
    l2_analytical = torch.sqrt(torch.mean(u_analytical**2))
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

    # 2D contour plot
    im1 = ax1.contourf(x, y, u, levels=20, cmap="viridis")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(f"{title} - Contour")
    ax1.set_aspect("equal")
    plt.colorbar(im1, ax=ax1, label="Temperature")

    # 3D surface plot
    ax2 = fig.add_subplot(122, projection="3d")
    surf = ax2.plot_surface(x, y, u, cmap="viridis", alpha=0.8)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("Temperature")
    ax2.set_title(f"{title} - 3D Surface")

    plt.tight_layout()
    plt.show()


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
    im1 = axes[0, 0].contourf(
        x.detach().numpy(),
        y.detach().numpy(),
        u_numerical.detach().numpy(),
        levels=20,
        cmap="viridis",
    )
    axes[0, 0].set_title("Numerical Solution")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].set_aspect("equal")
    plt.colorbar(im1, ax=axes[0, 0], label="Temperature")

    # Analytical solution
    im2 = axes[0, 1].contourf(
        x.detach().numpy(),
        y.detach().numpy(),
        u_analytical.detach().numpy(),
        levels=20,
        cmap="viridis",
    )
    axes[0, 1].set_title("Analytical Solution")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    axes[0, 1].set_aspect("equal")
    plt.colorbar(im2, ax=axes[0, 1], label="Temperature")

    # Error
    im3 = axes[1, 0].contourf(
        x.detach().numpy(),
        y.detach().numpy(),
        error.detach().numpy(),
        levels=20,
        cmap="RdBu_r",
    )
    axes[1, 0].set_title("Error (Numerical - Analytical)")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    axes[1, 0].set_aspect("equal")
    plt.colorbar(im3, ax=axes[1, 0], label="Error")

    # Cross-section at y = 0.5
    y_mid = y.shape[0] // 2
    axes[1, 1].plot(
        x[y_mid, :].detach().numpy(),
        u_numerical[y_mid, :].detach().numpy(),
        "b-",
        label="Numerical",
        linewidth=2,
    )
    axes[1, 1].plot(
        x[y_mid, :].detach().numpy(),
        u_analytical[y_mid, :].detach().numpy(),
        "r--",
        label="Analytical",
        linewidth=2,
    )
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("Temperature")
    axes[1, 1].set_title(f"Cross-section at y = {y[y_mid, 0]:.2f}")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_convergence_analysis(h_values, errors, title="Convergence Analysis"):
    """
    Plot convergence analysis.

    Args:
        h_values: list of grid spacings
        errors: list of corresponding errors
        title: plot title
    """
    plt.figure(figsize=(10, 6))

    plt.loglog(
        h_values, errors, "bo-", linewidth=2, markersize=8, label="Numerical Error"
    )

    # Reference lines for convergence rates
    h_ref = np.array(h_values)
    plt.loglog(h_ref, h_ref, "r--", alpha=0.7, label="O(h)")
    plt.loglog(h_ref, h_ref**2, "g--", alpha=0.7, label="O(h²)")

    plt.xlabel("Grid Spacing h")
    plt.ylabel("L2 Error")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


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
    print(f"Stability limit: τ_max = {1.0/(4*M**2*sigma_max):.6f}")
    print(f"CFL number: {tau * 4 * sigma_max * M**2:.4f}")
    print("=" * 50)


def precompute_source_history(source_func, M, n_frames, tau, device):
    h = 1.0 / M
    x = torch.linspace(0, 1, M + 1, device=device)[:-1] + h / 2  # Cell centers
    y = torch.linspace(0, 1, M + 1, device=device)[:-1] + h / 2

    # Create coordinate grids
    X, Y = torch.meshgrid(x, y, indexing="ij")

    ts = torch.arange(n_frames, device=device) * tau

    with torch.no_grad():
        source_history = torch.stack([source_func(X, Y, t) for t in ts], dim=0)

    return source_history


def compare(u_history, u_gt_history, source_history, tau):
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
