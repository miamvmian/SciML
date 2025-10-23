import torch


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
        sigma_left = sigma[:-1, :]
        sigma_right = sigma[1:, :]
        return 2 / (1 / sigma_left + 1 / sigma_right)

    elif axis == 1:  # Y-direction
        sigma_bottom = sigma[:, :-1]
        sigma_top = sigma[:, 1:]
        return 2 / (1 / sigma_bottom + 1 / sigma_top)


def harmonic_mean(a, b, eps=1e-12):
    return 2.0 * a * b / (a + b + eps)


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

    # Physical fluxes at interfaces
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


def fv_euler_step_neumann(
    u: torch.Tensor,
    sigma: torch.Tensor,
    f: torch.Tensor,
    h: float,
    tau: float,
):
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

    # For boundary faces, use sigma_face = cell value (harmonic with itself)
    s_iphalf_j[..., -1, :] = sigma[..., -1, :]
    s_imhalf_j[..., 0, :] = sigma[..., 0, :]
    s_ijphalf[..., :, -1] = sigma[..., :, -1]
    s_ijmhalf[..., :, 0] = sigma[..., :, 0]

    # FV sum
    S = s_iphalf_j * dU + s_imhalf_j * dD + s_ijphalf * dR + s_ijmhalf * dL

    u_next = u + tau * (f - S / (h * h))
    return u_next


def solve_heat(u_init, sigma, fs, h, tau, include_init=False):
    if include_init:
        evolution = [u_init]
    else:
        evolution = []
    u = u_init.clone()
    for f in fs:
        u = heat_step(u, sigma, f, h, tau)
        evolution.append(u)
    return torch.stack(evolution, dim=0)
