"""
Visualization script for inverse thermometry results.
Plots:
- Estimated conductivity field
- True conductivity field
- Comparison (difference)
- Loss history
- Source waveform over time
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import strategy for utils
try:
    from .utils import create_conductivity_field, sinusoidal_source
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from utils import create_conductivity_field, sinusoidal_source


def plot_conductivity_comparison(sigma_est, sigma_true, save_path=None):
    """
    Plot estimated vs true conductivity fields with difference map.
    
    Args:
        sigma_est: estimated conductivity [M, M]
        sigma_true: true conductivity [M, M]
        save_path: path to save figure (optional)
    """
    # Convert to numpy if needed
    if isinstance(sigma_est, torch.Tensor):
        sigma_est = sigma_est.cpu().numpy()
    if isinstance(sigma_true, torch.Tensor):
        sigma_true = sigma_true.cpu().numpy()
    
    M = sigma_est.shape[0]
    h = 1.0 / M
    x = np.linspace(0, 1, M+1)[:-1] + h/2
    y = np.linspace(0, 1, M+1)[:-1] + h/2
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Compute difference
    diff = sigma_est - sigma_true
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # True conductivity
    im0 = axes[0, 0].contourf(X, Y, sigma_true, levels=20, cmap='viridis')
    axes[0, 0].set_title('True Conductivity σ(x,y)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Estimated conductivity
    im1 = axes[0, 1].contourf(X, Y, sigma_est, levels=20, cmap='viridis')
    axes[0, 1].set_title('Estimated Conductivity σ_est(x,y)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Difference
    vmax = np.max(np.abs(diff))
    im2 = axes[1, 0].contourf(X, Y, diff, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title('Difference (Est - True)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Statistics
    mse = np.mean(diff**2)
    mae = np.mean(np.abs(diff))
    max_err = np.max(np.abs(diff))
    rel_err = np.linalg.norm(diff) / np.linalg.norm(sigma_true)
    
    stats_text = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nMax |Error|: {max_err:.6f}\nRel. Error: {rel_err:.6f}'
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=14, family='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Error Statistics', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved conductivity comparison to {save_path}")
    
    plt.show()
    
    return mse, mae, max_err, rel_err


def plot_loss_history(loss_history: dict[str, np.ndarray]|np.ndarray|torch.Tensor, save_path=None, log_scale=False):
    """
    Plot one or multiple optimization loss histories.
    
    Args:
        loss_history: either a 1D array-like (single series) or a dict
                      mapping label -> 1D array-like (multiple series)
        save_path: path to save figure (optional)
        log_scale: use log scale for y-axis
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Multiple series: dict of label -> array-like
    if isinstance(loss_history, dict):
        for label, series in loss_history.items():
            if isinstance(series, torch.Tensor):
                series = series.detach().cpu().numpy()
            series = np.asarray(series)
            if series.size == 0:
                continue
            iterations = np.arange(1, len(series) + 1)
            ax.plot(iterations, series, linewidth=2, label=str(label))
    else:
        # Single series
        series = loss_history
        if isinstance(series, torch.Tensor):
            series = series.detach().cpu().numpy()
        series = np.asarray(series)
        iterations = np.arange(1, len(series) + 1)
        ax.plot(iterations, series, linewidth=2, color='steelblue', label='Loss')
        # Annotate final value for single-series case
        if series.size > 0:
            final_loss = float(series[-1])
            ax.annotate(f'Final: {final_loss:.6e}',
                        xy=(len(series), final_loss),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, color='red',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='red'))
    
    if log_scale:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Optimization Loss History', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved loss history to {save_path}")
    
    plt.show()


def plot_source_waveform(T=1.0, n_points=1000, save_path=None):
    """
    Plot the source term waveform over time.
    
    Args:
        T: total time
        n_points: number of time points to sample
        save_path: path to save figure (optional)
    """
    t = np.linspace(0, T, n_points)
    
    # Evaluate source at a fixed spatial point (e.g., center)
    x_center = torch.tensor(0.5)
    y_center = torch.tensor(0.5)
    
    # Source function: sin(π*t)
    f_values = []
    for ti in t:
        f_val = sinusoidal_source(x_center, y_center, torch.tensor(ti))
        if isinstance(f_val, torch.Tensor):
            f_val = f_val.item()
        f_values.append(f_val)
    
    f_values = np.array(f_values)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t, f_values, linewidth=2.5, color='darkred', label='f(x,y,t) = sin(πt)')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('Source f(x,y,t)', fontsize=12)
    ax.set_title('Source Term Waveform: f(x,y,t) = sin(πt)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved source waveform to {save_path}")
    
    plt.show()


def plot_source_spatial(M=10, t=None, T=1.0, spatial=True, device='cpu', save_path=None):
    """
    Plot the 2D spatial distribution of the source f(x,y,t) at a given time.
    
    Args:
        M: grid size
        t: time to visualize (defaults to T/2 if None)
        T: total time (used only if t is None)
        spatial: pass spatial=True to modulate by cos(2πx)cos(2πy)
        device: torch device
        save_path: path to save figure (optional)
    """
    # Build grid at cell centers
    h = 1.0 / M
    x = torch.linspace(0, 1, M+1, device=device)[:-1] + h/2
    y = torch.linspace(0, 1, M+1, device=device)[:-1] + h/2
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Choose time
    if t is None:
        t = T * 0.5
    t_torch = torch.tensor(float(t), dtype=X.dtype, device=X.device)
    
    # Evaluate source
    F = sinusoidal_source(X, Y, t_torch, spatial=spatial)
    
    # Plot
    Xn = X.detach().cpu().numpy()
    Yn = Y.detach().cpu().numpy()
    Fn = F.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.contourf(Xn, Yn, Fn, levels=20, cmap='RdBu_r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_title(f'Source f(x,y,t) at t={float(t):.3f}{" (spatial)" if spatial else ""}', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved source spatial plot to {save_path}")
    
    plt.show()


def plot_all_results(results_dir='InverseThermometry/results', 
                     images_dir='InverseThermometry/images',
                     M=10, 
                     T=1.0,
                     conductivity_type='linear',
                     device='cpu'):
    """
    Load and plot all results: sigma_est, loss_history, source waveform, and true sigma.
    
    Args:
        results_dir: directory containing sigma_est.npy and loss_history.npy
        images_dir: directory to save output figures
        M: grid size (must match the one used in training)
        T: total time
        conductivity_type: type of true conductivity field ('linear', 'constant', 'sigmoid')
        device: torch device
    """
    print("Loading results...")
    
    # Load estimated conductivity
    sigma_est_path = os.path.join(results_dir, 'sigma_est.npy')
    if not os.path.exists(sigma_est_path):
        raise FileNotFoundError(f"Cannot find {sigma_est_path}. Run inverse solver first.")
    sigma_est = torch.tensor(np.load(sigma_est_path), device=device)
    
    # Load loss history
    total_loss_history_path = os.path.join(results_dir, 'total_loss_history.npy')
    if not os.path.exists(total_loss_history_path):
        raise FileNotFoundError(f"Cannot find {total_loss_history_path}. Run inverse solver first.")
    total_loss_history = np.load(total_loss_history_path)
    data_loss_history_path = os.path.join(results_dir, 'data_loss_history.npy')
    if not os.path.exists(data_loss_history_path):
        raise FileNotFoundError(f"Cannot find {data_loss_history_path}. Run inverse solver first.")
    data_loss_history = np.load(data_loss_history_path)
    reg_loss_history_path = os.path.join(results_dir, 'reg_loss_history.npy')
    if not os.path.exists(reg_loss_history_path):
        raise FileNotFoundError(f"Cannot find {reg_loss_history_path}. Run inverse solver first.")
    reg_loss_history = np.load(reg_loss_history_path)
    
    # Create true conductivity field
    print(f"Creating true conductivity field (type: {conductivity_type})...")
    sigma_true = create_conductivity_field(M, conductivity_type, device=device)
    
    # Create output directory
    os.makedirs(images_dir, exist_ok=True)
    
    # Plot conductivity comparison
    print("\nPlotting conductivity comparison...")
    plot_conductivity_comparison(
        sigma_est, 
        sigma_true,
        save_path=os.path.join(images_dir, 'conductivity_comparison.png')
    )
    
    # Plot loss history
    print("\nPlotting loss history...")
    loss_history = {
        'total': total_loss_history,
        'data': data_loss_history,
        'reg': reg_loss_history
    }
    plot_loss_history(
        loss_history,
        save_path=os.path.join(images_dir, 'loss_history.png')
    )
    
    # Plot source waveform
    print("\nPlotting source waveform...")
    plot_source_waveform(
        T=T,
        save_path=os.path.join(images_dir, 'source_waveform.png')
    )
    
    # Plot 2D spatial source at mid-time
    print("\nPlotting 2D spatial source at t=T/2...")
    plot_source_spatial(
        M=M,
        t=T*0.5,
        T=T,
        spatial=True,
        device=device,
        save_path=os.path.join(images_dir, 'source_spatial_tmid.png')
    )
    
    print(f"\nAll plots saved to {images_dir}/")


def plot_boundary_last_column(results_dir='InverseThermometry/results',
                              data_path='InverseThermometry/data/boundary_dataset.npz',
                              images_dir='InverseThermometry/images',
                              when='final'  # 'final' or int index
                              ):
    """
    Plot the boundary temperatures along the last spatial column (right boundary)
    for both prediction (U_pred.npy) and dataset (boundary_dataset.npz) at a chosen time.

    Saves a figure boundary_last_column.png in images_dir.
    """
    u_pred_path = os.path.join(results_dir, 'U_pred.npy')
    if not os.path.exists(u_pred_path):
        raise FileNotFoundError(f"Cannot find {u_pred_path}. Run training to produce U_pred.npy.")
    U_pred = np.load(u_pred_path)  # [K, m]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find {data_path}. Generate the dataset first.")
    data = np.load(data_path)
    coords = data['coords']  # [m, 2]
    time_samples = data['time_samples']  # [K]
    # Load both clean (true) and noisy measurements
    U_clean = data['u_clean']  # true temperature
    U_noisy = data['d_noisy']  # noisy measurements

    # Align K in case of slight mismatch
    K = min(U_pred.shape[0], U_clean.shape[0], U_noisy.shape[0], len(time_samples))
    U_pred = U_pred[:K]
    U_clean = U_clean[:K]
    U_noisy = U_noisy[:K]
    time_samples = time_samples[:K]

    # Determine time index
    if when == 'final':
        k = K - 1
    elif isinstance(when, int):
        k = int(np.clip(when, 0, K - 1))
    else:
        k = K - 1

    # Identify right boundary (last column): x close to its maximum
    x = coords[:, 0]
    y = coords[:, 1]
    x_max = np.max(x)
    tol = 1e-9
    right_idx = np.where(x >= x_max - tol)[0]
    if right_idx.size == 0:
        raise RuntimeError("Could not locate right boundary indices from coords.")

    # Sort selected indices by y to make a proper line plot from bottom to top
    order = np.argsort(y[right_idx])
    sel = right_idx[order]

    y_line = y[sel]
    u_pred_line = U_pred[k, sel]
    u_clean_line = U_clean[k, sel]
    u_noisy_line = U_noisy[k, sel]

    # Plot
    os.makedirs(images_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(y_line, u_clean_line, 'o-', label='True ($u$)', linewidth=2, markersize=5)
    ax.plot(y_line, u_pred_line, '^-', label='Predicted ($u_{pred}$)', linewidth=2, markersize=5)
    ax.plot(y_line, u_noisy_line, 's--', label='Noisy ($d_{sk}$)', linewidth=2, alpha=0.7)
    ax.set_xlabel('y (right boundary x=1)')
    ax.set_ylabel('Boundary temperature')
    ax.set_title(f'Boundary Temperatures at t={float(time_samples[k]):.4f}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(images_dir, 'boundary_last_column.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved last-column boundary plot to {out_path}")
    plt.show()


    # (animation functionality removed at user request)



def main():
    """
    Run with simple in-file defaults. This entrypoint now generates all plots
    (conductivity comparison, loss, source waveform, 2D source snapshot) and the
    last-column boundary comparison.
    """
    results_dir = 'InverseThermometry/results'
    images_dir = 'InverseThermometry/images'
    data_path = 'InverseThermometry/data/boundary_dataset.npz'
    # Match these to your training/data generation settings
    M = 10
    T = 1.0
    conductivity_type = 'linear'
    device = 'cpu'

    # Full set of static plots with saved figures
    plot_all_results(
        results_dir=results_dir,
        images_dir=images_dir,
        M=M,
        T=T,
        conductivity_type=conductivity_type,
        device=device
    )

    # Last-column boundary plot (U_pred vs d_noisy) at final time
    plot_boundary_last_column(
        results_dir=results_dir,
        data_path=data_path,
        images_dir=images_dir,
        when='final'
    )
    


if __name__ == '__main__':
    main()

