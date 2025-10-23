import math
import random
import itertools
import numpy as np
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from IPython.display import display
from matplotlib import pyplot as plt
from pathlib import Path
import csv

from utils import (
    SimpleSigma,
    boundary_mask,
    precompute_source_history,
    create_conductivity_field,
    optimal_steppings,
    sine_cosine_source,
    sine_sine_source,
    sine_gauss_source,
    sine_source,
)
from heat_solver import solve_heat
from inverse_solver import estimate_conductivity

noise_levels = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
alphas = [0.0, 1e-4, 1e-2]
SOURCE_FREQS = {
    "sincos": (0.1, 0.3),
    "sinsin": (0.1, 0.3),
    "singa": (0.1, 0.3),
    "sin": (0.1, 0.3),
}
source_keys = [
    f"{prefix}{omega}" for prefix, omegas in SOURCE_FREQS.items() for omega in omegas
]
sources_k = [0.1, 2.0, 3.0]
sigmas = ("constant", "gaussian", "checkerboard", "sigmoid", "linear")
sigmas_k = [0.1, 1.0, 3.0]
WORKERS = 4
MAX_ITERS = 100
M = 10
T = 1.0
LR = 1e-2
MAX_SIGMA_HINT = 5.0
device = "cpu"

if "hypersearch_results" not in globals():
    hypersearch_results = np.empty((0, 8), dtype=object)

_source_prefix_order = sorted(SOURCE_FREQS.keys(), key=len, reverse=True)


def parse_source_key(source_key: str) -> tuple[str, float]:
    for prefix in _source_prefix_order:
        if source_key.startswith(prefix):
            omega_str = source_key[len(prefix) :]
            if not omega_str:
                raise ValueError(f"Missing frequency in source key: {source_key}")
            return prefix, float(omega_str)
    raise ValueError(f"Unsupported source key: {source_key}")


def create_source_function(source_key: str, source_k: float, device: torch.device):
    prefix, omega = parse_source_key(source_key)

    if prefix == "sincos":

        def base(x, y, t, omega=omega):
            return sine_cosine_source(x, y, t, omega=omega, device=device)

    elif prefix == "sinsin":

        def base(x, y, t, omega=omega):
            return sine_sine_source(x, y, t, omega=omega)

    elif prefix == "singa":

        def base(x, y, t, omega=omega):
            return sine_gauss_source(x, y, t, omega=omega)

    elif prefix == "sin":

        def base(x, y, t, omega=omega):
            return sine_source(x, y, t, omega=omega)

    else:
        raise ValueError(f"Unsupported source key: {source_key}")

    def gated_source(x, y, t):
        base_val = base(x, y, t)
        if isinstance(t, torch.Tensor):
            t_val = float(t.detach().cpu())
        else:
            t_val = float(t)
        scale = source_k if t_val < 0.5 else 0.0
        return base_val * scale

    return gated_source


def run_hyper_trial(task: dict) -> tuple:
    noise_level = task["noise_level"]
    alpha = task["alpha"]
    source_key = task["source_key"]
    source_k = task["source_k"]
    sigma_key = task["sigma_key"]
    sigma_k = task["sigma_k"]
    seed = task["seed"]
    device_str = task["device"]
    max_iters = task["max_iters"]
    lr = task["lr"]
    m_local = task["M"]
    t_local = task["T"]
    sigma_init = task["sigma_init"]
    max_sigma_hint = task["max_sigma_hint"]

    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        device_obj = torch.device(device_str)
        sigma_gt = sigma_k * create_conductivity_field(
            m_local, pattern=sigma_key, device=device_obj
        )
        max_sigma = max(max_sigma_hint, float(sigma_gt.max().item()) * 1.1)
        n_steps, tau, h = optimal_steppings(m_local, t_local, max_sigma)

        source_func = create_source_function(source_key, source_k, device_obj)
        source_history = precompute_source_history(
            source_func, m_local, n_steps, tau, h, device_obj
        )

        with torch.no_grad():
            u_init = torch.zeros_like(sigma_gt)
            u_gt_history = solve_heat(u_init, sigma_gt, source_history, h, tau)

        mask = boundary_mask(m_local, device_obj)
        torch.manual_seed(seed + 1)
        noise = 1 + noise_level * torch.randn_like(u_gt_history[:, mask])
        u_b_noisy = noise * u_gt_history[:, mask]

        sigma_field = SimpleSigma(m_local, sigma_init).to(device_obj)

        generator = estimate_conductivity(
            sigma_field,
            u_b_noisy,
            mask,
            source_history,
            sigma_init,
            alpha,
            h,
            tau,
            lr,
            max_iters,
        )

        final_sigma = None
        final_history = None
        for sigma_est, u_hist, *_ in generator:
            final_sigma = sigma_est.detach()
            final_history = u_hist.detach()

        if final_sigma is None:
            fin_rrmse = float("nan")
            fin_crmse = float("nan")
        else:
            fin_crmse = torch.sqrt(torch.mean((final_sigma - sigma_gt) ** 2)).item()
            fin_rrmse = torch.sqrt(
                torch.mean((final_history - u_gt_history) ** 2)
            ).item()

        return (
            noise_level,
            alpha,
            source_key,
            source_k,
            sigma_key,
            sigma_k,
            fin_rrmse,
            fin_crmse,
        )
    except Exception:
        return (
            noise_level,
            alpha,
            source_key,
            source_k,
            sigma_key,
            sigma_k,
            float("nan"),
            float("nan"),
        )


def format_output_fields(result: tuple) -> list[str]:
    (
        noise_level,
        alpha,
        source_key,
        source_k,
        sigma_key,
        sigma_k,
        fin_rrmse,
        fin_crmse,
    ) = result

    def fmt(value, fmt_code):
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return "nan"
        if math.isnan(value_float):
            return "nan"
        return format(value_float, fmt_code)

    return [
        fmt(noise_level, ".4f"),
        fmt(alpha, ".2e"),
        source_key,
        fmt(source_k, ".3f"),
        sigma_key,
        fmt(sigma_k, ".3f"),
        fmt(fin_rrmse, ".6f"),
        fmt(fin_crmse, ".6f"),
    ]


def plot_group(ax, rr, cr, groups, title):
    ax.set_title(title)
    ax.set_xlabel("fin_rrmse")
    ax.set_ylabel("fin_crmse")
    ax.grid(True, alpha=0.25)

    if rr.size == 0:
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center", fontsize=10)
        return

    unique_groups = list(dict.fromkeys(groups))
    cmap = plt.get_cmap("tab20", max(1, len(unique_groups)))

    for idx, group in enumerate(unique_groups):
        mask = groups == group
        if not np.any(mask):
            continue
        ax.scatter(
            rr[mask], cr[mask], color=cmap(idx), label=str(group), alpha=0.75, s=36
        )

    if len(unique_groups) <= 12:
        ax.legend(fontsize="x-small", loc="best", frameon=False)


def refresh_plot(fig, axes, display_handle, results_array):
    def _render():
        fig.canvas.draw_idle()
        if hasattr(display_handle, "update"):
            display_handle.update(fig)
        else:
            plt.pause(0.001)

    for axis in axes.flat:
        axis.cla()

    if not isinstance(results_array, np.ndarray) or results_array.size == 0:
        for axis in axes.flat:
            axis.set_xlabel("fin_rrmse")
            axis.set_ylabel("fin_crmse")
            axis.text(
                0.5, 0.5, "waiting for results", ha="center", va="center", fontsize=10
            )
        fig.suptitle("Hypersearch metrics (0 runs)")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        _render()
        return

    arr = results_array
    rr_all = arr[:, 6].astype(float)
    cr_all = arr[:, 7].astype(float)
    valid = ~(np.isnan(rr_all) | np.isnan(cr_all))

    rr = rr_all[valid]
    cr = cr_all[valid]

    group_specs = [
        ("By source", arr[:, 2].astype(str)[valid]),
        (
            "By source_k",
            np.array([f"{float(v):.3f}" for v in arr[:, 3][valid]], dtype=object),
        ),
        ("By sigma", arr[:, 4].astype(str)[valid]),
        (
            "By sigma_k",
            np.array([f"{float(v):.3f}" for v in arr[:, 5][valid]], dtype=object),
        ),
        (
            "By alpha",
            np.array([f"{float(v):.2e}" for v in arr[:, 1][valid]], dtype=object),
        ),
        (
            "By noise",
            np.array([f"{float(v):.3f}" for v in arr[:, 0][valid]], dtype=object),
        ),
    ]

    for axis, (title, groups) in zip(axes.flat, group_specs):
        plot_group(axis, rr, cr, groups, title)

    fig.suptitle(f"Hypersearch metrics ({len(arr)} runs)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    _render()


def dump_results_to_csv(results_array, file_path="hypersearch_results.csv"):
    header = [
        "noise_level",
        "alpha",
        "source",
        "source_k",
        "sigma",
        "sigma_k",
        "fin_rrmse",
        "fin_crmse",
    ]
    path = Path(file_path)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        if isinstance(results_array, np.ndarray) and results_array.size:
            for row in results_array:
                writer.writerow(row.tolist() if hasattr(row, "tolist") else list(row))


def sample_task() -> dict:
    sigma_scale = random.choice(sigmas_k)
    return {
        "noise_level": random.choice(noise_levels),
        "alpha": random.choice(alphas),
        "source_key": random.choice(source_keys),
        "source_k": random.choice(sources_k),
        "sigma_key": random.choice(sigmas),
        "sigma_k": sigma_scale,
        "sigma_init": sigma_scale,
        "seed": random.randrange(1 << 30),
        "device": device,
        "M": M,
        "T": T,
        "lr": LR,
        "max_iters": MAX_ITERS,
        "max_sigma_hint": MAX_SIGMA_HINT,
    }


def main():
    global hypersearch_results

    existing = []
    if isinstance(hypersearch_results, np.ndarray) and hypersearch_results.size:
        existing = [tuple(row) for row in hypersearch_results.tolist()]

    results_buffer = existing.copy()

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    display_handle = display(fig, display_id=True)
    refresh_plot(fig, axes, display_handle, hypersearch_results)

    ctx = mp.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx)
    futures: set = set()

    try:
        while True:
            while len(futures) < WORKERS:
                task = sample_task()
                futures.add(executor.submit(run_hyper_trial, task))

            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                futures.remove(fut)
                try:
                    result = fut.result()
                except Exception as exc:
                    print(f"worker failed: {exc}", flush=True)
                    continue

                results_buffer.append(result)
                hypersearch_results = np.array(results_buffer, dtype=object)
                print(
                    ",".join(format_output_fields(result)),
                    flush=True,
                )
                refresh_plot(fig, axes, display_handle, hypersearch_results)
    except KeyboardInterrupt:
        print("Stopping hyperparameter search.", flush=True)
    finally:
        for fut in futures:
            fut.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        hypersearch_results = np.array(results_buffer, dtype=object)
        refresh_plot(fig, axes, display_handle, hypersearch_results)
        dump_results_to_csv(hypersearch_results)


if __name__ == "__main__":
    main()
