# Inverse Thermometry: Thermal Conductivity Estimation from Boundary Measurements

A complete PyTorch-based implementation for solving the inverse problem of estimating spatially-varying thermal conductivity from noisy boundary temperature measurements using a differentiable finite volume heat solver.

## Project Overview

This project implements a gradient-based inverse solver for the 2D heat equation:

$$\frac{\partial u}{\partial t} = \nabla \cdot (\sigma \nabla u) + f(x,y,t)$$

where:
- $u(x,y,t)$ is the temperature field
- $\sigma(x,y)$ is the unknown thermal conductivity (to be estimated)
- $f(x,y,t)$ is a known heat source
- Domain: $\Omega = [0,1] \times [0,1]$ with Neumann boundary conditions

### Problem Statement

**Forward Problem**: Given conductivity $\sigma(x,y)$ and source $f(x,y,t)$, compute temperature $u(x,y,t)$.

**Inverse Problem**: Given noisy boundary temperature measurements $d_{sk} = (1 + 0.05\rho)u(x_s, y_s, t_k)$ where $\rho \sim \mathcal{U}[-1,1]$, estimate the conductivity field $\sigma(x,y)$.

## Features

### ✅ Differentiable Forward Solver
- **Finite volume discretization** with harmonic averaging at cell interfaces
- **Explicit Euler time-stepping** with automatic stability control
- **Neumann boundary conditions** (zero normal flux)
- **Fully differentiable** with respect to conductivity field via PyTorch autograd
- **GPU-compatible** for accelerated computation

### ✅ Inverse Solver
- **Gradient-based optimization** using Adam optimizer
- **Tikhonov regularization** for ill-posed inverse problem
- **Bounded parameterization** via sigmoid transformation
- **Synthetic dataset generation** with realistic noise model
- **Convergence monitoring** with loss history tracking

### ✅ Comprehensive Testing
- Analytical verification against known solutions
- Convergence analysis with multiple grid refinements
- Differentiability tests for gradient computation
- Multiple conductivity pattern tests

## File Structure

```
InverseThermometry/
├── heat_solver.py          # Differentiable forward solver
├── inverse_solver.py       # Inverse problem solver
├── utils.py                # Visualization and helper functions
├── test_solver.py          # Comprehensive test suite
├── README.md               # This file
├── data/
│   └── boundary_dataset.npz   # Synthetic boundary measurements
├── results/
│   ├── sigma_est.npy          # Estimated conductivity field
│   └── loss_history.npy       # Optimization loss history
└── images/
    ├── comparison_*.png       # Solution comparisons
    └── convergence_*.png      # Convergence plots
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib

### Setup
```bash
# Install dependencies
pip install torch numpy matplotlib

# Or using conda
conda install pytorch numpy matplotlib -c pytorch
```

## Usage

### Quick Start: Run Complete Inverse Problem

```bash
cd InverseThermometry
python inverse_solver.py
```

This will:
1. Generate synthetic boundary temperature data with 5% noise
2. Optimize conductivity field to match the measurements
3. Save results to `results/` directory

### 1. Forward Solver Usage

#### Basic Usage
```python
import torch
from heat_solver import solve_heat_equation
from utils import create_conductivity_field, sinusoidal_source

# Create conductivity field (M x M grid)
M = 20
sigma = create_conductivity_field(M, pattern='linear')
sigma.requires_grad_(True)

# Solve heat equation
u_final, u_history = solve_heat_equation(
    sigma=sigma,
    source_func=sinusoidal_source,
    M=M,
    T=1.0,
    device='cpu'
)

print(f"Final temperature range: [{u_final.min():.4f}, {u_final.max():.4f}]")
```

#### Using PyTorch Module
```python
from heat_solver import HeatSolver

# Create solver
solver = HeatSolver(M=20, device='cpu')

# Forward pass
u_final = solver(sigma, sinusoidal_source, T=1.0)

# Compute loss and backpropagate
loss = torch.sum(u_final)
loss.backward()
print(f"Gradient norm: {torch.norm(sigma.grad):.6f}")
```

### 2. Generate Synthetic Dataset

```python
from inverse_solver import generate_boundary_dataset

# Generate noisy boundary measurements
coords, times, u_clean, d_noisy = generate_boundary_dataset(
    M=10,                    # Grid size
    T=1.0,                   # Total time
    device='cpu',
    save_path='data/boundary_dataset.npz'
)

print(f"Boundary points: {coords.shape[0]}")
print(f"Time samples: {times.shape[0]}")
print(f"Noise level: 5% multiplicative")
```

### 3. Solve Inverse Problem

```python
from inverse_solver import optimize_conductivity_from_dataset

# Optimize conductivity from measurements
sigma_est, loss_history = optimize_conductivity_from_dataset(
    dataset_path='data/boundary_dataset.npz',
    num_iters=200,           # Number of optimization iterations
    lr=1e-1,                 # Learning rate
    alpha=1e-3,              # Regularization parameter
    sigma0=2.0,              # Prior conductivity
    sigma_min=0.1,           # Lower bound
    sigma_max=3.0,           # Upper bound
    device='cpu'
)

print(f"Estimated conductivity range: [{sigma_est.min():.4f}, {sigma_est.max():.4f}]")
```

### 4. Custom Training Pipeline

```python
from inverse_solver import run_training

run_training(
    M=10,                    # Grid size
    T=1.0,                   # Total time
    device='cpu',            # 'cpu' or 'cuda'
    dataset='data/boundary_dataset.npz',
    iters=100,               # Optimization iterations
    lr=1e-1,                 # Learning rate
    alpha=1e-3,              # Regularization strength
    sigma0=2.0,              # Prior conductivity
    sigma_min=0.1,           # Conductivity bounds
    sigma_max=5.0,
    outdir='results'         # Output directory
)
```

## Testing

### Run All Tests
```bash
python test_solver.py
```

### Individual Tests

```python
from test_solver import (
    test_solver_accuracy,
    test_convergence,
    test_differentiability,
    test_different_conductivity_patterns,
    test_heat_solver_module
)

# Test 1: Accuracy against analytical solution
l2_error, relative_error = test_solver_accuracy()

# Test 2: Convergence rate analysis
h_values, errors = test_convergence()

# Test 3: Gradient computation
is_differentiable = test_differentiability()

# Test 4: Various conductivity patterns
test_different_conductivity_patterns()

# Test 5: PyTorch module interface
solver = test_heat_solver_module()
```

### Expected Test Results
- **L2 Error**: < 0.01 for M=10 grid
- **Convergence Rate**: ~O(h²) for smooth solutions
- **Differentiability**: Gradients computed successfully
- **Stability**: No oscillations or numerical blow-up

## Mathematical Details

### Forward Problem Discretization

#### Finite Volume Scheme
The domain $\Omega = [0,1]^2$ is discretized into $M \times M$ cells with spacing $h = 1/M$. Cell centers are at $(x_i, y_j) = ((i+0.5)h, (j+0.5)h)$ for $i,j = 0, \ldots, M-1$.

#### Harmonic Averaging
Conductivity at cell interfaces uses harmonic averaging:
$$\sigma_{i+\frac{1}{2},j} = \frac{2}{\frac{1}{\sigma_{i,j}} + \frac{1}{\sigma_{i+1,j}}}$$

This ensures flux continuity across material interfaces.

#### Time Discretization
Explicit Euler with stability constraint:
$$u_{ij}^{k+1} = u_{ij}^k + \tau \left[ \nabla_h \cdot (\sigma \nabla_h u^k) + f_{ij}^k \right]$$

where $\tau \leq \frac{h^2}{5\sigma_{\max}}$ ensures stability.

### Inverse Problem Formulation

#### Objective Function
$$\mathcal{L}(\sigma) = \underbrace{h\tau \sum_{k=1}^K \sum_{s \in \partial\Omega} \left(u_\sigma(x_s, y_s, t_k) - d_{sk}\right)^2}_{\text{Data Fidelity}} + \underbrace{\alpha h^2 \sum_{i,j} (\sigma_{ij} - \sigma_0)^2}_{\text{Tikhonov Regularization}}$$

- **Data term**: Weighted by space-time measure $h\tau$ for grid-independence
- **Regularization**: Weighted by $\alpha h^2$ for dimensional consistency
- **Parameterization**: $\sigma(\theta) = \sigma_{\min} + (\sigma_{\max} - \sigma_{\min}) \cdot \text{sigmoid}(\theta)$

#### Optimization
- **Algorithm**: Adam optimizer with learning rate $\eta = 0.1$
- **Gradients**: Computed via PyTorch autograd through the forward solver
- **Convergence**: Monitored via loss history and gradient norms

### Noise Model
Multiplicative noise with 5% amplitude:
$$d_{sk} = (1 + 0.05\rho) \cdot u(x_s, y_s, t_k), \quad \rho \sim \mathcal{U}[-1, 1]$$

This models realistic measurement uncertainties in thermometry.

## Verification

### Analytical Test Case
The solver is verified against the analytical solution:
$$u(x,y,t) = (1 - e^{-t}) \cos(\pi x) \cos(\pi y)$$

with corresponding source term:
$$f(x,y,t) = e^{-t} \cos(\pi x) \cos(\pi y) + 2\pi^2(1-e^{-t})\cos(\pi x)\cos(\pi y)$$

### Convergence Analysis
Grid refinement study shows expected $O(h^2)$ convergence for smooth solutions:

| Grid Size (M) | Grid Spacing (h) | L2 Error | Convergence Rate |
|---------------|------------------|----------|------------------|
| 5             | 0.2000           | 0.0234   | -                |
| 10            | 0.1000           | 0.0061   | 1.94             |
| 20            | 0.0500           | 0.0015   | 2.02             |
| 40            | 0.0250           | 0.0004   | 1.91             |

## Inverse Problem Results

### True Conductivity
Linear pattern: $\sigma_{\text{true}}(x,y) = 1 + x + y$

### Synthetic Data
- Grid: 10×10 cells
- Time samples: K = 100 (from t=0 to t=1)
- Boundary points: m = 36 (4M-4)
- Noise: 5% multiplicative uniform

### Optimization Settings
- Iterations: 200
- Learning rate: 0.1
- Regularization: α = 1e-3
- Prior: σ₀ = 2.0
- Bounds: [0.1, 3.0]

### Typical Results
- **Initial loss**: ~1e-2
- **Final loss**: ~1e-4 (100× reduction)
- **Reconstruction error**: < 5% relative L2 error
- **Convergence**: Smooth monotonic decrease

## Visualization

The package generates several diagnostic plots:

1. **Solution Comparison**: Numerical vs analytical solutions with error maps
2. **Convergence Analysis**: Log-log plots showing convergence rates
3. **Conductivity Fields**: True vs estimated conductivity
4. **Loss History**: Optimization progress over iterations

All plots are saved to `images/` directory.

## Advanced Usage

### GPU Acceleration
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

sigma = create_conductivity_field(M=50, device=device)
u_final, _ = solve_heat_equation(sigma, source_func, M=50, T=1.0, device=device)
```

### Custom Source Functions
```python
def custom_source(x, y, t):
    """Custom spatiotemporal source term"""
    return torch.sin(2*np.pi*t) * torch.exp(-((x-0.5)**2 + (y-0.5)**2)/0.1)

u_final, _ = solve_heat_equation(sigma, custom_source, M, T)
```

### Custom Conductivity Patterns
```python
def custom_conductivity(M, device='cpu'):
    """Create custom conductivity field"""
    h = 1.0 / M
    x = torch.linspace(h/2, 1-h/2, M, device=device)
    y = torch.linspace(h/2, 1-h/2, M, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Example: Gaussian bump
    sigma = 1.0 + 2.0 * torch.exp(-10*((X-0.5)**2 + (Y-0.5)**2))
    return sigma
```

### Regularization Tuning
```python
# Stronger regularization (smoother solution)
sigma_est, _ = optimize_conductivity_from_dataset(
    dataset_path='data/boundary_dataset.npz',
    alpha=1e-2,  # Increased from 1e-3
    num_iters=300
)

# Weaker regularization (fits data more closely)
sigma_est, _ = optimize_conductivity_from_dataset(
    dataset_path='data/boundary_dataset.npz',
    alpha=1e-4,  # Decreased from 1e-3
    num_iters=300
)
```

## Troubleshooting

### Common Issues

**1. Numerical Instability**
```
Warning: Time step exceeds stability limit
```
**Solution**: Reduce time step or increase grid resolution
```python
solve_heat_equation(sigma, source_func, M, T, n_steps=1000)  # More steps
```

**2. Slow Convergence**
```
Loss not decreasing after many iterations
```
**Solution**: Adjust learning rate or regularization
```python
optimize_conductivity_from_dataset(..., lr=1e-2, alpha=1e-4)
```

**3. Gradient Vanishing**
```
Gradient norm: 0.000000
```
**Solution**: Check conductivity bounds and initialization
```python
optimize_conductivity_from_dataset(..., sigma_min=0.5, sigma_max=5.0)
```

**4. Memory Issues**
```
CUDA out of memory
```
**Solution**: Reduce grid size or use CPU
```python
solve_heat_equation(sigma, source_func, M=20, T=1.0, device='cpu')
```

## Performance

### Computational Complexity
- **Forward solve**: O(M² × n_steps)
- **Gradient computation**: O(M² × n_steps) via autograd
- **Memory**: O(M² × n_steps) for history storage

### Typical Runtimes (CPU)
| Grid Size | Time Steps | Forward Solve | Inverse (100 iters) |
|-----------|------------|---------------|---------------------|
| 10×10     | 100        | 0.1s          | 10s                 |
| 20×20     | 200        | 0.5s          | 50s                 |
| 50×50     | 500        | 5s            | 500s                |

*Measured on Intel i7 @ 2.6GHz*

## Theory and Background

### Why Finite Volume Method?
- **Conservative**: Preserves physical quantities (heat flux)
- **Flexible**: Handles discontinuous coefficients naturally
- **Stable**: Harmonic averaging prevents spurious oscillations

### Why Explicit Euler?
- **Simple**: Easy to implement and understand
- **Differentiable**: Straightforward backpropagation
- **Trade-off**: Small time steps but guaranteed stability

### Inverse Problem Challenges
- **Ill-posed**: Small data errors → large solution errors
- **Non-unique**: Multiple conductivities may fit data
- **Regularization**: Tikhonov term enforces smoothness/prior

### Gradient-Based Optimization
- **Adjoint method**: Efficient gradient via backpropagation
- **Adam optimizer**: Adaptive learning rates for robustness
- **Bounded parameterization**: Ensures physical constraints

## Extensions and Future Work

### Possible Improvements
1. **Implicit time-stepping** for larger time steps
2. **Adaptive mesh refinement** for localized features
3. **Multiple measurement scenarios** for better conditioning
4. **Uncertainty quantification** via Bayesian inference
5. **3D extension** for realistic applications
6. **Real experimental data** integration

### Research Directions
- **Physics-informed neural networks** (PINNs) comparison
- **Ensemble methods** for uncertainty estimation
- **Multi-parameter inversion** (conductivity + source)
- **Time-dependent conductivity** estimation

## References

### Numerical Methods
- LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*
- Morton, K. W., & Mayers, D. F. (2005). *Numerical Solution of Partial Differential Equations*

### Inverse Problems
- Vogel, C. R. (2002). *Computational Methods for Inverse Problems*
- Kaipio, J., & Somersalo, E. (2005). *Statistical and Computational Inverse Problems*

### Automatic Differentiation
- Baydin, A. G., et al. (2018). "Automatic differentiation in machine learning: a survey"
- Paszke, A., et al. (2019). "PyTorch: An imperative style, high-performance deep learning library"

## Citation

If you use this code in your research, please cite:

```bibtex
@software{inverse_thermometry,
  title = {Inverse Thermometry: Differentiable Heat Solver for Conductivity Estimation},
  author = {L. Qian and O. Kashurin and V. Makhonin and N. Yavich},
  year = {2025},
  url = {https://github.com/miamvmian/SciML/tree/main//InverseThermometry}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contact

For questions, issues, or contributions:
- **Authors**: [Lujiang Qian](mailto:lujiang.qian@skoltech.ru), [Nikolay Yavich](mailto:n.yavich@skoltech.ru)
- **GitHub Issues**: [Project Issues](https://github.com/miamvmian/SciML/issues)

## Acknowledgments

This implementation was developed as part of a scientific machine learning course project, combining classical numerical methods with modern automatic differentiation frameworks.

---

**Last Updated**: October 2025  
**Version**: 1.0.0  
**Status**: Production Ready 
