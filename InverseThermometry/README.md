# Differentiable Heat Solver for Thermometry Inverse Problem

This implementation provides a differentiable forward solver for the 2D heat equation using PyTorch, finite volume method, and explicit Euler time-stepping.

## Files

- `heat_solver.py`: Main solver implementation with finite volume discretization
- `test_solver.py`: Comprehensive test suite for accuracy verification
- `utils.py`: Helper functions for visualization and error computation

## Installation

```bash
pip install torch matplotlib numpy
```

## Usage

### Basic Usage

```python
import torch
from heat_solver import solve_heat_equation
from utils import verification_source, create_conductivity_field

# Create conductivity field
M = 20  # Grid size
sigma = create_conductivity_field(M, pattern='constant')
sigma.requires_grad_(True)

# Solve heat equation
u_final, u_history = solve_heat_equation(
    sigma, verification_source, M, T=1.0
)

print(f"Final temperature range: [{u_final.min():.4f}, {u_final.max():.4f}]")
```

### Using the PyTorch Module

```python
from heat_solver import HeatSolver

# Create solver
solver = HeatSolver(M=20)

# Forward pass
u_final = solver(sigma, verification_source, T=1.0)

# Compute loss and backpropagate
loss = torch.sum(u_final)
loss.backward()
print(f"Gradient norm: {torch.norm(sigma.grad):.6f}")
```

## Running Tests

```bash
cd InverseThermometry
python test_solver.py
```

The test suite includes:
1. **Accuracy Test**: Compares numerical solution with analytical solution
2. **Convergence Test**: Verifies convergence rate with grid refinement
3. **Differentiability Test**: Ensures gradients can be computed
4. **Pattern Test**: Tests different conductivity field patterns
5. **Module Test**: Tests the PyTorch module interface

## Key Features

### Finite Volume Discretization
- Uses harmonic averaging for conductivity at cell interfaces
- Implements Neumann boundary conditions (zero gradient)
- Explicit Euler time-stepping with stability constraint

### Differentiability
- Fully differentiable with respect to conductivity field σ(x,y)
- Compatible with PyTorch's automatic differentiation
- Ready for gradient-based optimization in inverse problems

### Verification
- Tested against analytical solution: u(x,y,t) = (1-exp(-t))cos(πx)cos(πy)
- Convergence analysis shows expected accuracy
- Multiple conductivity patterns supported

## Mathematical Details

### Heat Equation
```
∂u/∂t = ∂/∂x(σ ∂u/∂x) + ∂/∂y(σ ∂u/∂y) + f(x,y,t)
```

### Discretization Scheme
```
(u_{ij}^{k+1} - u_{ij}^k)/τ + 
[σ_{i+½,j}(u_{ij}^k - u_{i+1,j}^k) + σ_{i-½,j}(u_{ij}^k - u_{i-1,j}^k)]/h² +
[σ_{i,j+½}(u_{ij}^k - u_{ij+1}^k) + σ_{i,j-½}(u_{ij}^k - u_{ij-1}^k)]/h² = f_{ij}^k
```

### Stability Condition
```
τ ≤ h²/(4σ_max)
```

## Expected Results

- **L2 Error**: < 0.01 for M=10 grid
- **Convergence Rate**: ~O(h²) for smooth solutions
- **Differentiability**: Gradients computed successfully
- **Stability**: No oscillations or blow-up

## Next Steps

This solver is ready for Part 2 of the project: using it in an inverse problem to estimate thermal conductivity from temperature measurements.
