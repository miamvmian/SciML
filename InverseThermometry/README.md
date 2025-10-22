# Differentiable Heat Solver for Thermometry Inverse Problem

This implementation provides a differentiable solver for the 2D heat equation using PyTorch, finite volume method, and explicit Euler time-stepping.

## Files

- `heat_solver.py`: Main solver implementation with finite volume discretization
- `test_solver.py`: Comprehensive test suite for accuracy verification
- `utils.py`: Helper functions for visualization and error computation
- `experiments.ipynb`: Code for all the experiments

## Installation

```bash
pip install torch matplotlib numpy
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
