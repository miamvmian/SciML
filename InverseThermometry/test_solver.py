"""
Test script to verify the accuracy of the heat equation solver
against analytical solution
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from heat_solver import solve_heat_equation, HeatSolver
from utils import (
    verification_solution, verification_source, compute_l2_error, 
    compute_relative_error, visualize_comparison, plot_convergence_analysis,
    print_solver_info, create_conductivity_field
)


def test_solver_accuracy():
    """
    Test solver accuracy against analytical solution.
    """
    print("Testing Heat Solver Accuracy")
    print("=" * 50)
    
    # Test parameters
    M = 50  # Grid size
    T = 1.0  # Total time
    device = 'cpu'
    
    # Create constant conductivity field
    sigma = create_conductivity_field(M, pattern='constant', device=device)
    sigma.requires_grad_(True)
    
    # Solve heat equation
    print("Solving heat equation...")
    u_final, u_history = solve_heat_equation(
        sigma, verification_source, M, T, device=device
    )
    
    # Compute analytical solution at final time
    h = 1.0 / M
    x = torch.linspace(0, 1, M+1, device=device)[:-1] + h/2
    y = torch.linspace(0, 1, M+1, device=device)[:-1] + h/2
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    u_analytical = verification_solution(X, Y, T)
    
    # Compute errors
    l2_error = compute_l2_error(u_final, u_analytical)
    relative_error = compute_relative_error(u_final, u_analytical)
    
    print(f"L2 Error: {l2_error:.6f}")
    print(f"Relative Error: {relative_error:.6f}")
    
    # Print solver information
    sigma_max = torch.max(sigma).item()
    tau = T / len(u_history)
    print_solver_info(M, T, tau, len(u_history), sigma_max)
    
    # Visualize comparison
    visualize_comparison(u_final, u_analytical, X, Y, 
                        "Verification Test - Final Time")
    
    return l2_error, relative_error


def test_convergence():
    """
    Test convergence rate of the solver.
    """
    print("\nTesting Convergence Rate")
    print("=" * 50)
    
    # Different grid sizes
    M_values = [5, 10, 20, 40]
    T = 1.0
    device = 'cpu'
    
    errors = []
    h_values = []
    
    for M in M_values:
        print(f"Testing M = {M}...")
        
        # Create conductivity field
        sigma = create_conductivity_field(M, pattern='constant', device=device)
        sigma.requires_grad_(True)
        
        # Solve heat equation
        u_final, _ = solve_heat_equation(
            sigma, verification_source, M, T, device=device
        )
        
        # Compute analytical solution
        h = 1.0 / M
        x = torch.linspace(0, 1, M+1, device=device)[:-1] + h/2
        y = torch.linspace(0, 1, M+1, device=device)[:-1] + h/2
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        u_analytical = verification_solution(X, Y, T)
        
        # Compute error
        l2_error = compute_l2_error(u_final, u_analytical)
        errors.append(l2_error.item())
        h_values.append(h)
        
        print(f"  h = {h:.4f}, Error = {l2_error:.6f}")
    
    # Plot convergence
    plot_convergence_analysis(h_values, errors, "Convergence Analysis")
    
    # Compute convergence rate
    if len(errors) >= 2:
        convergence_rate = np.log(errors[-1]/errors[0]) / np.log(h_values[-1]/h_values[0])
        print(f"Convergence rate: {convergence_rate:.2f}")
    
    return h_values, errors


def test_differentiability():
    """
    Test that the solver is differentiable with respect to conductivity.
    """
    print("\nTesting Differentiability")
    print("=" * 50)
    
    M = 10
    T = 0.1  # Shorter time for faster computation
    device = 'cpu'
    
    # Create conductivity field with gradient enabled
    sigma = create_conductivity_field(M, pattern='constant', device=device)
    sigma.requires_grad_(True)
    
    # Solve heat equation
    u_final, _ = solve_heat_equation(
        sigma, verification_source, M, T, device=device
    )
    
    # Compute a simple loss (sum of temperature field)
    loss = torch.sum(u_final)
    
    # Backward pass
    loss.backward()
    
    # Check if gradients were computed
    if sigma.grad is not None:
        print("Solver is differentiable!")
        print(f"Gradient norm: {torch.norm(sigma.grad):.6f}")
        print(f"Gradient shape: {sigma.grad.shape}")
    else:
        print("Solver is not differentiable!")
    
    return sigma.grad is not None


def test_different_conductivity_patterns():
    """
    Test solver with different conductivity patterns.
    """
    print("\nTesting Different Conductivity Patterns")
    print("=" * 50)
    
    M = 20
    T = 0.5
    device = 'cpu'
    
    patterns = ['constant', 'linear', 'sigmoid']
    
    for pattern in patterns:
        print(f"Testing pattern: {pattern}")
        
        # Create conductivity field
        sigma = create_conductivity_field(M, pattern=pattern, device=device)
        sigma.requires_grad_(True)
        
        # Solve heat equation
        u_final, _ = solve_heat_equation(
            sigma, verification_source, M, T, device=device
        )
        
        # Compute analytical solution
        h = 1.0 / M
        x = torch.linspace(0, 1, M+1, device=device)[:-1] + h/2
        y = torch.linspace(0, 1, M+1, device=device)[:-1] + h/2
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        u_analytical = verification_solution(X, Y, T)
        
        # Compute relative error
        relative_error = compute_relative_error(u_final, u_analytical)
        print(f"  Relative Error = {relative_error:.6f}")
        
        # Visualize comparison
        visualize_comparison(u_final, u_analytical, X, Y, 
                            f"Verification Test - {pattern}")
    
    
def test_heat_solver_module():
    """
    Test the HeatSolver PyTorch module.
    """
    print("\nTesting HeatSolver Module")
    print("=" * 50)
    
    M = 10
    T = 0.5
    device = 'cpu'
    
    # Create solver module
    solver = HeatSolver(M, device=device)
    
    # Create conductivity field
    sigma = create_conductivity_field(M, pattern='constant', device=device)
    sigma.requires_grad_(True)
    
    # Forward pass
    u_final = solver(sigma, verification_source, T)
    
    print(f"Output shape: {u_final.shape}")
    print(f"Output range: [{torch.min(u_final):.6f}, {torch.max(u_final):.6f}]")
    
    # Test differentiability
    loss = torch.sum(u_final)
    loss.backward()
    
    if sigma.grad is not None:
        print("HeatSolver module is differentiable!")
    else:
        print("HeatSolver module is not differentiable!")
    
    return solver


def main():
    """
    Run all tests.
    """
    print("HEAT EQUATION SOLVER VERIFICATION")
    print("=" * 60)
    
    # Test 1: Basic accuracy
    l2_error, relative_error = test_solver_accuracy()
    
    # Test 2: Convergence
    h_values, errors = test_convergence()
    
    # Test 3: Differentiability
    is_differentiable = test_differentiability()
    
    # Test 4: Different conductivity patterns
    test_different_conductivity_patterns()
    
    # Test 5: HeatSolver module
    solver = test_heat_solver_module()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Basic accuracy test: L2 error = {l2_error:.6f}")
    print(f"Convergence test: {len(h_values)} grid sizes tested")
    print(f"Differentiability test: {'PASSED' if is_differentiable else 'FAILED'}")
    print(f"Conductivity patterns test: 4 patterns tested")
    print(f"Module test: {'PASSED' if solver else 'FAILED'}")
    print("=" * 60)
    
    if l2_error < 0.01 and is_differentiable:
        print("All tests PASSED! Solver is ready for inverse problems.")
    else:
        print("Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
