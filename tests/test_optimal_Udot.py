"""
Tests for optimal_Udot module.
"""
import sys
sys.path.insert(0, '../src')
import config
import numpy as np
import pytest

from grid_setup import construct_grid
from construct_A import construct_A
from optimal_Udot import (
    compute_projection_matrices,
    compute_optimal_Udot,
    compute_M_neg_sqrt
)


class TestMNegSqrt:
    """Tests for compute_M_neg_sqrt function."""
    grid = construct_grid(P=10, Q=8, W=1.0, H=1.0)
    def test_shape(self):
        """Test M_neg_sqrt has correct shape."""
        M_neg_sqrt = compute_M_neg_sqrt(self.grid, rho=1.0)
        assert M_neg_sqrt.shape == (2 * 10 * 8,)
    
    def test_value(self):
        """Test M_neg_sqrt has correct value."""
        rho = 1000.0
        dm = rho * self.grid.dx * self.grid.dy
        expected = dm ** (-0.5)
        
        M_neg_sqrt = compute_M_neg_sqrt(self.grid, rho=rho)
        
        assert np.allclose(M_neg_sqrt, expected)
    
    def test_uniform(self):
        """Test all elements are equal (uniform grid)."""
        M_neg_sqrt = compute_M_neg_sqrt(self.grid, rho=1.0)
        assert np.allclose(M_neg_sqrt, M_neg_sqrt[0])


class TestProjectionMatrices:
    """Tests for compute_projection_matrices function."""
    
    grid = construct_grid(P=5, Q=5, W=1.0, H=1.0)
    def test_shapes(self):
        """Test P and N have correct shapes."""
        A = construct_A(self.grid)
        M_neg_sqrt = compute_M_neg_sqrt(self.grid, rho=1.0)
        
        P, N = compute_projection_matrices(A, M_neg_sqrt)
        
        n_dof = 2 * self.grid.P * self.grid.Q
        assert P.shape == (n_dof, n_dof)
        assert N.shape == (n_dof, n_dof)
    
    def test_P_is_projector(self):
        """Test that P is a projection matrix (P^2 = P)."""
        A = construct_A(self.grid)
        M_neg_sqrt = compute_M_neg_sqrt(self.grid, rho=1.0)
        
        P, N = compute_projection_matrices(A, M_neg_sqrt,sparse=False)
        
        P_squared = P @ P
        assert np.allclose(P_squared, P, atol=1e-10), "P should satisfy P^2 = P"
    
    def test_N_is_projector(self):
        """Test that N is a projection matrix (N^2 = N)."""
        A = construct_A(self.grid)
        M_neg_sqrt = compute_M_neg_sqrt(self.grid, rho=1.0)
        
        P, N = compute_projection_matrices(A, M_neg_sqrt,sparse=False)
        
        N_squared = N @ N
        assert np.allclose(N_squared, N, atol=1e-10), "N should satisfy N^2 = N"
    
    def test_P_plus_N_is_identity(self):
        """Test that P + N = I."""
        A = construct_A(self.grid)
        M_neg_sqrt = compute_M_neg_sqrt(self.grid, rho=1.0)
        
        P, N = compute_projection_matrices(A, M_neg_sqrt,sparse=False)
        
        n_dof = 2 * self.grid.P * self.grid.Q
        assert np.allclose(P + N, np.eye(n_dof), atol=1e-10), "P + N should equal I"
    
    def test_P_N_orthogonal(self):
        """Test that P @ N = N @ P = 0."""
        A = construct_A(self.grid)
        M_neg_sqrt = compute_M_neg_sqrt(self.grid, rho=1.0)
        
        P, N = compute_projection_matrices(A, M_neg_sqrt,sparse=False)
        
        assert np.allclose(P @ N, 0, atol=1e-10), "P @ N should be zero"
        assert np.allclose(N @ P, 0, atol=1e-10), "N @ P should be zero"
    
    def test_P_is_symmetric(self):
        """Test that P is symmetric."""
        A = construct_A(self.grid)
        M_neg_sqrt = compute_M_neg_sqrt(self.grid, rho=1.0)
        
        P, N = compute_projection_matrices(A, M_neg_sqrt,sparse=False)
        
        assert np.allclose(P, P.T, atol=1e-10), "P should be symmetric"
    
    def test_N_is_symmetric(self):
        """Test that N is symmetric."""
        A = construct_A(self.grid)
        M_neg_sqrt = compute_M_neg_sqrt(self.grid, rho=1.0)
        
        P, N = compute_projection_matrices(A, M_neg_sqrt,sparse=False)
        
        assert np.allclose(N, N.T, atol=1e-10), "N should be symmetric"


class TestOptimalUdot:
    """Tests for compute_optimal_Udot function."""
    grid = construct_grid(P=5, Q=5, W=1.0, H=1.0)
    def test_output_shapes(self):
        """Test U_dot has correct shape and S_star is scalar."""
        
        A = construct_A(self.grid)
        M_neg_sqrt = compute_M_neg_sqrt(self.grid, rho=1.0)
        P, N = compute_projection_matrices(A, M_neg_sqrt)
        
        C = np.random.rand(2 * self.grid.P * self.grid.Q)
        U_dot, S_star = compute_optimal_Udot(M_neg_sqrt, C, P, N)
        
        assert U_dot.shape == (2 * self.grid.P * self.grid.Q,)
    
    def test_zero_C_gives_zero_Udot(self):
        """Test that C=0 gives U_dot=0."""
        A = construct_A(self.grid)
        M_neg_sqrt = compute_M_neg_sqrt(self.grid, rho=1.0)
        P, N = compute_projection_matrices(A, M_neg_sqrt)
        
        C = np.zeros(2 * self.grid.P * self.grid.Q)
        U_dot, S_star = compute_optimal_Udot(M_neg_sqrt, C, P, N)
        
        assert np.allclose(U_dot, 0.0)
        assert np.isclose(S_star, 0.0)
    
    def test_S_star_non_negative(self):
        """Test that Appellian is non-negative."""
        A = construct_A(self.grid)
        M_neg_sqrt = compute_M_neg_sqrt(self.grid, rho=1.0)
        P, N = compute_projection_matrices(A, M_neg_sqrt)
        
        # Test with random C vectors
        for _ in range(10):
            C = np.random.randn(2 * self.grid.P * self.grid.Q)
            U_dot, S_star = compute_optimal_Udot(M_neg_sqrt, C, P, N)
            assert S_star >= -1e-10, f"S_star should be non-negative, got {S_star}"
    
    def test_Udot_is_divergence_free(self):
        """Test that the optimal U_dot satisfies A @ U_dot ≈ 0."""
        A = construct_A(self.grid)
        M_neg_sqrt = compute_M_neg_sqrt(self.grid, rho=1.0)
        P, N = compute_projection_matrices(A, M_neg_sqrt)
        
        C = np.random.randn(2 * self.grid.P * self.grid.Q)
        U_dot, S_star = compute_optimal_Udot(M_neg_sqrt, C, P, N)
        
        # A @ U_dot should be approximately zero
        divergence = A @ U_dot
        assert np.allclose(divergence, 0, atol=1e-10), \
            f"U_dot should be divergence-free, max div = {np.max(np.abs(divergence))}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])