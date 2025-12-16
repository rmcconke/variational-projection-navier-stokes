"""
Tests for grid_setup module.
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, '../src')
from grid_setup import create_grid_indices


class TestGridIndices:
    """Tests for create_grid_indices function."""
    
    def test_4x3_corners(self):
        """Test corner indices for 4x3 grid."""
        grid = create_grid_indices(4, 3)
        expected = np.array([0, 3, 8, 11])
        np.testing.assert_array_equal(grid['corners'], expected)
    
    def test_4x3_left_boundary(self):
        """Test left boundary indices for 4x3 grid."""
        grid = create_grid_indices(4, 3)
        expected = np.array([4])
        np.testing.assert_array_equal(grid['left_b'], expected)
    
    def test_4x3_right_boundary(self):
        """Test right boundary indices for 4x3 grid."""
        grid = create_grid_indices(4, 3)
        expected = np.array([7])
        np.testing.assert_array_equal(grid['right_b'], expected)
    
    def test_4x3_bottom_boundary(self):
        """Test bottom boundary indices for 4x3 grid."""
        grid = create_grid_indices(4, 3)
        expected = np.array([1, 2])
        np.testing.assert_array_equal(grid['bottom_b'], expected)
    
    def test_4x3_top_boundary(self):
        """Test top boundary indices for 4x3 grid."""
        grid = create_grid_indices(4, 3)
        expected = np.array([9, 10])
        np.testing.assert_array_equal(grid['top_b'], expected)
    
    def test_4x3_interior(self):
        """Test interior indices for 4x3 grid."""
        grid = create_grid_indices(4, 3)
        expected = np.array([5, 6])
        np.testing.assert_array_equal(grid['interior'], expected)
    
    def test_total_points_equals_PQ(self):
        """Test that all index sets together cover all P*Q points."""
        for P, Q in [(4, 3), (10, 10), (50, 50), (5, 8)]:
            grid = create_grid_indices(P, Q)
            all_indices = np.concatenate([
                grid['corners'],
                grid['left_b'],
                grid['right_b'],
                grid['bottom_b'],
                grid['top_b'],
                grid['interior']
            ])
            assert len(all_indices) == P * Q, f"Failed for P={P}, Q={Q}"
    
    def test_no_duplicate_indices(self):
        """Test that no index appears in multiple sets."""
        for P, Q in [(4, 3), (10, 10), (50, 50)]:
            grid = create_grid_indices(P, Q)
            all_indices = np.concatenate([
                grid['corners'],
                grid['left_b'],
                grid['right_b'],
                grid['bottom_b'],
                grid['top_b'],
                grid['interior']
            ])
            assert len(all_indices) == len(np.unique(all_indices)), f"Duplicates found for P={P}, Q={Q}"
    
    def test_indices_in_valid_range(self):
        """Test that all indices are in range [0, P*Q - 1]."""
        for P, Q in [(4, 3), (10, 10), (50, 50)]:
            grid = create_grid_indices(P, Q)
            all_indices = np.concatenate([
                grid['corners'],
                grid['left_b'],
                grid['right_b'],
                grid['bottom_b'],
                grid['top_b'],
                grid['interior']
            ])
            assert np.all(all_indices >= 0), f"Negative index for P={P}, Q={Q}"
            assert np.all(all_indices < P * Q), f"Index out of bounds for P={P}, Q={Q}"
    
    def test_boundary_count(self):
        """Test that boundary point count is 2P + 2Q - 4."""
        for P, Q in [(4, 3), (10, 10), (50, 50), (5, 8)]:
            grid = create_grid_indices(P, Q)
            n_boundary = (len(grid['corners']) + len(grid['left_b']) + 
                         len(grid['right_b']) + len(grid['bottom_b']) + 
                         len(grid['top_b']))
            expected = 2 * P + 2 * Q - 4
            assert n_boundary == expected, f"Failed for P={P}, Q={Q}"
    
    def test_interior_count(self):
        """Test that interior point count is (P-2)*(Q-2)."""
        for P, Q in [(4, 3), (10, 10), (50, 50), (5, 8)]:
            grid = create_grid_indices(P, Q)
            expected = (P - 2) * (Q - 2)
            assert len(grid['interior']) == expected, f"Failed for P={P}, Q={Q}"
    
    def test_corners_are_actual_corners(self):
        """Test that corner indices correspond to grid corners."""
        P, Q = 10, 8
        grid = create_grid_indices(P, Q)
        bl, br, tl, tr = grid['corners']
        
        # Bottom-left: row 0, col 0
        assert bl == 0
        # Bottom-right: row 0, col P-1
        assert br == P - 1
        # Top-left: row Q-1, col 0
        assert tl == (Q - 1) * P
        # Top-right: row Q-1, col P-1
        assert tr == (Q - 1) * P + (P - 1)
    
    def test_left_boundary_column_zero(self):
        """Test that left boundary points are all in column 0."""
        P, Q = 10, 8
        grid = create_grid_indices(P, Q)
        for m in grid['left_b']:
            col = m % P
            assert col == 0, f"Point {m} has column {col}, expected 0"
    
    def test_right_boundary_column_P_minus_1(self):
        """Test that right boundary points are all in column P-1."""
        P, Q = 10, 8
        grid = create_grid_indices(P, Q)
        for m in grid['right_b']:
            col = m % P
            assert col == P - 1, f"Point {m} has column {col}, expected {P-1}"
    
    def test_bottom_boundary_row_zero(self):
        """Test that bottom boundary points are all in row 0."""
        P, Q = 10, 8
        grid = create_grid_indices(P, Q)
        for m in grid['bottom_b']:
            row = m // P
            assert row == 0, f"Point {m} has row {row}, expected 0"
    
    def test_top_boundary_row_Q_minus_1(self):
        """Test that top boundary points are all in row Q-1."""
        P, Q = 10, 8
        grid = create_grid_indices(P, Q)
        for m in grid['top_b']:
            row = m // P
            assert row == Q - 1, f"Point {m} has row {row}, expected {Q-1}"
    
    def test_interior_not_on_boundary(self):
        """Test that interior points are not on any boundary."""
        P, Q = 10, 8
        grid = create_grid_indices(P, Q)
        for m in grid['interior']:
            col = m % P
            row = m // P
            assert col > 0 and col < P - 1, f"Interior point {m} on left/right boundary"
            assert row > 0 and row < Q - 1, f"Interior point {m} on top/bottom boundary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])