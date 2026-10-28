import pytest
import numpy as np
import pandas as pd
from src.core.process.physics import PhysicsProcessor, simulate_trajectory

class TestPhysicsProcessor:
    @pytest.fixture
    def processor(self):
        return PhysicsProcessor()

    def test_simulate_trajectory_basic(self, processor):
        """Test that a trajectory moves in the expected direction."""
        A = np.array([0.0, 0.0, 1.0])
        V0 = np.array([10.0, 0.0, 5.0])
        T = 1.0
        dt = 0.04
        
        traj = simulate_trajectory(A, V0, T, dt, processor.m, processor.A_area, processor.C_d, processor.rho, processor.g_val)
        
        assert traj.shape[0] > 1
        # X should increase (positive V0_x)
        assert traj[-1, 0] > A[0]
        # Z should eventually decrease due to gravity
        # A_z + V0_z*T - 0.5*g*T^2 approx 1 + 5 - 4.9 = 1.1
        # Drag will make it even lower.
        assert traj[-1, 2] < (A[2] + V0[2] * T)

    def test_solve_bvp_identity(self, processor):
        """
        Mathematical Identity Test: 
        If we simulate a trajectory from A to B, the BVP solver should be able 
        to recover the initial velocity V0.
        """
        A = np.array([0.0, 0.0, 1.0])
        V0_true = np.array([15.0, 2.0, 8.0])
        T = 0.5
        dt = 0.04
        
        # 1. Generate ground truth B
        traj_true = simulate_trajectory(A, V0_true, T, dt, processor.m, processor.A_area, processor.C_d, processor.rho, processor.g_val)
        B = traj_true[-1]
        
        # 2. Recover V0 using BVP solver
        V0_recovered, success = processor._solve_bvp(A, B, T, dt)
        
        assert success, "BVP solver failed to converge on a known-good trajectory"
        np.testing.assert_allclose(V0_recovered, V0_true, atol=1e-2)

    def test_solve_bvp_failure(self, processor):
        """Test that impossible trajectories are flagged as failures."""
        A = np.array([0.0, 0.0, 1.0])
        B = np.array([1000.0, 1000.0, 1000.0]) # Physically impossible in 0.1s
        T = 0.1
        dt = 0.04
        
        V0, success = processor._solve_bvp(A, B, T, dt)
        assert not success, "BVP solver should have failed for impossible trajectory"

    def test_camera_ray_projection(self, processor):
        """Test that camera ray intersection is geometrically sound."""
        # Mock camera parameters
        K = np.eye(3)
        R = np.eye(3)
        tvec = np.array([0, 0, 0])
        
        # Pixel (0,0) with Identity K/R/tvec should be a ray [0,0,1]
        C, D = processor._camera_ray(0, 0, K, R, tvec)
        np.testing.assert_array_almost_equal(C, [0, 0, 0])
        np.testing.assert_array_almost_equal(D, [0, 0, 1])
        
        # Intersect with Y-plane at 5.0
        # Since D is [0,0,1], this should FAIL or be parallel 
        # because the ray is moving along Z, not Y.
        res = processor._intersect_ray_y_plane(C, D, 5.0)
        assert res is None

        # Try a ray that actually hits the Y plane (D[1] != 0)
        D_hit = np.array([0, 1, 0]) # Moving directly along Y
        res_hit = processor._intersect_ray_y_plane(C, D_hit, 5.0)
        np.testing.assert_array_almost_equal(res_hit, [0, 5.0, 0])

    def test_pixel_to_court_homography(self, processor):
        """Test that homography transforms points correctly."""
        # Simple scaling homography (2x2)
        H = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # (10, 20) -> (20, 40)
        x, y = processor._pixel_to_court_homography(10, 20, H)
        assert x == 20.0
        assert y == 40.0
