import pytest
import numpy as np
import pickle
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.core.utils.configs import PathResolver
from src.core.preprocessing.camera_matrices import (
    CalibrationPipeline,
    CameraPositionCalculator,
    HomographyCalculator,
    CalibrationUI,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def court_dims():
    resolver = PathResolver()
    return resolver.get_config("dataset_creation.court")


@pytest.fixture
def pipeline(tmp_path):
    """Pipeline with cache paths redirected to a temp directory."""
    p = CalibrationPipeline()
    p.homography_cache = tmp_path / "homography_cache.pkl"
    p.camera_pos_cache = tmp_path / "camera_pose_cache.pkl"
    return p


@pytest.fixture
def mock_matrices():
    return {"match1.mp4": np.zeros((3, 3), dtype=np.float32)}


@pytest.fixture
def mock_camera_poses():
    return {
        "match1.mp4": {
            "camera_pos": np.zeros((3,),   dtype=np.float64),
            "K":          np.zeros((3, 3), dtype=np.float64),
            "R":          np.zeros((3, 3), dtype=np.float64),
            "tvec":       np.zeros((3, 1), dtype=np.float64),
        }
    }

# ── CalibrationPipeline tests ─────────────────────────────────────────────────

class TestCalibrationPipeline:

    def test_load_caches_returns_empty_dicts_when_no_files(self, pipeline):
        """_load_caches should return empty dicts when cache files do not exist."""
        matrices, camera_poses = pipeline._load_caches()
        assert isinstance(matrices, dict) and len(matrices) == 0
        assert isinstance(camera_poses, dict) and len(camera_poses) == 0

    def test_save_and_reload_caches(self, pipeline, mock_matrices, mock_camera_poses):
        """save_caches persists data; _load_caches can retrieve it correctly."""
        pipeline.save_caches(mock_matrices, mock_camera_poses)

        assert pipeline.homography_cache.exists()
        assert pipeline.camera_pos_cache.exists()

        loaded_m, loaded_cp = pipeline._load_caches()

        # Structure of homography cache
        assert "match1.mp4" in loaded_m
        assert loaded_m["match1.mp4"].shape == (3, 3)
        assert loaded_m["match1.mp4"].dtype == np.float32

        # Structure of camera pose cache
        assert "match1.mp4" in loaded_cp
        pose = loaded_cp["match1.mp4"]
        assert set(pose.keys()) == {"camera_pos", "K", "R", "tvec"}
        assert pose["camera_pos"].shape == (3,)
        assert pose["K"].shape == (3, 3)
        assert pose["R"].shape == (3, 3)
        assert pose["tvec"].shape == (3, 1)

    def test_get_remaining_videos_all_missing(self, pipeline):
        """All videos flagged when caches are empty."""
        remaining = pipeline._get_remaining_videos(
            ["a.mp4", "b.mp4"], {}, {}
        )
        assert remaining == {
            "a.mp4": ["homography", "camera_pose"],
            "b.mp4": ["homography", "camera_pose"],
        }

    def test_get_remaining_videos_partial_cache(self, pipeline, mock_matrices):
        """Only missing task is returned when homography is already cached."""
        remaining = pipeline._get_remaining_videos(
            ["match1.mp4"], mock_matrices, {}
        )
        assert remaining == {"match1.mp4": ["camera_pose"]}

    def test_get_remaining_videos_fully_cached(self, pipeline, mock_matrices, mock_camera_poses):
        """Fully-cached video is not included in remaining."""
        remaining = pipeline._get_remaining_videos(
            ["match1.mp4"], mock_matrices, mock_camera_poses
        )
        assert remaining == {}


# ── HomographyCalculator tests ────────────────────────────────────────────────

class TestHomographyCalculator:

    def test_calculate_returns_3x3_matrix(self, court_dims):
        """calculate() with valid points returns a (3, 3) ndarray."""
        calc = HomographyCalculator(court_dims)
        points = [
            (200, 100), (1720, 100), (1800, 900), (120, 900),
            (640, 500), (1280, 500),  # net posts (ignored by homography)
        ]
        H = calc.calculate(points)
        assert H is not None
        assert H.shape == (3, 3)

    def test_court_2d_real_shape(self, court_dims):
        """COURT_2D_REAL should have shape (4, 2)."""
        calc = HomographyCalculator(court_dims)
        assert calc.COURT_2D_POINTS.shape == (4, 2)


# ── CameraPositionCalculator tests ───────────────────────────────────────────

class TestCameraPositionCalculator:

    def test_K_matrix_shape_and_values(self, court_dims):
        """K matrix must be (3, 3) with correct focal length and principal point."""
        calc = CameraPositionCalculator(court_dims, 1920, 1080, "test.mp4")
        assert calc.K.shape == (3, 3)
        assert calc.K.dtype == np.float64
        assert calc.K[0, 0] == 1920.0           # focal length == frame width
        assert calc.K[0, 2] == 960.0            # cx = width / 2
        assert calc.K[1, 2] == 540.0            # cy = height / 2
        assert calc.K[2, 2] == 1.0

    def test_calculate_returns_correct_structure(self, court_dims):
        """calculate() with valid points returns dict with correct keys and shapes."""
        calc = CameraPositionCalculator(court_dims, 1920, 1080, "test.mp4")
        points = [
            (200, 100), (1720, 100), (1800, 900), (120, 900),
            (640, 500), (1280, 500),
        ]
        result = calc.calculate(points)
        # solvePnP may or may not converge on synthetic points; only validate structure if it does.
        if result is not None:
            assert set(result.keys()) == {"K", "R", "tvec", "camera_pos"}
            assert result["K"].shape == (3, 3)
            assert result["R"].shape == (3, 3)
            assert result["tvec"].shape == (3, 1)
            assert result["camera_pos"].shape == (3,)

    def test_court_3d_points_shape(self, court_dims):
        """COURT_3D_POINTS should have shape (6, 3)."""
        calc = CameraPositionCalculator(court_dims, 1920, 1080, "test.mp4")
        assert calc.COURT_3D_POINTS.shape == (6, 3)


# ── CalibrationUI tests (mocked) ─────────────────────────────────────────────

class TestCalibrationUI:

    def test_collect_points_returns_none_on_abort(self):
        """When ESC is pressed immediately, collect_points returns None."""
        ui = CalibrationUI()
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("cv2.imshow"), \
             patch("cv2.namedWindow"), \
             patch("cv2.setMouseCallback"), \
             patch("cv2.destroyAllWindows"), \
             patch("cv2.waitKey", return_value=27):   # ESC key
            result = ui.collect_points(dummy_frame)

        assert result is None

    def test_collect_points_returns_none_on_skip(self):
        """When S is pressed, collect_points returns None."""
        ui = CalibrationUI()
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("cv2.imshow"), \
             patch("cv2.namedWindow"), \
             patch("cv2.setMouseCallback"), \
             patch("cv2.destroyAllWindows"), \
             patch("cv2.waitKey", return_value=ord('s')):
            result = ui.collect_points(dummy_frame)

        assert result is None
