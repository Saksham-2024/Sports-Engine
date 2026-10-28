
import math
import pytest
import logging
import pathlib
import numpy as np
import pandas as pd
from src.core.process.combine import Executer
from src.core.utils.configs import PathResolver
from src.core.utils.logger import get_logger

class TestExecuter:

    @pytest.fixture
    def executer(self):
        return Executer()
    
    def straight_line(self, x0, y0, slope, c, steps):
        x_values = np.linspace(x0, x0 + steps, steps + 1)
        y_values = slope * x_values + c
        return x_values, y_values
    
    def mock_linear_trajectories_for_hit_detection(self, theta):
        path1_x, path1_y = self.straight_line(0, 0, 1, 0, 50)
        m1 = (path1_y[-1] - path1_y[0]) / (path1_x[-1] - path1_x[0])
        tan_value = abs(math.tan(theta))
        m2 = (tan_value + m1) / (1 - m1 * tan_value)
        path2_x, path2_y = self.straight_line(path1_x[-1], path1_y[-1], m2, 0, 51)
        path2_x, path2_y = path2_x * 2, path2_y * 2 
        shuttle_x = np.concatenate([path1_x, path2_x[1:]])
        shuttle_y = np.concatenate([path1_y, path2_y[1:]])
        return shuttle_x, shuttle_y
    
    def test__init__(self, executer):
        resolver = PathResolver()
        assert isinstance(executer.resolver, PathResolver)
        assert isinstance(executer.logger, logging.Logger)
        assert isinstance(executer.segments_dir, pathlib.Path)

    def test__natural_sort_key(self, executer):
        mock_data = [f"match{i}" for i in range(10, -1, -1)]
        expected_sorted_mock_data = [f"match{i}" for i in range(11)]
        matches = sorted(mock_data, key=executer._natural_sort_key)
        assert matches == expected_sorted_mock_data

    def test__detect_hit_frames_basic(self, executer):
        shuttle_x, shuttle_y = self.mock_linear_trajectories_for_hit_detection(math.degrees(45))
        df = pd.DataFrame({
            "shuttle_x": shuttle_x,
            "shuttle_y": shuttle_y,
            "shuttle_vis": [1.0] * len(shuttle_x),
        })
        hit_frames = executer._detect_hit_frames(df)
        assert hit_frames[51] == True

    def test__detect_hit_frames_low_vis(self, executer):
        shuttle_x, shuttle_y = self.mock_linear_trajectories_for_hit_detection(math.degrees(45))
        df = pd.DataFrame({
            "shuttle_x": shuttle_x,
            "shuttle_y": shuttle_y,
            "shuttle_vis": [0.0] * len(shuttle_x),
        })
        hit_frames = executer._detect_hit_frames(df)
        assert all(hit_frames == False)

    def test__detect_hit_frames_too_short(self, executer):
        """Verify that DataFrames < 4 frames do not crash and return all-False."""
        df = pd.DataFrame({
            "shuttle_x": [1.0, 2.0],
            "shuttle_y": [1.0, 2.0],
            "shuttle_vis": [1.0, 1.0],
        })
        hit_frames = executer._detect_hit_frames(df)
        assert len(hit_frames) == 2
        assert all(hit_frames == False)

    def test__detect_hit_frames_with_nans(self, executer):
        """Verify that NaNs in coordinates do not cause a crash."""
        shuttle_x, shuttle_y = self.mock_linear_trajectories_for_hit_detection(math.degrees(45))
        df = pd.DataFrame({
            "shuttle_x": shuttle_x,
            "shuttle_y": shuttle_y,
            "shuttle_vis": [1.0] * len(shuttle_x),
        })
        # Inject NaNs
        df.loc[10:15, "shuttle_x"] = np.nan
        
        # Should not crash
        hit_frames = executer._detect_hit_frames(df)
        assert isinstance(hit_frames, pd.Series)

    def test__process_match_structure(self, executer, match_id="match4"):
        res = executer._process_match(match_id)
        if res is not None:
            assert isinstance(res, list)
            assert "match_id" in res[0].columns 
            assert "segment_idx" in res[0].columns
            assert "is_hit_frame" in res[0].columns
