import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.core.utils.configs import PathResolver
from src.core.preprocessing.clean_anomalies import XYAnomalyHandler, CleanDatasetPipeline

class TestXYAnomalyHandler:
    @pytest.fixture
    def handler(self, **kwargs):
        return XYAnomalyHandler(**kwargs)

    @pytest.fixture
    def df_small_gap(self):
        shuttle_x = np.linspace(-1, 6.5, 10)
        shuttle_y = np.linspace(-1, 14.5, 10)
        shuttle_z = np.linspace(0, 10, 10)
        shuttle_x[4:5] = np.nan
        shuttle_y[3:8] = np.nan
        shuttle_z[5:7] = np.nan
        return pd.DataFrame({"shuttle_x": shuttle_x, "shuttle_y": shuttle_y, "shuttle_z": shuttle_z})

    @pytest.fixture
    def df_medium_gap(self):
        shuttle_x = np.linspace(-1, 6.5, 20)
        shuttle_y = np.linspace(-1, 14.5, 20)
        shuttle_z = np.linspace(0, 10, 20)
        shuttle_x[7:14] = np.nan
        shuttle_y[10:16] = np.nan
        shuttle_z[6:9] = np.nan
        return pd.DataFrame({"shuttle_x": shuttle_x, "shuttle_y": shuttle_y, "shuttle_z": shuttle_z})

    @pytest.fixture
    def df_large_gap(self):
        shuttle_x = np.linspace(-1, 6.5, 30)
        shuttle_y = np.linspace(-1, 14.5, 30)
        shuttle_z = np.linspace(0, 10, 30)
        shuttle_x[15:] = np.nan
        shuttle_y[17:] = np.nan
        shuttle_z[25:] = np.nan
        return pd.DataFrame({"shuttle_x": shuttle_x, "shuttle_y": shuttle_y, "shuttle_z": shuttle_z})
    
    @pytest.fixture
    def segment(self, df_small_gap, df_medium_gap, df_large_gap):
        match_id = np.array([1] * 60)
        segment_idx = np.array([1] * 10 + [2] * 20 + [3] * 30)
        df = pd.concat([df_small_gap, df_medium_gap, df_large_gap], axis=0, ignore_index=True)
        df['match_id'] = match_id
        df['segment_idx'] = segment_idx
        return df

    def test__init__with_no_args(self, handler):
        assert handler.x_min == -1
        assert handler.x_max == 6.5
        assert handler.y_min == -1
        assert handler.y_max == 14.5
        assert handler.interp_threshold == 5
        assert handler.mask_threshold == 10
        assert handler.stats == {
            'segments_processed': 0,
            'anomalies_interpolated': 0,
            'anomalies_masked': 0,
            'segments_split': 0,
            'frames_interpolated': 0,
            'frames_masked': 0,
            'segments_created': 0
        }
    
    def test__init__with_custom_args(self):
        kwargs = {"x_bounds": (-2, 7), "y_bounds": (-2, 15), "interp_threshold": 7, "mask_threshold": 30}
        handler = XYAnomalyHandler(**kwargs)
        assert handler.x_min == -2
        assert handler.x_max == 7
        assert handler.y_min == -2
        assert handler.y_max == 15
        assert handler.interp_threshold == 7
        assert handler.mask_threshold == 30
        assert handler.stats == {
            'segments_processed': 0,
            'anomalies_interpolated': 0,
            'anomalies_masked': 0,
            'segments_split': 0,
            'frames_interpolated': 0,
            'frames_masked': 0,
            'segments_created': 0
        }
    
    def test_is_anomalous(self, handler):
        assert handler.is_anomalous(np.nan, np.nan, np.nan) == True
        assert handler.is_anomalous(np.nan, np.nan, 5.0) == True
        assert handler.is_anomalous(np.nan, 8, np.nan) == True
        assert handler.is_anomalous(np.nan, 8, 5.0) == True
        assert handler.is_anomalous(4, np.nan, np.nan) == True
        assert handler.is_anomalous(4, np.nan, 5.0) == True
        assert handler.is_anomalous(-1, -1, 0) == False
        assert handler.is_anomalous(6.5, 14.5, 15) == False
        assert handler.is_anomalous(-2, 4, 5) == True
        assert handler.is_anomalous(5, -3, 2) == True
        assert handler.is_anomalous(6.6, 2, 16) == True
        assert handler.is_anomalous(0, 19, 1) == True
        
    def test_find_anomally_bands(self, handler):
        x = pd.Series([0, 0, 0, np.nan, np.nan, np.nan, 0, 0])
        y = pd.Series([0, 0, 0, 0, 0, 0, 0, np.nan])
        z = pd.Series([np.nan, 0, 0, 0, 0, 0, 0, 0])
        bands = handler.find_anomaly_bands(x, y, z)
        assert bands == [(0, 1, 1), (3, 6, 3), (7, 8, 1)]
    
    def test_interpolate_band(self, handler):
        x = pd.Series([0, 0, 0, np.nan, np.nan, np.nan, 0, 0], dtype=float)
        y = pd.Series([0, 0, 0, 0, 0, 0, 0, np.nan], dtype=float)
        z = pd.Series([np.nan, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        x_interp, y_interp, z_interp = handler.interpolate_band(x, y, z, 3, 6)
        pd.testing.assert_series_equal(x_interp, pd.Series([0, 0, 0, 0, 0, 0, 0, 0], dtype=float))
        pd.testing.assert_series_equal(y_interp, pd.Series([0, 0, 0, 0, 0, 0, 0, np.nan], dtype=float))
        pd.testing.assert_series_equal(z_interp, pd.Series([np.nan, 0, 0, 0, 0, 0, 0, 0], dtype=float))

    def test_process_dataframe(self, handler, segment):
        # Not having a test_process_segment test here as this test automatically tests for that method as well
        clean_df, stats = handler.process_dataframe(segment)
        interpolated_seg, masked_seg, split_seg = clean_df[clean_df["segment_idx"] == 1], clean_df[clean_df["segment_idx"] == 2], clean_df[clean_df["segment_idx"] == 3]
        assert len(interpolated_seg) == 10
        assert not any(interpolated_seg['shuttle_x'].isna())
        assert not any(interpolated_seg['shuttle_y'].isna())
        assert not any(interpolated_seg['shuttle_z'].isna())
        pd.testing.assert_series_equal(interpolated_seg['shuttle_x'].reset_index(drop=True), pd.Series(np.linspace(-1, 6.5, 10)).reset_index(drop=True), check_names=False)
        pd.testing.assert_series_equal(interpolated_seg['shuttle_y'].reset_index(drop=True), pd.Series(np.linspace(-1, 14.5, 10)).reset_index(drop=True), check_names=False)
        pd.testing.assert_series_equal(interpolated_seg['shuttle_z'].reset_index(drop=True), pd.Series(np.linspace(0, 10, 10)).reset_index(drop=True), check_names=False)

        assert len(masked_seg) == 20
        print("masked_seg:", masked_seg)
        assert all(masked_seg['shuttle_x'][6:16].isna())
        assert all(masked_seg['shuttle_y'][6:16].isna())
        assert all(masked_seg['shuttle_z'][6:16].isna())
        
        assert len(split_seg) == 15
        assert not any(split_seg['shuttle_x'].isna())
        assert not any(split_seg['shuttle_y'].isna())
        assert not any(split_seg['shuttle_z'].isna())
        

class TestCleanDatasetPipeline:

    def test__init__(self):
        resolver = PathResolver()
        pipeline = CleanDatasetPipeline()
        assert resolver.get_path("dataset_creation.transformer_dataset_csv") == pipeline.input_file
        assert resolver.get_path("tranSPORTmer.data.cleaned_csv") == pipeline.output_file
        assert resolver.get_path("tranSPORTmer.data.cleaning_stats") == pipeline.stats_file
    
    def test_run(self):
        ...