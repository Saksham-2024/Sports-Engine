import pytest
import numpy as np
import pandas as pd
from src.core.preprocessing.normalize import BadmintonNormalizer, NormalizeDatasetPipeline
from src.core.utils.configs import PathResolver

class TestBadmintonNormalizer:
    @pytest.fixture
    def normalizer(self):
        # Using standard bounds
        return BadmintonNormalizer(
            x_bounds=(-1, 6.5),
            y_bounds=(-1, 14.5),
            z_bounds=(0, 10)
        )

    @pytest.fixture
    def mock_df(self):
        # Create a dataframe with min, max, mid, NaN, and out-of-bounds values
        return pd.DataFrame({
            'p1_x': [-1.0, 6.5, 2.75, np.nan, 10.0], # min, max, mid, nan, over
            'p1_y': [-1.0, 14.5, 6.75, np.nan, -5.0], # min, max, mid, nan, under
            'p1_z': [0.0, 10.0, 5.0, np.nan, 15.0],    # min, max, mid, nan, over
            'p2_x': [0.0, 0.0, 0.0, 0.0, 0.0],
            'p2_y': [0.0, 0.0, 0.0, 0.0, 0.0],
            'p2_z': [0.0, 0.0, 0.0, 0.0, 0.0],
            'shuttle_x': [-1.0, 6.5, 2.75, np.nan, 10.0],
            'shuttle_y': [-1.0, 14.5, 6.75, np.nan, -5.0],
            'shuttle_z': [0.0, 10.0, 5.0, np.nan, 15.0],
        })

    def test_normalize_single_values(self, normalizer):
        # Test X
        assert normalizer.normalize(-1.0, 'x') == 0.0
        assert normalizer.normalize(6.5, 'x') == 1.0
        assert normalizer.normalize(2.75, 'x') == pytest.approx(0.5)
        
        # Test Y
        assert normalizer.normalize(-1.0, 'y') == 0.0
        assert normalizer.normalize(14.5, 'y') == 1.0
        
        # Test Z
        assert normalizer.normalize(0.0, 'z') == 0.0
        assert normalizer.normalize(10.0, 'z') == 1.0

    def test_normalize_nan_preservation(self, normalizer):
        assert np.isnan(normalizer.normalize(np.nan, 'x'))
        assert np.isnan(normalizer.normalize(np.nan, 'y'))
        assert np.isnan(normalizer.normalize(np.nan, 'z'))

    def test_normalize_clipping(self, normalizer):
        # Value above max should be clipped to 1.0
        assert normalizer.normalize(100.0, 'x') == 1.0
        # Value below min should be clipped to 0.0
        assert normalizer.normalize(-100.0, 'y') == 0.0

    def test_process_dataframe(self, normalizer, mock_df):
        df_norm, bounds = normalizer.process_dataframe(mock_df)
        
        # Verify shapes
        assert df_norm.shape == mock_df.shape
        
        # Verify specific mapped values for shuttle_x
        # -1.0 -> 0.0
        # 6.5  -> 1.0
        # 2.75 -> 0.5
        # nan  -> nan
        # 10.0 -> 1.0 (clipped)
        expected_x = [0.0, 1.0, 0.5, np.nan, 1.0]
        pd.testing.assert_series_equal(
            df_norm['shuttle_x'].reset_index(drop=True), 
            pd.Series(expected_x), 
            check_names=False
        )

    def test_validate_normalized(self, normalizer, mock_df):
        df_norm, _ = normalizer.process_dataframe(mock_df)
        report = normalizer.validate_normalized(df_norm)
        
        assert report['valid'] is True
        assert 'shuttle_x' in report['coord_stats']
        assert report['coord_stats']['shuttle_x']['min'] >= 0.0
        assert report['coord_stats']['shuttle_x']['max'] <= 1.0

    def test_invalid_coord_type(self, normalizer):
        with pytest.raises(ValueError, match="Unknown coord_type"):
            normalizer.normalize(5.0, 'w')

class TestNormalizeDatasetPipeline:
    def test_init(self):
        pipeline = NormalizeDatasetPipeline()
        resolver = PathResolver()
        assert pipeline.input_file == resolver.get_path("tranSPORTmer.data.cleaned_csv")
        assert pipeline.output_file == resolver.get_path("tranSPORTmer.data.normalized_csv")