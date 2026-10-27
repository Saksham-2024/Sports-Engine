import pytest
import numpy as np
import pandas as pd
import torch
import os
import pickle
from src.core.preprocessing.training_windows import TrainingWindowGenerator, FormTrainingWindows

class TestTrainingWindowGenerator:
    @pytest.fixture
    def generator(self):
        # Small window sizes for easy testing
        return TrainingWindowGenerator(window_size=10, target_size=5, stride=5)

    @pytest.fixture
    def mock_segment(self):
        """Create a segment with 20 frames, containing some NaNs."""
        df = pd.DataFrame({
            'frame': np.arange(20),
            'p1_x': np.random.rand(20), 'p1_y': np.random.rand(20), 'p1_z': np.random.rand(20),
            'p2_x': np.random.rand(20), 'p2_y': np.random.rand(20), 'p2_z': np.random.rand(20),
            'shuttle_x': np.random.rand(20), 'shuttle_y': np.random.rand(20), 'shuttle_z': np.random.rand(20),
        })
        # Inject some NaNs to test masking
        df.loc[2:4, 'shuttle_x'] = np.nan
        return df

    def test_create_segment_windows_shapes(self, generator, mock_segment):
        windows = generator.create_segment_windows(mock_segment, segment_id=1, match_id=101)
        
        assert len(windows) > 0
        window = windows[0]
        
        # Check input shapes: (window_size, 3 agents, 3 coords)
        assert window['input_traj'].shape == (10, 3, 3)
        assert window['input_mask'].shape == (10, 3, 3)
        
        # Check target shapes: (target_size, 3 agents, 3 coords)
        assert window['target_traj'].shape == (5, 3, 3)
        assert window['target_mask'].shape == (5, 3, 3)

    def test_nan_handling_and_masking(self, generator):
        # Create a tiny segment with a NaN
        df = pd.DataFrame({
            'frame': [0, 1, 2, 3, 4, 5],
            'p1_x': [0] * 6, 'p1_y': [0] * 6, 'p1_z': [0] * 6,
            'p2_x': [0] * 6, 'p2_y': [0] * 6, 'p2_z': [0] * 6,
            'shuttle_x': [0, np.nan, 0, 0, 0, 0], # NaN at index 1
            'shuttle_y': [0] * 6, 'shuttle_z': [0] * 6,
        })
        # window_size=2, target_size=1
        gen = TrainingWindowGenerator(window_size=2, target_size=1, stride=1)
        windows = gen.create_segment_windows(df, 1, 101)
        
        # Window 0: frame [0, 1]. Index 1 is NaN.
        # Input traj shape (2, 3, 3). Shuttle is index 2 in the agent dimension.
        # Traj[1, 2, 0] should be -inf
        assert windows[0]['input_traj'][1, 2, 0] == -np.inf
        # Mask[1, 2, 0] should be 1.0
        assert windows[0]['input_mask'][1, 2, 0] == 1.0

    def test_padding_logic(self, generator):
        # Create a segment that is just long enough to have a target, 
        # but short enough to require padding.
        # window=10, target=5. Let's give it 12 frames.
        df = pd.DataFrame({
            'frame': np.arange(12),
            'p1_x': [0] * 12, 'p1_y': [0] * 12, 'p1_z': [0] * 12,
            'p2_x': [0] * 12, 'p2_y': [0] * 12, 'p2_z': [0] * 12,
            'shuttle_x': [0] * 12, 'shuttle_y': [0] * 12, 'shuttle_z': [0] * 12,
        })
        
        windows = generator.create_segment_windows(df, 1, 101)
        
        # Now it should create at least one window
        assert len(windows) > 0
        
        # Verify that target_traj is padded because we only have 2 frames of target
        # (Frames 10 and 11), but target_size is 5.
        target_traj = windows[0]['target_traj']
        assert target_traj.shape == (5, 3, 3)
        # Last 3 frames of target should be padding (-inf)
        assert np.all(target_traj[2:] == -np.inf)

    def test_split_train_val_test_leakage(self, generator):
        # Create windows from 10 different matches
        windows = []
        for m_id in range(10):
            for w_id in range(5):
                windows.append({'match_id': m_id, 'input_traj': np.zeros((10,3,3))})
        
        train, val, test, info = generator.split_train_val_test(windows)
        
        # Extract match IDs from each split
        train_matches = set(w['match_id'] for w in train)
        val_matches = set(w['match_id'] for w in val)
        test_matches = set(w['match_id'] for w in test)
        
        # Verify that match sets are disjoint (NO LEAKAGE)
        assert train_matches.isdisjoint(val_matches)
        assert train_matches.isdisjoint(test_matches)
        assert val_matches.isdisjoint(test_matches)

class TestFormTrainingWindows:
    def test_save_windows(self, tmp_path):
        pipeline = FormTrainingWindows()
        out_path = str(tmp_path / "train.pkl")
        
        # Mock windows
        windows = [{
            'input_traj': np.random.rand(10, 3, 3),
            'input_mask': np.random.randint(0, 2, (10, 3, 3)),
            'target_traj': np.random.rand(5, 3, 3),
            'target_mask': np.random.randint(0, 2, (5, 3, 3)),
            'match_id': 1, 'segment_id': 1, 'window_start_frame': 0, 'window_end_frame': 9, 'length': 10
        }]
        
        pipeline.save_windows(windows, out_path)
        
        with open(out_path, 'rb') as f:
            data = pickle.load(f)
            
        assert 'input_trajectories' in data
        assert data['input_trajectories'].shape == (1, 10, 3, 3)
        assert data['metadata'][0]['match_id'] == 1