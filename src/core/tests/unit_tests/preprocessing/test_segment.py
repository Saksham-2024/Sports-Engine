import numpy as np
import pandas as pd
import pytest
import torch
import json
import os
from unittest.mock import MagicMock, patch
from src.core.preprocessing.segment import FormSegments

class TestFormSegments:

    @pytest.fixture
    def handler(self, tmp_path):
        """Fixture to provide a FormSegments instance with mocked heavy dependencies and temp paths."""
        with patch('src.core.utils.configs.PathResolver.get_path') as mock_get_path:
            # Force all path resolutions to go to the temp directory
            mock_get_path.return_value = str(tmp_path)
            tool = FormSegments()
        
        # Mock the model and device to avoid needing a GPU/Weights during tests
        tool.device = torch.device('cpu')
        tool.sacnn = MagicMock() 
        tool.preprocess = MagicMock()
        return tool
    
    def test__init__(self, handler):
        assert handler.resolver is not None
        assert handler.logger is not None
        assert handler.video_dir is not None
        assert handler.output_dir is not None
        assert handler.model_path is not None

    def test_handle_single_segment_amateur(self, handler):
        """Test that a single giant segment is split into 750-frame chunks."""
        input_seg = [[0, 2000]]
        result = handler._handle_single_segment(input_seg)
        assert len(result) == 3 # 0-750, 750-1500, 1500-2000
        assert result[0] == [0, 750]
        assert result[-1] == [1500, 2000]

    def test_handle_single_segment_pro(self, handler):
        """Test that multiple found segments are NOT split into chunks."""
        input_seg = [[100, 500], [1000, 1500]]
        result = handler._handle_single_segment(input_seg)
        assert len(result) == 2
        assert result == input_seg

    def test_traverse_video_logic(self, handler):
        """Test the state machine that identifies gameplay segments from model labels.
        
        Key details about the pipeline:
        1. SACNN outputs shape [1, 2] (2 classes). torch.argmax(dim=1) picks the class index.
           - Class 0 (non-gameplay): logits like [1.0, 0.0]
           - Class 1 (gameplay):     logits like [0.0, 1.0]
        2. ShotAngleQueue has max_len=5: the first 5 pushes only fill the buffer and
           return (None, None). Transitions are only detected starting from the 6th push.
        3. The queue also applies majority-vote smoothing before emitting transitions,
           so we need sustained runs of each class to trigger condition 1 (start) and 3 (end).
        
        Pattern: 5 zeros (fill queue) + 5 ones (trigger start) + 5 zeros (trigger end) = 15 frames.
        """
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True

        num_frames = 15
        # 15 readable frames + 1 end-of-video signal
        mock_cap.read.side_effect = [
            (True, np.zeros((216, 216, 3), dtype=np.uint8)) for _ in range(num_frames)
        ] + [(False, None)]

        # SACNN mock: must return shape [1, 2] so argmax(dim=1) gives the correct class
        class_0 = torch.tensor([[1.0, 0.0]])  # argmax -> 0
        class_1 = torch.tensor([[0.0, 1.0]])  # argmax -> 1
        handler.sacnn.side_effect = (
            [class_0] * 5 +   # Frames 0-4:  non-gameplay (fills queue)
            [class_1] * 5 +   # Frames 5-9:  gameplay (triggers start after queue delay)
            [class_0] * 5     # Frames 10-14: non-gameplay (triggers end after queue delay)
        )

        res = handler._traverse_video(mock_cap, 0, 100, 1)

        assert len(res) >= 1, f"Expected at least 1 segment, found {len(res)}: {res}"

    def test_write_segments(self, handler, tmp_path):
        """Verify that segments are written to JSON correctly."""
        out_path = str(tmp_path / "test_segments.json")
        segments = [[0, 100], [200, 300]]
        
        handler._write_segments(segments, out_path)
        
        with open(out_path, 'r') as f:
            data = json.load(f)
        
        assert data['segments'] == segments

    def test_segment_single_video_orchestration(self, handler, tmp_path):
        """Test the plumbing of _segment_single_video without running the model."""
        # Setup environment
        handler.video_dir = str(tmp_path)
        handler.output_dir = str(tmp_path)
        
        # Create a dummy video file so os.path.exists passes
        vid_path = os.path.join(handler.video_dir, "match1.mp4")
        with open(vid_path, 'w') as f: f.write("dummy")
        
        # Mock child methods to avoid running heavy logic
        handler._traverse_video = MagicMock(return_value=[[0, 100]])
        handler._handle_single_segment = MagicMock(side_effect=lambda x: x)
        
        # Mock cv2.VideoCapture to avoid needing a real video file
        with patch('cv2.VideoCapture') as mock_vc:
            mock_vc.return_value.get.side_effect = [25.0, 1000] # FPS, FrameCount
            
            segments, out_path = handler._segment_single_video(1)
            
            assert segments == [[0, 100]]
            assert "match1.json" in out_path
            handler._traverse_video.assert_called_once()

    def test_segment_single_video_skip_existing(self, handler, tmp_path):
        """Verify that already segmented videos are skipped."""
        handler.video_dir = str(tmp_path)
        handler.output_dir = str(tmp_path)
        
        # Create dummy video and dummy output JSON
        with open(os.path.join(handler.video_dir, "match1.mp4"), 'w') as f: f.write("d")
        with open(os.path.join(handler.output_dir, "match1.json"), 'w') as f: f.write("d")
        
        segments, out_path = handler._segment_single_video(1)
        assert segments is None
        assert out_path is None
