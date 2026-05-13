"""
Inference Module: Two Modes

Mode 1: Next-Shot Trajectory Prediction
  - Given player movement + shuttle position up to hit frame
  - Predict shuttle trajectory for upcoming shot
  - Output: Landing zone (X, Y, Z profile)
  
Mode 2: Post-Hit Player Movement
  - Given current rally state + just-hit shuttle
  - Predict opponent's position after they prepare response shot
  - Output: Opponent position (X, Y) and movement vector
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import os
import numpy as np
import pickle
import yaml

with open('../../configs/configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

from transformer import create_model


class TranSPORTmerInference:
    """Inference wrapper with denormalization and visualization."""
    
    def __init__(self, checkpoint_path, bounds_path, device='cpu'):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            bounds_path: Path to normalization bounds JSON
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        
        # Load bounds
        with open(bounds_path, 'r') as f:
            bounds = json.load(f)
        
        self.bounds = bounds
        
        # Create model
        config = {
            'input_dim': 3,
            'd_model': 128,
            'num_heads': 8,
            'd_ff': 512,
            'dropout': 0.1,
            'num_agents': 3,
            'seq_len': 125
        }
        
        self.model = create_model(config).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    def denormalize(self, normalized_coords, coord_type):
        """Convert [0, 1] back to court coordinates."""
        bounds = self.bounds[coord_type]
        min_v, max_v = bounds[0], bounds[1]
        
        return normalized_coords * (max_v - min_v) + min_v
    
    def normalize(self, coords, coord_type):
        """Convert court coordinates to [0, 1]."""
        bounds = self.bounds[coord_type]
        min_v, max_v = bounds[0], bounds[1]
        
        return (coords - min_v) / (max_v - min_v)
    
    @torch.no_grad()
    def predict_trajectory(self, X_history, hit_frame_idx, hitter_id):
        """
        Mode 1: Predict shuttle trajectory for next shot.
        
        Args:
            X_history: [T, 3, 3] normalized historical trajectory (past frames)
            hit_frame_idx: int, frame index where shuttle was hit
            hitter_id: int, 0 (player1) or 1 (player2)
            
        Returns:
            {
                'shuttle_trajectory': [future_T, 3] in court coords,
                'landing_zone': (x, y, z_peak),
                'shot_type': str ('smash', 'drop', 'clear', 'drive', 'lift'),
                'confidence': float
            }
        """
        # Ensure input is on device
        X_history = torch.from_numpy(X_history).float().to(self.device)
        
        if X_history.dim() == 3:  # [T, 3, 3]
            X_history = X_history.unsqueeze(0)  # [1, T, 3, 3]
        
        # Predict full trajectory (includes future frames)
        X_pred = self.model(X_history, mask=None)  # [1, T, 3, 3]
        
        X_pred = X_pred.squeeze(0).cpu().numpy()  # [T, 3, 3]
        
        # Extract shuttle trajectory (agent 2)
        shuttle_traj_norm = X_pred[:, 2, :]  # [T, 3]
        
        # Denormalize
        shuttle_traj = shuttle_traj_norm.copy()
        shuttle_traj[:, 0] = self.denormalize(shuttle_traj_norm[:, 0], 'x')
        shuttle_traj[:, 1] = self.denormalize(shuttle_traj_norm[:, 1], 'y')
        shuttle_traj[:, 2] = self.denormalize(shuttle_traj_norm[:, 2], 'z')
        
        # Future frames (after hit)
        future_shuttle = shuttle_traj[hit_frame_idx:]
        
        if len(future_shuttle) == 0:
            future_shuttle = shuttle_traj  # Fallback
        
        # Extract landing zone
        landing_idx = np.argmin(future_shuttle[:, 2])  # Where Z is lowest (lands)
        if future_shuttle[landing_idx, 2] > 0.1:
            landing_idx = -1  # Use final frame if no landing
        
        landing_zone = future_shuttle[landing_idx, :3]
        
        # Classify shot type based on trajectory profile
        shot_type = self._classify_shot(future_shuttle, hitter_id)
        
        # Confidence (based on shuttle visibility in history)
        confidence = self._compute_confidence(X_history.squeeze(0).numpy())
        
        return {
            'shuttle_trajectory': future_shuttle,
            'landing_zone': landing_zone,
            'shot_type': shot_type,
            'confidence': confidence,
            'full_trajectory': shuttle_traj
        }
    
    @torch.no_grad()
    def predict_opponent_movement(self, X_history, hitter_id):
        """
        Mode 2: Predict opponent's positioning after they respond to shot.
        
        Args:
            X_history: [T, 3, 3] normalized historical trajectory
            hitter_id: int, 0 (player1) or 1 (player2) who just hit
            
        Returns:
            {
                'opponent_position': (x, y) in court coords,
                'movement_vector': (dx, dy),
                'time_to_shot_ms': int,
                'positioning_quality': str ('optimal', 'suboptimal', 'defensive')
            }
        """
        X_history = torch.from_numpy(X_history).float().to(self.device)
        
        if X_history.dim() == 3:
            X_history = X_history.unsqueeze(0)
        
        X_pred = self.model(X_history, mask=None)
        X_pred = X_pred.squeeze(0).cpu().numpy()
        
        # Opponent is the other player
        opponent_id = 1 - hitter_id
        opponent_traj_norm = X_pred[:, opponent_id, :]  # [T, 3]
        
        # Denormalize
        opponent_pos = opponent_traj_norm.copy()
        opponent_pos[:, 0] = self.denormalize(opponent_traj_norm[:, 0], 'x')
        opponent_pos[:, 1] = self.denormalize(opponent_traj_norm[:, 1], 'y')
        
        # Current position (start of future prediction)
        current_pos = opponent_pos[-2] if len(opponent_pos) > 1 else opponent_pos[-1]
        
        # Future position (when they should hit)
        future_pos = opponent_pos[-1]
        
        # Movement vector
        movement_vector = future_pos[:2] - current_pos[:2]
        
        # Evaluate positioning quality
        positioning = self._evaluate_positioning(
            opponent_pos, hitter_id, X_history.squeeze(0).numpy()
        )
        
        # Time estimate (frames to response = ~0.5-1 second = 12-25 frames at 25fps)
        time_to_shot_ms = int(20 * 40)  # ~800ms (empirical average)
        
        return {
            'opponent_position': future_pos[:2],
            'current_position': current_pos[:2],
            'movement_vector': movement_vector,
            'time_to_shot_ms': time_to_shot_ms,
            'positioning_quality': positioning
        }
    
    def _classify_shot(self, shuttle_traj, hitter_id):
        """Infer shot type from shuttle trajectory profile."""
        if len(shuttle_traj) < 2:
            return 'unknown'
        
        # Trajectory characteristics
        max_height = shuttle_traj[:, 2].max()
        height_at_end = shuttle_traj[-1, 2]
        horizontal_dist = np.linalg.norm(shuttle_traj[-1, :2] - shuttle_traj[0, :2])
        descent_rate = (shuttle_traj[0, 2] - shuttle_traj[-1, 2]) / max(len(shuttle_traj), 1)
        
        # Simple heuristics
        if max_height > 1.5 and descent_rate < 0.02:
            return 'clear'  # High arc, slow descent
        elif max_height > 0.8 and descent_rate > 0.05:
            return 'smash'  # High then drops steeply
        elif max_height < 0.5 and horizontal_dist < 3:
            return 'drop'   # Shallow, short
        elif max_height < 0.3:
            return 'drive'  # Flat trajectory
        else:
            return 'lift'   # Defensive, medium height
    
    def _compute_confidence(self, X_hist):
        """Confidence based on shuttle observability in history."""
        shuttle_pos = X_hist[:, 2, :]  # [T, 3]
        valid_frames = (~np.isnan(shuttle_pos[:, 0])).sum()
        return valid_frames / len(shuttle_pos)
    
    def _evaluate_positioning(self, opponent_traj, hitter_id, X_hist):
        """Evaluate if opponent's positioning is optimal."""
        # Optimal position for badminton: near court center
        court_center = np.array([2.59, 6.7])  # Court center [x, y]
        
        opponent_final = opponent_traj[-1, :2]
        distance_to_center = np.linalg.norm(opponent_final - court_center)
        
        if distance_to_center < 1.0:
            return 'optimal'
        elif distance_to_center < 2.0:
            return 'suboptimal'
        else:
            return 'defensive'


def load_test_sequence(segment_path, window_idx=0):
    """
    Load a test window from dataset.
    
    Args:
        segment_path: Path to test_windows.pkl
        window_idx: Index of window to load
        
    Returns:
        numpy array [T, 3, 3]
    """
    with open(segment_path, 'rb') as f:
        data = pickle.load(f)
    
    trajectory = data['trajectories'][window_idx]  # numpy [T, 3, 3]
    return trajectory


def demo():
    """Demonstration of inference modes."""
    
    print("=" * 80)
    print("TRANSPORTER INFERENCE DEMO")
    print("=" * 80)
    
    # Paths
    checkpoint = Path(os.path.join(configs['global']['project_root'], configs['tranSPORTmer']['weights']['best_weights']))
    bounds = Path(os.path.join(configs['global']['project_root'], configs['tranSPORTmer']['data']['normalization_stats']))
    test_data = Path(os.path.join(configs['global']['project_root'], configs['tranSPORTmer']['data']['training_data_dir'], 'test_windows.pkl'))
    
    if not checkpoint.exists():
        print(f"✗ Checkpoint not found: {checkpoint}")
        print("  Run train.py first")
        return
    
    # Initialize inference
    print(f"\n[1/3] Loading model...")
    inference = TranSPORTmerInference(
        str(checkpoint),
        str(bounds),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"      ✓ Model loaded")
    
    # Load test sequence
    print(f"\n[2/3] Loading test sequence...")
    X_test = load_test_sequence(test_data, window_idx=0)
    print(f"      Shape: {X_test.shape}")
    print(f"      ✓ Test sequence loaded")
    
    # Demo: Next-shot trajectory
    print(f"\n[3/3] Running inference demos...")
    
    print(f"\n--- MODE 1: Next-Shot Trajectory ---")
    hit_frame = 60  # Assume hit at frame 60
    hitter = 0  # Player 1 hit
    
    result_shot = inference.predict_trajectory(X_test, hit_frame, hitter)
    
    print(f"Shot type: {result_shot['shot_type']}")
    print(f"Landing zone: {result_shot['landing_zone']}")
    print(f"Confidence: {result_shot['confidence']:.2%}")
    
    print(f"\n--- MODE 2: Post-Hit Player Movement ---")
    result_movement = inference.predict_opponent_movement(X_test, hitter)
    
    print(f"Opponent position: {result_movement['opponent_position']}")
    print(f"Movement vector: {result_movement['movement_vector']}")
    print(f"Positioning quality: {result_movement['positioning_quality']}")
    print(f"Est. time to response: {result_movement['time_to_shot_ms']}ms")
    
    print("\n" + "=" * 80)
    print("✓ INFERENCE DEMO COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    demo()
