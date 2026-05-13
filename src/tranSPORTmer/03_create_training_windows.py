import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
from sklearn.model_selection import train_test_split
import pickle
import yaml

with open('../../configs/configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

class TrainingWindowGenerator:
    def __init__(self, window_size=125, target_size=35, stride=25):
        self.window_size = window_size
        self.target_size = target_size
        self.stride = stride
        
        self.stats = {
            'total_segments': 0,
            'total_frames': 0,
            'windows_created': 0,
            'windows_padded': 0,
            'train_windows': 0,
            'val_windows': 0,
            'test_windows': 0
        }
    
    def create_segment_windows(self, segment_df, segment_id, match_id):
        windows = []
        segment_df = segment_df.sort_values('frame').reset_index(drop=True)
        num_frames = len(segment_df)
        
        def extract_and_pad(df, req_len):
            if len(df) == 0:
                traj = np.full((req_len, 3, 3), -np.inf)
                mask = np.ones((req_len, 3, 3))
                return traj, mask
                
            p1 = df[['p1_x', 'p1_y', 'p1_z']].values
            p2 = df[['p2_x', 'p2_y', 'p2_z']].values
            shuttle = df[['shuttle_x', 'shuttle_y', 'shuttle_z']].values
            traj = np.stack([p1, p2, shuttle], axis=1)
            
            mask = np.isnan(traj).astype(float)
            # Use -inf for NaN as requested
            traj = np.nan_to_num(traj, nan=-np.inf)
            
            if len(traj) < req_len:
                pad_len = req_len - len(traj)
                padding = np.full((pad_len, 3, 3), -np.inf)
                padding_mask = np.ones((pad_len, 3, 3))
                traj = np.vstack([traj, padding])
                mask = np.vstack([mask, padding_mask])
                
            return traj, mask

        for start in range(0, num_frames, self.stride):
            end = min(start + self.window_size, num_frames)
            target_end = min(end + self.target_size, num_frames)
            
            input_df = segment_df.iloc[start:end].copy()
            target_df = segment_df.iloc[end:target_end].copy()
            
            # Skip window if there is absolutely no future trajectory to predict
            if len(target_df) == 0:
                continue
                
            input_traj, input_mask = extract_and_pad(input_df, self.window_size)
            target_traj, target_mask = extract_and_pad(target_df, self.target_size)
            
            if len(input_df) < self.window_size or len(target_df) < self.target_size:
                self.stats['windows_padded'] += 1

            window_dict = {
                'match_id': match_id,
                'segment_id': segment_id,
                'window_start_frame': int(input_df.iloc[0]['frame']),
                'window_end_frame': int(input_df.iloc[-1]['frame']),
                'input_traj': input_traj,
                'input_mask': input_mask,
                'target_traj': target_traj,
                'target_mask': target_mask,
                'length': len(input_df)
            }
            
            windows.append(window_dict)
            self.stats['windows_created'] += 1
        
        return windows
    
    def create_all_windows(self, df_norm):
        all_windows = []
        for match_id in df_norm['match_id'].unique():
            match_df = df_norm[df_norm['match_id'] == match_id]
            for seg_id in match_df['segment_idx'].unique():
                seg_df = match_df[match_df['segment_idx'] == seg_id]
                windows = self.create_segment_windows(seg_df, seg_id, match_id)
                all_windows.extend(windows)
                self.stats['total_segments'] += 1
                self.stats['total_frames'] += len(seg_df)
        return all_windows
    
    def split_train_val_test(self, windows, train_ratio=0.70, val_ratio=0.25):
        unique_matches = list(set(w['match_id'] for w in windows))
        test_ratio = 1.0 - train_ratio - val_ratio
        
        train_val_matches, test_matches = train_test_split(
            unique_matches, test_size=test_ratio, random_state=42
        )
        
        val_size_within_tv = val_ratio / (train_ratio + val_ratio)
        train_matches, val_matches = train_test_split(
            train_val_matches, test_size=val_size_within_tv, random_state=42
        )
        
        train_windows = [w for w in windows if w['match_id'] in train_matches]
        val_windows = [w for w in windows if w['match_id'] in val_matches]
        test_windows = [w for w in windows if w['match_id'] in test_matches]
        
        self.stats['train_windows'] = len(train_windows)
        self.stats['val_windows'] = len(val_windows)
        self.stats['test_windows'] = len(test_windows)
        
        split_info = {
            'train_matches': train_matches,
            'val_matches': val_matches,
            'test_matches': test_matches,
            'train_count': len(train_windows),
            'val_count': len(val_windows),
            'test_count': len(test_windows)
        }
        
        return train_windows, val_windows, test_windows, split_info

def save_windows(windows, output_path):
    dataset = {
        'input_trajectories': np.stack([w['input_traj'] for w in windows]),
        'input_masks': np.stack([w['input_mask'] for w in windows]),
        'target_trajectories': np.stack([w['target_traj'] for w in windows]),
        'target_masks': np.stack([w['target_mask'] for w in windows]),
        'metadata': [{
            'match_id': w['match_id'],
            'segment_id': w['segment_id'],
            'window_start_frame': w['window_start_frame'],
            'window_end_frame': w['window_end_frame'],
            'length': w['length']
        } for w in windows]
    }
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

def main():
    input_file = os.path.join(configs['global']['project_root'], configs['tranSPORTmer']['data']['normalized_csv'])
    output_dir = Path(os.path.join(configs['global']['project_root'], configs['tranSPORTmer']['data']['training_data_dir']))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("STEP 3: CREATE TRAINING WINDOWS")
    print("=" * 80)
    
    df_norm = pd.read_csv(input_file)
    generator = TrainingWindowGenerator(window_size=125, target_size=35, stride=25)
    all_windows = generator.create_all_windows(df_norm)
    train_windows, val_windows, test_windows, split_info = generator.split_train_val_test(all_windows)
    
    print(f"Train: {split_info['train_count']:,} | Val: {split_info['val_count']:,} | Test: {split_info['test_count']:,}")
    
    save_windows(train_windows, output_dir / 'train_windows.pkl')
    save_windows(val_windows, output_dir / 'val_windows.pkl')
    save_windows(test_windows, output_dir / 'test_windows.pkl')
    
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump({'split_info': split_info, 'window_stats': generator.stats}, f, indent=2)

if __name__ == '__main__':
    main()