import os
import json
import glob
import re
import numpy as np
import pandas as pd
import yaml
from src.core.utils.configs import PathResolver
from src.core.utils.logger import get_logger

class Executer:
    def __init__(self):
        self.resolver = PathResolver()
        self.logger = get_logger(__name__)
        self.segments_dir = self.resolver.get_path("dataset_creation.segments_dir")
        self.shuttle_dir = self.resolver.get_path("dataset_creation.shuttle_tracks_dir")
        self.player_dir = self.resolver.get_path("dataset_creation.player_tracks_dir")
        self.output_csv = self.resolver.get_path("dataset_creation.pre_final_csv")
    
    def _natural_sort_key(self, s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    def _load_segments_file(self, match):
        segments_file = os.path.join(self.segments_dir, f"{match}.json")
        if not os.path.exists(segments_file):
            return None
        with open(segments_file, 'r') as f:
            segments = json.load(f).get("segments", [])

        return segments 
    
    def _load_players_file(self, match):
        player_file = os.path.join(self.player_dir, f"{match}_players.json")
        player_data = {}  # segment_tuple → [{frame, p1_cx, ...}, ...]
        if os.path.exists(player_file):
            with open(player_file, 'r') as f:
                pdata = json.load(f)
            for rally in pdata.get('rally', []):
                seg_key = tuple(rally['segment'])
                player_data[seg_key] = rally['positions']
            return player_data
        else:
            self.logger.error(f"No player tracks for match {match}")
            return None

    def _detect_hit_frames(self, df, vis_threshold=0.5, speed_dev_factor=0.5, angle_change_deg=30, min_gap=7):
        hits = pd.Series(False, index=df.index)

        # Need at least 4 frames to compute meaningful velocity
        if len(df) < 4:
            return hits

        x = df['shuttle_x'].values.astype(float)
        y = df['shuttle_y'].values.astype(float)
        vis = df['shuttle_vis'].values.astype(float)

        # Velocity vectors (pixel/frame)
        dx = np.diff(x)
        dy = np.diff(y)
        speed = np.sqrt(dx**2 + dy**2)

        # Running median speed (window=7, padded)
        speed_series = pd.Series(speed)
        median_speed = speed_series.rolling(7, center=True, min_periods=1).median().values
        median_speed = np.maximum(median_speed, 1.0)  # avoid divide-by-zero

        # Direction angle change between consecutive velocity vectors
        angle_change = np.zeros(len(dx) - 1)
        for i in range(len(dx) - 1):
            v1 = np.array([dx[i], dy[i]])
            v2 = np.array([dx[i+1], dy[i+1]])
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 0.5 and n2 > 0.5:  # only if both vectors are non-trivial
                cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angle_change[i] = np.degrees(np.arccos(cos_theta))

        # Scan for hits
        last_hit_idx = -min_gap - 1  # allow first hit immediately

        for i in range(1, len(speed)):
            frame_pos = i  # position in the diff arrays (0-indexed)

            # Must be visible at this frame
            if vis[i] < vis_threshold or vis[i+1 if i+1 < len(vis) else i] < vis_threshold:
                continue

            # 1. Sudden speed change — either increase (smash) or decrease (drop)
            speed_deviation = abs(speed[i] - median_speed[i]) / median_speed[i]
            is_speed_change = speed_deviation > speed_dev_factor

            # 2. Direction change
            is_direction_change = False
            if frame_pos < len(angle_change):
                is_direction_change = angle_change[frame_pos] > angle_change_deg
            # Also check the frame before for ±1 tolerance
            if frame_pos - 1 >= 0 and frame_pos - 1 < len(angle_change):
                is_direction_change = is_direction_change or (angle_change[frame_pos - 1] > angle_change_deg)

            # BOTH conditions must be true
            if is_direction_change and is_speed_change:
                # Enforce minimum gap between hits
                if (i - last_hit_idx) >= min_gap:
                    hits.iloc[i + 1] = True  # +1 because diff shifts by 1
                    last_hit_idx = i

        return hits

    def _process_match(self, match):
        segments = self._load_segments_file(match)
        if segments is None:
            self.logger.error(f"Segments file not found for match {match}")
            return None

        player_data = self._load_players_file(match)
        if player_data is None:
            return None
        
        match_shuttle_dir = os.path.join(self.shuttle_dir, match)
        if not os.path.isdir(match_shuttle_dir):
            self.logger.error(f"No shuttle directory for {match}")
            return None
        
        consolidated_shuttle_dfs = []
        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            shuttle_path_for_segment = os.path.join(match_shuttle_dir, f"segment{seg_idx + 1}_ball.csv")
            if not os.path.exists(shuttle_path_for_segment):
                self.logger.warning(f"No shuttle CSV for segment {seg_idx + 1} in {match}")
                continue
            
            shuttle_df = pd.read_csv(shuttle_path_for_segment)
            shuttle_df['frame'] = shuttle_df['Frame'] + seg_start
            shuttle_df = shuttle_df.rename(columns={
                'Visibility': 'shuttle_vis',
                'X':          'shuttle_x',
                'Y':          'shuttle_y',
            }).drop(columns=['Frame'])
            
            key = (seg_start, seg_end)
            players = player_data.get(key, [])
            if players:
                player_df = pd.DataFrame(player_data[key])
            else:
                # Create empty player columns
                player_df = pd.DataFrame({
                    'frame': shuttle_df['frame'],
                    'p1_cx': np.nan, 'p1_cy': np.nan,
                    'p1x1': np.nan, 'p1y1': np.nan, 'p1x2': np.nan, 'p1y2': np.nan,
                    'p2_cx': np.nan, 'p2_cy': np.nan,
                    'p2x1': np.nan, 'p2y1': np.nan, 'p2x2': np.nan, 'p2y2': np.nan,
                })
            
            merged = pd.merge(shuttle_df, player_df, on='frame', how='outer')
            merged = merged.sort_values('frame').reset_index(drop=True)

            hit_mask = self._detect_hit_frames(merged)
            merged['is_hit_frame'] = hit_mask.astype(int)

            # ── Tag identifiers ──
            merged.insert(0, 'segment_idx', seg_idx)
            merged.insert(0, 'match_id', match)

            consolidated_shuttle_dfs.append(merged)

        return consolidated_shuttle_dfs     

    def run(self):
        if not os.path.isdir(self.shuttle_dir):
            self.logger.error("Shuttle directory not found. Exiting")
            return
        if not os.path.isdir(self.player_dir):
            self.logger.error("Player directory not found. Exiting")
            return
        if not os.path.isdir(self.segments_dir):
            self.logger.error("Segments directory not found. Exiting")
            return
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        matches = sorted(
            [d for d in os.listdir(self.shuttle_dir) if os.path.isdir(os.path.join(self.shuttle_dir, d))],
            key=self._natural_sort_key
        )
        if not matches:
            self.logger.error("No matches found. Exiting")
            return

        dataset = []
        for match in matches:
            complete_match_df = self._process_match(match)
            if complete_match_df is not None:
                dataset.append(complete_match_df)
                total_segs = len(complete_match_df)
                total_hits = sum(d['is_hit_frame'].sum() for d in complete_match_df)
                self.logger.info(f"Match {match}: {total_segs} segments, {total_hits} hits")
            else:
                self.logger.warning(f"Match {match}: No data processed")

        if not dataset:
            self.logger.error("No dataset created. Exiting")
            return
        
        master_df = pd.concat(dataset, ignore_index=True)
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        master_df.to_csv(self.output_csv, index=False)
        self.logger.info(f"Dataset created at {self.output_csv}")
            
        
        
        
