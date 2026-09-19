import os
import numpy as np
import pandas as pd
from src.core.utils.configs import PathResolver
from src.core.utils.logger import get_logger

class CoordinateVerifier:
    def __init__(self):
        self.path_resolver = PathResolver()
        self.logger = get_logger(__name__)
        
        self.csv_path = self.path_resolver.get_path('dataset_creation.transformer_dataset_csv')
        
        court_cfg = self.path_resolver.get_config('dataset_creation.court')
        self.court_w = float(court_cfg.get('width', 6.1)) if court_cfg else 6.1
        self.court_l = float(court_cfg.get('length', 13.4)) if court_cfg else 13.4

    def verify(self):
        self.logger.info("=" * 80)
        self.logger.info("EXTENSIVE SHUTTLE COORDINATE VERIFICATION SUITE")
        self.logger.info("=" * 80)

        if not os.path.exists(self.csv_path):
            self.logger.error(f"Dataset not found at {self.csv_path}")
            return False

        df = pd.read_csv(self.csv_path)
        valid_df = df.dropna(subset=['shuttle_x', 'shuttle_y', 'shuttle_z']).copy()
        
        self._check_completeness_integrity(df)
        self._check_kinematic_anomalies(valid_df)
        self._check_geometric_bounds(valid_df)
        self._check_player_hit_proximity(df)
        self._check_sticky_ground_logic(valid_df)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("DIAGNOSTICS COMPLETE")
        self.logger.info("=" * 80)
        
        return True

    def _check_completeness_integrity(self, df):
        self.logger.info("\n[1] NaN & Completeness Integrity")
        self.logger.info("-" * 50)
        
        player_nans = df['p1_x'].isna().sum() + df['p2_x'].isna().sum()
        self.logger.info(f"  Player 1/2 NaNs (Total): {player_nans}")
        
        active_mask = np.zeros(len(df), dtype=bool)
        contiguous_nan_blocks = 0
        total_active_nans = 0
        
        for (match_id, seg_idx), seg_df in df.groupby(['match_id', 'segment_idx']):
            hits = seg_df[seg_df['is_hit_frame'] == 1]
            if len(hits) < 2:
                continue
            
            first_hit = hits['frame'].min()
            last_hit = hits['frame'].max()
            
            # Mask active play
            idx_active = seg_df[(seg_df['frame'] >= first_hit) & (seg_df['frame'] <= last_hit)].index
            active_mask[idx_active] = True
            
            # Check NaNs in active play
            active_seg = df.loc[idx_active]
            nan_mask = active_seg['shuttle_x'].isna()
            if nan_mask.any():
                nan_count = nan_mask.sum()
                total_active_nans += nan_count
                
                # Find contiguous blocks vs scattered
                nan_diff = nan_mask.astype(int).diff()
                num_blocks = (nan_diff == 1).sum()
                
                # If the first frame is NaN, .diff() won't catch it, so adjust
                if nan_mask.iloc[0]:
                    num_blocks += 1
                    
                contiguous_nan_blocks += num_blocks
                
        self.logger.info(f"  Active Play Frames: {active_mask.sum()}")
        if total_active_nans == 0:
            self.logger.info(f"  Active Play Shuttle NaNs: 0  (✓ Perfect!)")
        else:
            self.logger.info(f"  Active Play Shuttle NaNs: {total_active_nans} (across {contiguous_nan_blocks} distinct block(s))")
            if contiguous_nan_blocks == total_active_nans:
                self.logger.info(f"    -> All NaNs are scattered (single frames).")
            else:
                self.logger.info(f"    -> NaNs are clustered into contiguous chunks.")

    def _check_kinematic_anomalies(self, valid):
        self.logger.info("\n[2] Kinematic & Velocity Anomalies")
        self.logger.info("-" * 50)
        
        dx = valid['shuttle_x'].diff()
        dy = valid['shuttle_y'].diff()
        dz = valid['shuttle_z'].diff()
        
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        valid['dist'] = dist
        # Invalidate dist between different segments
        valid.loc[(valid['segment_idx'] != valid['segment_idx'].shift()), 'dist'] = np.nan
        
        max_dist = valid['dist'].max()
        p99_dist = valid['dist'].quantile(0.99)
        teleports = valid[valid['dist'] > 4.5]
        
        self.logger.info(f"  Max Frame-to-Frame Displacement: {max_dist:.2f} m")
        self.logger.info(f"  99th Percentile Displacement:    {p99_dist:.2f} m")
        
        if len(teleports) == 0:
            self.logger.info("  Teleportation (>4.5m/frame):     0 instances (✓ Realistic speeds)")
        else:
            self.logger.warning(f"  Teleportation (>4.5m/frame):     {len(teleports)} instances (✗ Anomalous)")
            self.logger.info("    Sample teleports:")
            for _, row in teleports.head(3).iterrows():
                self.logger.info(f"      - {row['match_id']} Seg {row['segment_idx']} Frame {row['frame']}: moved {row['dist']:.2f}m")

    def _check_geometric_bounds(self, valid):
        self.logger.info("\n[3] Geometric Bounds")
        self.logger.info("-" * 50)
        
        min_z = valid['shuttle_z'].min()
        max_z = valid['shuttle_z'].max()
        self.logger.info(f"  Z-Height Range: [{min_z:.2f}m, {max_z:.2f}m]")
        if min_z < 0:
            self.logger.warning(f"  ✗ WARNING: Negative Z coordinates detected!")
        else:
            self.logger.info(f"  ✓ Z coordinates strictly >= 0.")
            
        in_court = valid[
            (valid['shuttle_x'] >= 0) & (valid['shuttle_x'] <= self.court_w) &
            (valid['shuttle_y'] >= 0) & (valid['shuttle_y'] <= self.court_l)
        ]
        pct_in = 100 * len(in_court) / len(valid) if len(valid) > 0 else 0
        self.logger.info(f"  In-Court Percentage: {pct_in:.1f}%")

    def _check_player_hit_proximity(self, df):
        self.logger.info("\n[4] Player-to-Hit Proximity")
        self.logger.info("-" * 50)
        
        hits = df[df['is_hit_frame'] == 1].dropna(subset=['shuttle_x', 'shuttle_y'])
        
        p1_dist = np.sqrt((hits['shuttle_x'] - hits['p1_x'])**2 + (hits['shuttle_y'] - hits['p1_y'])**2)
        p2_dist = np.sqrt((hits['shuttle_x'] - hits['p2_x'])**2 + (hits['shuttle_y'] - hits['p2_y'])**2)
        
        hit_dist = np.where(hits['hitter'] == 'p1', p1_dist, p2_dist)
        hits = hits.copy()
        hits['hit_dist'] = hit_dist
        
        ghost_hits = hits[hits['hit_dist'] > 4.0]
        
        mean_hit_dist = np.nanmean(hit_dist) if len(hit_dist) > 0 else np.nan
        self.logger.info(f"  Average distance to hitter: {mean_hit_dist:.2f}m")
        if len(ghost_hits) == 0:
            self.logger.info(f"  Ghost Hits (>4m from player): 0 (✓ Assigned hitters are close)")
        else:
            self.logger.warning(f"  Ghost Hits (>4m from player): {len(ghost_hits)} (✗ Tracking/Hitter assignment errors)")
            self.logger.info("    Sample ghost hits:")
            for _, row in ghost_hits.head(3).iterrows():
                self.logger.info(f"      - {row['match_id']} Seg {row['segment_idx']} Frame {row['frame']}: Player is {row['hit_dist']:.2f}m away")

    def _check_sticky_ground_logic(self, valid):
        self.logger.info("\n[5] Sticky Ground Logic Verification")
        self.logger.info("-" * 50)
        
        sticky_failures = 0
        grounded_rallies = 0
        
        for (match_id, seg_idx), seg_df in valid.groupby(['match_id', 'segment_idx']):
            ground_hits = seg_df[seg_df['shuttle_z'] <= 0.0]
            if len(ground_hits) > 0:
                grounded_rallies += 1
                first_ground_frame = ground_hits['frame'].min()
                
                # All frames after the first ground frame must have Z=0 and same X,Y
                after_ground = seg_df[seg_df['frame'] > first_ground_frame]
                if len(after_ground) > 0:
                    first_ground_row = ground_hits.loc[ground_hits['frame'] == first_ground_frame].iloc[0]
                    gx, gy = first_ground_row['shuttle_x'], first_ground_row['shuttle_y']
                    
                    failed_x = (after_ground['shuttle_x'] != gx).sum()
                    failed_y = (after_ground['shuttle_y'] != gy).sum()
                    failed_z = (after_ground['shuttle_z'] != 0.0).sum()
                    
                    if failed_x > 0 or failed_y > 0 or failed_z > 0:
                        sticky_failures += 1

        self.logger.info(f"  Total rallies that terminated on ground (Z=0): {grounded_rallies}")
        if sticky_failures == 0:
            self.logger.info(f"  Sticky Logic Violations: 0 (✓ The shuttle stayed frozen)")
        else:
            self.logger.warning(f"  Sticky Logic Violations: {sticky_failures} (✗ The shuttle moved after hitting ground!)")

