import os
import numpy as np
import pandas as pd
import yaml

with open('../../configs/configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

CWD = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 80)
    print("EXTENSIVE SHUTTLE COORDINATE VERIFICATION SUITE")
    print("=" * 80)

    # Load dataset
    df = pd.read_csv(os.path.join(configs['global']['project_root'], configs['dataset_creation']['transformer_dataset_csv']))
    
    COURT_W, COURT_L = 6.1, 13.4
    
    # -------------------------------------------------------------------------
    # 1. NaN & Completeness Integrity
    # -------------------------------------------------------------------------
    print("\n[1] NaN & Completeness Integrity")
    print("-" * 50)
    
    player_nans = df['p1_x'].isna().sum() + df['p2_x'].isna().sum()
    print(f"  Player 1/2 NaNs (Total): {player_nans}")
    
    # Active Play NaNs
    # Active play is from the first hit of a segment to the last hit of a segment.
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
            
    print(f"  Active Play Frames: {active_mask.sum()}")
    if total_active_nans == 0:
        print(f"  Active Play Shuttle NaNs: 0  (✓ Perfect!)")
    else:
        print(f"  Active Play Shuttle NaNs: {total_active_nans} (across {contiguous_nan_blocks} distinct block(s))")
        if contiguous_nan_blocks == total_active_nans:
            print(f"    -> All NaNs are scattered (single frames).")
        else:
            print(f"    -> NaNs are clustered into contiguous chunks.")
            
    # -------------------------------------------------------------------------
    # 2. Kinematic & Velocity Anomalies
    # -------------------------------------------------------------------------
    print("\n[2] Kinematic & Velocity Anomalies")
    print("-" * 50)
    
    valid = df.dropna(subset=['shuttle_x', 'shuttle_y', 'shuttle_z']).copy()
    
    dx = valid['shuttle_x'].diff()
    dy = valid['shuttle_y'].diff()
    dz = valid['shuttle_z'].diff()
    
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    valid['dist'] = dist
    # Invalidate dist between different segments
    valid.loc[(valid['segment_idx'] != valid['segment_idx'].shift()), 'dist'] = np.nan
    
    max_dist = valid['dist'].max()
    p99_dist = valid['dist'].quantile(0.99)
    teleports = valid[valid['dist'] > 4.5]  # > 4.5m per frame (~135 m/s)
    
    print(f"  Max Frame-to-Frame Displacement: {max_dist:.2f} m")
    print(f"  99th Percentile Displacement:    {p99_dist:.2f} m")
    
    if len(teleports) == 0:
        print("  Teleportation (>4.5m/frame):     0 instances (✓ Realistic speeds)")
    else:
        print(f"  Teleportation (>4.5m/frame):     {len(teleports)} instances (✗ Anomalous)")
        print("    Sample teleports:")
        for _, row in teleports.head(3).iterrows():
            print(f"      - {row['match_id']} Seg {row['segment_idx']} Frame {row['frame']}: moved {row['dist']:.2f}m")
            
    # -------------------------------------------------------------------------
    # 3. Geometric Bounds
    # -------------------------------------------------------------------------
    print("\n[3] Geometric Bounds")
    print("-" * 50)
    
    min_z = valid['shuttle_z'].min()
    max_z = valid['shuttle_z'].max()
    print(f"  Z-Height Range: [{min_z:.2f}m, {max_z:.2f}m]")
    if min_z < 0:
        print(f"  ✗ WARNING: Negative Z coordinates detected!")
    else:
        print(f"  ✓ Z coordinates strictly >= 0.")
        
    in_court = valid[
        (valid['shuttle_x'] >= 0) & (valid['shuttle_x'] <= COURT_W) &
        (valid['shuttle_y'] >= 0) & (valid['shuttle_y'] <= COURT_L)
    ]
    pct_in = 100 * len(in_court) / len(valid)
    print(f"  In-Court Percentage: {pct_in:.1f}%")
    
    # -------------------------------------------------------------------------
    # 4. Player-to-Hit Proximity
    # -------------------------------------------------------------------------
    print("\n[4] Player-to-Hit Proximity")
    print("-" * 50)
    
    hits = df[df['is_hit_frame'] == 1].dropna(subset=['shuttle_x', 'shuttle_y'])
    
    p1_dist = np.sqrt((hits['shuttle_x'] - hits['p1_x'])**2 + (hits['shuttle_y'] - hits['p1_y'])**2)
    p2_dist = np.sqrt((hits['shuttle_x'] - hits['p2_x'])**2 + (hits['shuttle_y'] - hits['p2_y'])**2)
    
    hit_dist = np.where(hits['hitter'] == 'p1', p1_dist, p2_dist)
    hits = hits.copy()
    hits['hit_dist'] = hit_dist
    
    ghost_hits = hits[hits['hit_dist'] > 4.0] # > 4m away
    
    print(f"  Average distance to hitter: {np.nanmean(hit_dist):.2f}m")
    if len(ghost_hits) == 0:
        print(f"  Ghost Hits (>4m from player): 0 (✓ Assigned hitters are close)")
    else:
        print(f"  Ghost Hits (>4m from player): {len(ghost_hits)} (✗ Tracking/Hitter assignment errors)")
        print("    Sample ghost hits:")
        for _, row in ghost_hits.head(3).iterrows():
            print(f"      - {row['match_id']} Seg {row['segment_idx']} Frame {row['frame']}: Player is {row['hit_dist']:.2f}m away")
            
    # -------------------------------------------------------------------------
    # 5. Sticky Ground Logic Verification
    # -------------------------------------------------------------------------
    print("\n[5] Sticky Ground Logic Verification")
    print("-" * 50)
    
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

    print(f"  Total rallies that terminated on ground (Z=0): {grounded_rallies}")
    if sticky_failures == 0:
        print(f"  Sticky Logic Violations: 0 (✓ The shuttle stayed frozen)")
    else:
        print(f"  Sticky Logic Violations: {sticky_failures} (✗ The shuttle moved after hitting ground!)")

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
