"""
Combine step: merge shuttle tracks + player positions per segment, detect hit frames.

Pipeline position:
  segments pass → second_pass (shuttle tracks) → third_pass (player tracks) → THIS → apply_physics

Inputs:
  outputs/segments/matchN.json         — [start, end] frame ranges
  outputs/shuttle_tracks/matchN/segmentM/matchN_ball.csv — Frame (0-indexed), Visibility, X, Y
  outputs/player_tracks/matchN_players.json              — rally[].{segment, positions[]}

Output:
  outputs/pre_final_dataset.csv
    Columns: match_id, segment_idx, frame, shuttle_vis, shuttle_x, shuttle_y,
             p1_cx, p1_cy, p1x1, p1y1, p1x2, p1y2,
             p2_cx, p2_cy, p2x1, p2y1, p2x2, p2y2,
             is_hit_frame
"""

import os
import json
import glob
import re
import numpy as np
import pandas as pd
import yaml

with open('../../configs/configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

CWD = os.path.dirname(os.path.abspath(__file__))
SEGMENTS_DIR = os.path.join(configs['global']['project_root'], configs['dataset_creation']['segments_dir'])
SHUTTLE_DIR  = os.path.join(configs['global']['project_root'], configs['dataset_creation']['shuttle_tracks_dir'])
PLAYER_DIR   = os.path.join(configs['global']['project_root'], configs['dataset_creation']['player_tracks_dir'])
OUTPUT_CSV   = os.path.join(configs['global']['project_root'], configs['dataset_creation']['pre_final_csv'])

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


# ── Hit-frame detection ──────────────────────────────────────────────────────

def detect_hit_frames(df, vis_threshold=0.5, speed_dev_factor=0.5, angle_change_deg=30,
                      min_gap=7):
    """
    Detect hit frames from raw shuttle pixel tracks.

    A hit is flagged when BOTH conditions are met (within a ±1 frame window):
      1. Speed deviation: instantaneous speed deviates from the local median
         by more than `speed_dev_factor` (40%) in EITHER direction.
         - Increase → smash / drive
         - Decrease → drop shot / net shot
      2. Direction change: angle between consecutive velocity vectors > `angle_change_deg`

    No re-entry detection is used.  When the shuttle goes out of frame on a
    high lift and falls back in, its speed profile is smooth (gravity arc) so
    neither the speed-deviation nor the direction-change condition fires —
    the re-entry is naturally ignored without a special case.

    Returns a boolean Series aligned to `df.index`.
    """
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


# ── Main processing ──────────────────────────────────────────────────────────

def process_match(match_name):
    """
    Process a single match: load segments, shuttle CSVs, player JSON.
    Returns a list of DataFrames (one per segment) or empty list on failure.
    """
    seg_json_path = os.path.join(SEGMENTS_DIR, f"{match_name}.json")
    if not os.path.exists(seg_json_path):
        print(f"  [SKIP] No segments JSON for {match_name}")
        return []

    with open(seg_json_path, 'r') as f:
        segments = json.load(f).get('segments', [])
    if not segments:
        print(f"  [SKIP] Empty segments for {match_name}")
        return []

    # Load player tracks
    player_json_path = os.path.join(PLAYER_DIR, f"{match_name}_players.json")
    player_data = {}  # segment_tuple → [{frame, p1_cx, ...}, ...]
    if os.path.exists(player_json_path):
        with open(player_json_path, 'r') as f:
            pdata = json.load(f)
        for rally in pdata.get('rally', []):
            seg_key = tuple(rally['segment'])
            player_data[seg_key] = rally['positions']
    else:
        print(f"  [WARN] No player tracks for {match_name}")

    match_shuttle_dir = os.path.join(SHUTTLE_DIR, match_name)
    if not os.path.isdir(match_shuttle_dir):
        print(f"  [SKIP] No shuttle track directory for {match_name}")
        return []

    segment_dfs = []

    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        seg_num = seg_idx + 1  # segment directories are 1-indexed
        seg_dir = os.path.join(match_shuttle_dir, f"segment{seg_num}")

        # ── Load shuttle CSV ──
        ball_csvs = glob.glob(os.path.join(seg_dir, '*_ball.csv'))
        if not ball_csvs:
            continue
        shuttle_df = pd.read_csv(ball_csvs[0])

        # Convert relative Frame → absolute frame
        shuttle_df['frame'] = shuttle_df['Frame'] + seg_start
        shuttle_df = shuttle_df.rename(columns={
            'Visibility': 'shuttle_vis',
            'X':          'shuttle_x',
            'Y':          'shuttle_y',
        }).drop(columns=['Frame'])

        # ── Load player positions for this segment ──
        seg_key = (seg_start, seg_end)
        if seg_key in player_data:
            player_df = pd.DataFrame(player_data[seg_key])
        else:
            # Create empty player columns
            player_df = pd.DataFrame({
                'frame': shuttle_df['frame'],
                'p1_cx': np.nan, 'p1_cy': np.nan,
                'p1x1': np.nan, 'p1y1': np.nan, 'p1x2': np.nan, 'p1y2': np.nan,
                'p2_cx': np.nan, 'p2_cy': np.nan,
                'p2x1': np.nan, 'p2y1': np.nan, 'p2x2': np.nan, 'p2y2': np.nan,
            })

        # ── Merge on absolute frame ──
        merged = pd.merge(shuttle_df, player_df, on='frame', how='outer')
        merged = merged.sort_values('frame').reset_index(drop=True)

        # ── Detect hit frames ──
        hit_mask = detect_hit_frames(merged)
        merged['is_hit_frame'] = hit_mask.astype(int)

        # ── Tag identifiers ──
        merged.insert(0, 'segment_idx', seg_idx)
        merged.insert(0, 'match_id', match_name)

        segment_dfs.append(merged)

    return segment_dfs


def main():
    # Discover matches from shuttle_tracks directory (most reliable source)
    if not os.path.isdir(SHUTTLE_DIR):
        print(f"No shuttle tracks directory found at {SHUTTLE_DIR}")
        return

    matches = sorted(
        [d for d in os.listdir(SHUTTLE_DIR) if os.path.isdir(os.path.join(SHUTTLE_DIR, d))],
        key=natural_sort_key
    )

    print(f" Found {len(matches)} matches with shuttle tracks\n")

    all_dfs = []
    for match in matches:
        print(f"Processing {match}...")
        dfs = process_match(match)
        if dfs:
            all_dfs.extend(dfs)
            total_segs = len(dfs)
            total_hits = sum(d['is_hit_frame'].sum() for d in dfs)
            print(f"  ✓ {total_segs} segments, {total_hits} hit frames detected")
        else:
            print(f"  ✗ No data produced")

    if not all_dfs:
        print("\nNo data found — check that second_pass and third_pass have completed.")
        return

    master_df = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    master_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"Master dataset saved → '{OUTPUT_CSV}'")
    print(f"  Matches:      {master_df['match_id'].nunique()}")
    print(f"  Segments:     {master_df.groupby(['match_id', 'segment_idx']).ngroups}")
    print(f"  Total frames: {len(master_df)}")
    print(f"  Hit frames:   {master_df['is_hit_frame'].sum()}")
    print(f"  Columns:      {list(master_df.columns)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()