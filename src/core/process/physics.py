import os
import pickle
import numpy as np
import pandas as pd
import cv2
from numba import njit
from scipy.optimize import root
from tqdm import tqdm
from src.core.utils.configs import PathResolver
from src.core.utils.logger import get_logger

# ── Physics (BVP Solver) ──────────────────────────────────────────────────────
@njit
def rk4_step_3d(state, dt, m, A_area, C_d, rho, g_val):
    """Perform one step of Runge-Kutta 4th order integration for 3D shuttle dynamics."""
    def deriv(s):
        v = s[3:6]
        speed = np.linalg.norm(v) + 1e-8
        if speed > 200:
            speed = 200.0
        g = np.array([0.0, 0.0, -g_val])
        k = 0.5 * rho * A_area * C_d / m
        a = g - k * speed * v
        return np.concatenate((v, a))
    
    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

@njit
def simulate_trajectory(A, V0, T, dt, m, A_area, C_d, rho, g_val):
    """Simulate the 3D trajectory from position A with initial velocity V0 for duration T."""
    v_norm = np.linalg.norm(V0)
    if v_norm > 200:
        V0 = V0 * (200.0 / v_norm)
    n_steps = int(round(T / dt))
    if n_steps <= 0:
        traj = np.zeros((1, 3))
        traj[0] = A
        return traj
    state = np.concatenate((A, V0))
    traj = np.zeros((n_steps + 1, 3))
    traj[0] = state[:3]
    for i in range(1, n_steps + 1):
        state = rk4_step_3d(state, dt, m, A_area, C_d, rho, g_val)
        traj[i] = state[:3]
    return traj

class PhysicsProcessor:
    def __init__(self):
        self.path_resolver = PathResolver()
        self.logger = get_logger(__name__)

        # Load physics constants from config
        self.physics_params = self.path_resolver.get_config("dataset_creation.physics")
        self.m = float(self.physics_params.get("shuttle_mass", 0.0055))
        self.A_area = float(self.physics_params.get("cross_section_area", 0.0028))
        self.C_d = float(self.physics_params.get("drag_coefficient", 0.44))
        self.rho = float(self.physics_params.get("air_density", 1.225))
        self.g_val = float(self.physics_params.get("gravity", 9.81))

        # Resolve paths
        self.input_csv = self.path_resolver.get_path("dataset_creation.pre_final_csv")
        self.camera_pose_cache = self.path_resolver.get_path("dataset_creation.camera_pose_cache")
        self.homography_cache = self.path_resolver.get_path("dataset_creation.homography_cache")
        self.output_csv = self.path_resolver.get_path("dataset_creation.transformer_dataset_csv")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)

    def _bvp_objective(self, V0, A, B, T, dt):
        traj = simulate_trajectory(A, V0, T, dt, self.m, self.A_area, self.C_d, self.rho, self.g_val)
        return traj[-1] - B

    def _solve_bvp(self, A, B, T, dt):
        if T <= 0:
            return np.zeros(3), False
        V0_guess = (B - A) / T
        V0_guess[2] += 0.5 * self.g_val * T
        res = root(self._bvp_objective, V0_guess, args=(A, B, T, dt), method='hybr')
        if not res.success:
            return V0_guess, False
        return res.x, True

    # ── Geometry ──────────────────────────────────────────────────────────────────
    def _camera_ray(self, px, py, K, R, tvec):
        K_inv = np.linalg.inv(K)
        ray_cam = K_inv @ np.array([px, py, 1.0])
        D = R.T @ ray_cam
        D = D / np.linalg.norm(D)
        C = (-R.T @ tvec).flatten()
        return C, D

    def _pixel_to_court_homography(self, px, py, H):
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def _intersect_ray_y_plane(self, C, D, y_plane):
        """Find X and Z coordinates by intersecting camera ray with a known Y-depth plane."""
        if abs(D[1]) < 1e-8:
            return None
        s = (y_plane - C[1]) / D[1]
        x = C[0] + s * D[0]
        z = C[2] + s * D[2]
        return np.array([x, y_plane, z])

    # -- Hitter identification (court-space proximity) ----------------------------
    def _assign_hitters(self, rally_df, hit_frames, H):
        """Assign the nearest player as hitter using court-space Euclidean distance."""
        hitter_map = {}
        for f in hit_frames:
            row = rally_df[rally_df['frame'] == f].iloc[0]
            sx, sy = row['shuttle_x'], row['shuttle_y']
            
            if pd.isna(sx) or pd.isna(sy):
                # Fallback: alternate from previous
                if len(hitter_map) > 0:
                    last_f = max(hitter_map.keys())
                    hitter_map[f] = 'p2' if hitter_map[last_f] == 'p1' else 'p1'
                else:
                    hitter_map[f] = 'p1'
                continue
            
            # Project shuttle to court
            s_cx, s_cy = self._pixel_to_court_homography(sx, sy, H)
            
            p1_dist = float('inf')
            p2_dist = float('inf')
            
            if not (pd.isna(row['p1x1']) or pd.isna(row['p1x2']) or pd.isna(row['p1y2'])):
                p1_bcx = (row['p1x1'] + row['p1x2']) / 2.0
                p1_cx, p1_cy = self._pixel_to_court_homography(p1_bcx, row['p1y2'], H)
                p1_dist = (s_cx - p1_cx)**2 + (s_cy - p1_cy)**2
                
            if not (pd.isna(row['p2x1']) or pd.isna(row['p2x2']) or pd.isna(row['p2y2'])):
                p2_bcx = (row['p2x1'] + row['p2x2']) / 2.0
                p2_cx, p2_cy = self._pixel_to_court_homography(p2_bcx, row['p2y2'], H)
                p2_dist = (s_cx - p2_cx)**2 + (s_cy - p2_cy)**2
                
            hitter_map[f] = 'p1' if p1_dist <= p2_dist else 'p2'
            
        return hitter_map

    def _get_hitter_bbox(self, row, hitter_id):
        if hitter_id == 'p1':
            v = [row['p1x1'], row['p1y1'], row['p1x2'], row['p1y2']]
        else:
            v = [row['p2x1'], row['p2y1'], row['p2x2'], row['p2y2']]
        if any(pd.isna(x) for x in v):
            return None
        return [float(x) for x in v]

    def _get_player_feet_y(self, row, hitter_id, H):
        bbox = self._get_hitter_bbox(row, hitter_id)
        if bbox is None: return None
        bx = (bbox[0] + bbox[2]) / 2.0
        _, by = self._pixel_to_court_homography(bx, bbox[3], H)
        return by

    # ── Pipeline ──────────────────────────────────────────────────────────────────
    def run(self):
        if not os.path.exists(self.input_csv):
            self.logger.error(f"Input CSV not found at {self.input_csv}. Exiting.")
            return

        if not os.path.exists(self.camera_pose_cache):
            self.logger.error(f"Camera pose cache not found at {self.camera_pose_cache}. Exiting.")
            return

        if not os.path.exists(self.homography_cache):
            self.logger.error(f"Homography cache not found at {self.homography_cache}. Exiting.")
            return

        self.logger.info(f"Loading {self.input_csv} ...")
        df = pd.read_csv(self.input_csv)
        
        # Interpolate player bounding boxes per match to prevent missing_start/missing_end
        p_cols = ['p1_cx', 'p1_cy', 'p1x1', 'p1y1', 'p1x2', 'p1y2', 'p2_cx', 'p2_cy', 'p2x1', 'p2y1', 'p2x2', 'p2y2']
        df[p_cols] = df.groupby('match_id')[p_cols].transform(lambda x: x.interpolate(limit_direction='both'))
        
        with open(self.camera_pose_cache, 'rb') as f:
            camera_poses = pickle.load(f)
        with open(self.homography_cache, 'rb') as f:
            homographies = pickle.load(f)

        # hardcoded fps value because model has been trained on videos with 25 fps
        fps = float(self.physics_params.get('fps', 25.0))
        dt = 1.0 / fps
        output_rows = []
        
        skip_reasons = {
            'missing_start': 0,
            'missing_end': 0,
            'bvp_failed': 0
        }
        total_hits = 0

        rallies = df.groupby(['match_id', 'segment_idx'])
        for (match_no, point_no), rally_df in tqdm(rallies, total=len(rallies), desc="Segments"):
            match_id = str(match_no) if str(match_no).endswith('.mp4') else f"{match_no}.mp4"
            if match_id not in camera_poses or match_id not in homographies:
                self.logger.warning(f"Missing camera pose or homography for {match_id}, skipping.")
                continue
                
            pose_dict = camera_poses[match_id]
            K = pose_dict['K']
            R = pose_dict['R']
            tvec = pose_dict['tvec']
            H = homographies[match_id]
            
            hit_rows = rally_df[rally_df['is_hit_frame'] == 1]
            if hit_rows.empty:
                continue
                
            hit_frames = sorted(hit_rows['frame'].tolist())
            shuttle_3d = {}   # frame -> (x, y, z)
            
            hitter_map = self._assign_hitters(rally_df, hit_frames, H)
            
            rally_ended = False
            
            # Process segment by segment (A to B)
            for i in range(len(hit_frames)):
                if rally_ended:
                    break

                total_hits += 1
                frame_A = hit_frames[i]
                row_A = rally_df[rally_df['frame'] == frame_A].iloc[0]
                hitter_A = hitter_map[frame_A]
                
                # Find Start Y and Point A
                Y_A = self._get_player_feet_y(row_A, hitter_A, H)
                if Y_A is None:
                    skip_reasons['missing_start'] += 1
                    continue
                    
                s_px_A = row_A['shuttle_x']
                s_py_A = row_A['shuttle_y']
                if pd.isna(s_px_A) or pd.isna(s_py_A):
                    skip_reasons['missing_start'] += 1
                    continue
                    
                C_A, D_A = self._camera_ray(s_px_A, s_py_A, K, R, tvec)
                P_A = self._intersect_ray_y_plane(C_A, D_A, Y_A)
                if P_A is None:
                    skip_reasons['missing_start'] += 1
                    continue
                    
                # Find End Y and Point B
                frame_B = None
                if P_A is not None:
                    P_A[2] = max(0.05, P_A[2])
                    
                Y_B = None
                P_B = None
                
                if i + 1 < len(hit_frames):
                    # Mid-rally hit
                    frame_B = hit_frames[i+1]
                    row_B = rally_df[rally_df['frame'] == frame_B].iloc[0]
                    hitter_B = hitter_map[frame_B]
                    Y_B = self._get_player_feet_y(row_B, hitter_B, H)
                    if Y_B is not None:
                        s_px_B = row_B['shuttle_x']
                        s_py_B = row_B['shuttle_y']
                        if not (pd.isna(s_px_B) or pd.isna(s_py_B)):
                            C_B, D_B = self._camera_ray(s_px_B, s_py_B, K, R, tvec)
                            P_B = self._intersect_ray_y_plane(C_B, D_B, Y_B)
                
                if P_B is None:
                    # Last hit or missing next hit data. Use last visible pixel projection.
                    end_search = hit_frames[i+1] if i + 1 < len(hit_frames) else int(rally_df['frame'].max())
                    last_vis_row = None
                    for f in range(end_search, frame_A, -1):
                        f_rows = rally_df[rally_df['frame'] == f]
                        if not f_rows.empty:
                            r = f_rows.iloc[0]
                            if not (pd.isna(r['shuttle_x']) or pd.isna(r['shuttle_y'])):
                                last_vis_row = r
                                frame_B = f
                                break
                    
                    if last_vis_row is not None:
                        bx, by = self._pixel_to_court_homography(last_vis_row['shuttle_x'], last_vis_row['shuttle_y'], H)
                        P_B = np.array([bx, by, 0.05])
                        Y_B = by
                    else:
                        skip_reasons['missing_end'] += 1
                        continue
                        
                if P_B is not None:
                    P_B[2] = max(0.05, P_B[2])
                
                T = (frame_B - frame_A) * dt
                if T <= 0:
                    continue
                    
                # Solve BVP
                V0, success = self._solve_bvp(P_A, P_B, T, dt)
                if not success:
                    skip_reasons['bvp_failed'] += 1
                    V0 = (P_B - P_A) / T
                    V0[2] += 0.5 * self.g_val * T
                
                # Physics Trajectory
                traj_phys = simulate_trajectory(P_A, V0, T, dt, self.m, self.A_area, self.C_d, self.rho, self.g_val)
                
                # Video Hybrid Fusion
                last_frame_max = int(rally_df['frame'].max())
                for j, f in enumerate(range(frame_A, frame_B + 1)):
                    if j >= len(traj_phys): break
                    
                    # Check for smooth descent to ground (physics)
                    if traj_phys[j, 2] <= 0.0:
                        ground_pt = traj_phys[j].copy()
                        ground_pt[2] = 0.0
                        for future_f in range(f, last_frame_max + 1):
                            shuttle_3d[future_f] = ground_pt
                        rally_ended = True
                        break
                    
                    Y_phys = traj_phys[j, 1]
                    
                    f_rows = rally_df[rally_df['frame'] == f]
                    if f_rows.empty:
                        shuttle_3d[f] = traj_phys[j]
                        continue
                    row_f = f_rows.iloc[0]
                    sx, sy = row_f['shuttle_x'], row_f['shuttle_y']
                    
                    final_pt = traj_phys[j]  # default
                    if not (pd.isna(sx) or pd.isna(sy)):
                        C, D = self._camera_ray(sx, sy, K, R, tvec)
                        P_video = self._intersect_ray_y_plane(C, D, Y_phys)
                        if P_video is not None:
                            P_video[2] = max(0.0, P_video[2])
                            final_pt = P_video
                    
                    # If the final point is at ground level, freeze
                    if final_pt[2] <= 0.0:
                        ground_pt = final_pt.copy()
                        ground_pt[2] = 0.0
                        for future_f in range(f, last_frame_max + 1):
                            shuttle_3d[future_f] = ground_pt
                        rally_ended = True
                        break
                    
                    shuttle_3d[f] = final_pt

            # Build output rows
            rally_rows = []
            for _, row in rally_df.iterrows():
                frame = int(row['frame'])
                pos = shuttle_3d.get(frame, (np.nan, np.nan, np.nan))
                sx, sy, sz = pos
                hitter = hitter_map.get(frame, None)
                rally_rows.append(self._build_row(row, sx, sy, sz, hitter, H))
                
            output_rows.extend(rally_rows)

        out_df = pd.DataFrame(output_rows)
        out_df.to_csv(self.output_csv, index=False)

        self.logger.info(f"Transformer dataset written → {self.output_csv}")
        self.logger.info(f"Total rows: {len(out_df)} | Hit frames: {total_hits} | Skipped Segments: {sum(skip_reasons.values())}")
        for reason, count in skip_reasons.items():
            self.logger.info(f"  - {reason:15s}: {count}")

        nan_summary = []
        for col in ['p1_x', 'p1_y', 'p2_x', 'p2_y', 'shuttle_x', 'shuttle_y', 'shuttle_z']:
            pct = out_df[col].isna().mean() * 100
            nan_summary.append(f"{col}: {pct:.1f}%")
        self.logger.info("NaN report: " + " | ".join(nan_summary))

    def _build_row(self, row, sx, sy, sz, hitter_id, H):
        p1_cx_court, p1_cy_court = np.nan, np.nan
        if not (pd.isna(row['p1x1']) or pd.isna(row['p1x2']) or pd.isna(row['p1y2'])):
            p1_bottom_center_x = (row['p1x1'] + row['p1x2']) / 2.0
            p1_cx_court, p1_cy_court = self._pixel_to_court_homography(p1_bottom_center_x, row['p1y2'], H)

        p2_cx_court, p2_cy_court = np.nan, np.nan
        if not (pd.isna(row['p2x1']) or pd.isna(row['p2x2']) or pd.isna(row['p2y2'])):
            p2_bottom_center_x = (row['p2x1'] + row['p2x2']) / 2.0
            p2_cx_court, p2_cy_court = self._pixel_to_court_homography(p2_bottom_center_x, row['p2y2'], H)

        return {
            'match_id':   row['match_id'],
            'segment_idx':row['segment_idx'],
            'frame':      row['frame'],
            'is_hit_frame':row['is_hit_frame'],
            'hitter':     hitter_id,
            'p1_x':       p1_cx_court,
            'p1_y':       p1_cy_court,
            'p1_z':       0.0,
            'p2_x':       p2_cx_court,
            'p2_y':       p2_cy_court,
            'p2_z':       0.0,
            'shuttle_x':  sx,
            'shuttle_y':  sy,
            'shuttle_z':  sz,
        }


    
