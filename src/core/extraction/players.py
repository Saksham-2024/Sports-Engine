import os
import sys
import cv2
import re
import numpy as np
import pickle
import json
from ultralytics import YOLO
from src.core.utils.configs import PathResolver
from src.core.utils.logger import get_logger
"""
Core module for extracting player positions from match videos.

Provides:
- `PlayerDetector` – wraps YOLO model and homography to detect players per frame.
- `PlayerExtractionPipeline` – orchestrates processing of all videos, handling
  segment boundaries, homography lookup, and output serialization.

The implementation strives for clear separation of concerns and is
documented to aid future extensions (e.g., adding shuttle detection or
different camera setups).
"""

class PlayerDetector:
    """
    Detects player bounding boxes in a video frame using a YOLO model and
    maps the foot centre to court coordinates via a provided homography matrix.
    """
    def __init__(self, model_path, H):
        """
        Initialise the detector.

        Parameters
        ----------
        model_path: str
            Path to the YOLO model file.
        H: np.ndarray
            3x3 homography matrix mapping image pixels to court coordinates.
        """
        self.yolo = YOLO(model_path)
        self.H = H

    def _pixel_to_court(self, x, y):
        """Convert a pixel (x, y) to court coordinates using the homography matrix H."""
        return cv2.perspectiveTransform(np.array([[[x, y]]], dtype='float32'), self.H)[0][0]

    def _is_on_court(self, court_x, court_y, court_width=5.18, court_length=13.4):
        """Return ``True`` if the provided court coordinates lie inside the official court dimensions."""
        return 0 <= court_x <= court_width and 0 <= court_y <= court_length
    
    def yolo_predict(self, frame, frame_idx):
        """Run YOLO detection on a single frame and return a list of player dictionaries.

        The returned list contains dictionaries with court coordinates (c_x, c_y)
        and the original bounding‑box coordinates (x1, y1, x2, y2).
        """
        players_pos = []
        results = self.yolo.predict(frame, verbose=False)

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    if int(box.cls) == 0:  # class 0 = person
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        # Use the horizontal midpoint of the bottom edge as the
                        # foot contact point — more stable than the full bbox centre
                        feet_center_x = int((x1 + x2) / 2)
                        feet_center_y = int(y2)
                        court_coords = self._pixel_to_court(feet_center_x, feet_center_y)
                        court_x, court_y = court_coords[0], court_coords[1]
                        if self._is_on_court(court_x, court_y):
                            # Convert types so json serialization succeeds
                            player = {
                                "c_x": float(court_x),
                                "c_y": float(court_y),
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2)
                            }
                            players_pos.append(player)

        return self._heal_player_coords(players_pos, frame_idx)

    def _heal_player_coords(self, players_pos, frame_idx):
        """Create a flat row dictionary for a given frame.

        The function sorts detected players by their court ``c_y`` value so that
        ``p1`` corresponds to the far side (lower ``c_y``) and ``p2`` to the near
        side when exactly two players are present.
        """
        row = {
            'frame': frame_idx,
            'p1_cx': None, 'p1_cy': None,
            'p1x1': None, 'p1y1': None, 'p1x2': None, 'p1y2': None,
            'p2_cx': None, 'p2_cy': None,
            'p2x1': None, 'p2y1': None, 'p2x2': None, 'p2y2': None,
        }
        if len(players_pos) > 0:
            players_pos.sort(key=lambda p: p["c_y"])  # sort by court_y: lower = far side

            # P1 = far side (lower court_y)
            row['p1_cx'], row['p1_cy'] = players_pos[0]["c_x"], players_pos[0]["c_y"]
            row['p1x1'], row['p1y1'] = players_pos[0]["x1"], players_pos[0]["y1"]
            row['p1x2'], row['p1y2'] = players_pos[0]["x2"], players_pos[0]["y2"]

            if len(players_pos) == 2:
                # P2 = near side (higher court_y)
                row['p2_cx'], row['p2_cy'] = players_pos[1]["c_x"], players_pos[1]["c_y"]
                row['p2x1'], row['p2y1'] = players_pos[1]["x1"], players_pos[1]["y1"]
                row['p2x2'], row['p2y2'] = players_pos[1]["x2"], players_pos[1]["y2"]

            # TODO:  should ensure a case where one player is occluded and heal the coords according to that case
        return row

class PlayerExtractionPipeline:
    """High‑level pipeline that processes every match video in the dataset.

    It loads homography matrices, iterates over video files, extracts player
    positions per frame, respects segment boundaries, and writes the results to
    JSON files.
    """
    def __init__(self):
        """Initialise utilities, resolve configuration paths and load homographies.

        The resolver abstracts away absolute paths defined in ``configs.yaml``.
        """
        self.resolver = PathResolver()
        self.logger = get_logger(__name__)
        self.yolo = YOLO(str(self.resolver.get_path("models.yolo")))
        self.video_dir = self.resolver.get_path("global.video_dir")
        self.segments_dir = self.resolver.get_path("dataset_creation.segments_dir")
        self.homography_cache = self.resolver.get_path("dataset_creation.homography_cache")
        self.player_tracks_dir = self.resolver.get_path("dataset_creation.player_tracks_dir")
        os.makedirs(self.player_tracks_dir, exist_ok=True)
        self._load_homography_matrices()

    def _load_homography_matrices(self):
        """Load cached homography matrices from ``homography_cache.pkl``.

        The cache maps video filenames (without extension) to a 3×3 numpy
        matrix. If the cache is missing the pipeline aborts with a clear error.
        """
        if os.path.exists(self.homography_cache):
            with open(self.homography_cache, 'rb') as f:
                raw_matrices = pickle.load(f)
                self.matrices = {os.path.basename(k): v for k, v in raw_matrices.items()}
            self.logger.info("Homography matrices loaded.")
        else:
            raise FileNotFoundError(f"Homography cache not found at {self.homography_cache}")

    def _natural_sort_key(self, s):
        """Key function for natural sorting of filenames containing numbers.

        Example: ``match2.mp4`` comes before ``match10.mp4``.
        """
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
    
    def _setup_capture_header(self, video_path):
        """Create a ``cv2.VideoCapture`` object and verify that the file can be opened.
        Returns ``None`` on failure so the caller can safely skip the video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Error opening {video_path}")
            return None
        return cap
    
    def _total_frames(self, cap):
        """Return the total number of frames in the supplied capture object."""
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _seek_exact(self, cap, target_frame):
        """Seek to an exact frame, correcting for compression‑induced drift.

        After calling ``cap.set``, the actual position is verified. If the
        requested frame is not reached, additional reads or a small back‑track
        are performed to land precisely on ``target_frame``.
        """
        """Seek to exact frame, correcting for compression-induced drift."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if actual != target_frame:
            if actual < target_frame:
                # Behind target — read forward to catch up
                for _ in range(target_frame - actual):
                    cap.read()
            else:
                # Overshot — re-seek from slightly before and read forward
                safe_start = max(0, target_frame - 5)
                cap.set(cv2.CAP_PROP_POS_FRAMES, safe_start)
                for _ in range(target_frame - safe_start):
                    cap.read()
    
    def _get_segments(self, match_name):
        """Load per‑match segment information from a JSON file.

        ``match_name`` is the video filename without extension. The JSON file
        contains a ``segments`` list where each element is ``[start_frame,
        end_frame]`` describing a rally. ``None`` is returned when the file is
        absent, signalling that the whole video should be processed as a
        single segment.
        """
        segment_file = os.path.join(self.segments_dir, f"{match_name}.json")
        if os.path.exists(segment_file):
            with open(segment_file, 'r') as f:
                segments = json.load(f).get('segments', [])
            return segments
        else:
            self.logger.warning(f"Segments file not found for {match_name}")
            return None
        
    def _loop(self, cap, segments, total_frames, H):
        """Iterate over frames, apply detection and aggregate results.

        * If ``segments`` is provided, processing is limited to those frame
          ranges; otherwise the whole video is processed.
        * ``H`` is the homography matrix used by ``PlayerDetector``.
        * The method returns a list of rally dictionaries ready for JSON
          serialization.
        """
        rally_list = []
        processed_count = 0
        frame_idx = 0
        current_segment = None
        current_positions = []

        player_detector = PlayerDetector(self.yolo, H)
        while cap.isOpened() and frame_idx < total_frames:
            # Case 1 & 3: Check if segment boundaries are defined
            if segments is not None:
                in_seg = False
                active_seg = None

                # can be improved for time complexity
                for seg in segments:
                    if seg[0] <= frame_idx <= seg[1]:
                        in_seg = True
                        active_seg = seg
                        break
            
                # Case 2: Gap Case (outside any rally)
                if not in_seg:
                    # Close the previous rally if we were tracking one
                    if current_segment is not None:
                        rally_list.append({
                            "segment": current_segment,
                            "positions": current_positions
                        })
                        current_segment = None
                        current_positions = []

                    # Optimization: Skip decoding dead frames between rallies
                    next_start = None

                    # can be improved for time complexity
                    for seg in segments:
                        if seg[0] > frame_idx:
                            next_start = seg[0]
                            break
                    
                    if next_start is not None:
                        self._seek_exact(cap, next_start)
                        frame_idx = next_start
                        continue
                    else:
                        break # No more segments remaining in video
                else:
                    # Case 3: Active Rally Case (inside a segment)
                    # If entering a new rally, commit the previous one and start fresh
                    if current_segment != active_seg:
                        if current_segment is not None:
                            rally_list.append({
                                "segment": current_segment,
                                "positions": current_positions
                            })
                        current_segment = active_seg
                        current_positions = []

            ret, frame = cap.read()
            if not ret:
                break
            
            processed_count += 1
            row = player_detector.yolo_predict(frame, frame_idx)

            # Record detection: either to the active rally or directly to global positions
            if current_segment is not None:
                current_positions.append(row)
            else:
                # Case 1: No segments file present — accumulate every frame
                current_positions.append(row)

            frame_idx += 1
            if processed_count % 500 == 0:
                print(f"  ... Processed {processed_count} frames | Current frame: {frame_idx}/{total_frames} ({(frame_idx/total_frames)*100:.1f}%)")

        # Save any in-progress rally when the video ends mid-segment
        if current_segment is not None:
            rally_list.append({
                "segment": current_segment,
                "positions": current_positions
            })
        elif not segments and current_positions:
            # Case 1 Fallback: Wrap entire video into a single rally [0, total_frames - 1]
            rally_list.append({
                "segment": [0, total_frames - 1],
                "positions": current_positions
            })

        cap.release()
        return rally_list

    def _write_data(self, rally_list, match_name):
        """Write the extracted rally data to ``{match_name}_players.json``.

        The output schema mirrors the original ``player_pass.py`` script so
        downstream pipelines can consume it without modification.
        """
        save_path = os.path.join(self.player_tracks_dir, f"{match_name}_players.json")
        with open(save_path, 'w') as f:
            json.dump({"rally": rally_list}, f, indent=2)
        self.logger.info(f"Saved player positions to {save_path}")


    def _run(self):
        """Entry point for the pipeline.

        It discovers video files, resolves their corresponding homography
        matrices, and drives the per‑video processing loop. Logging provides a
        clear audit trail of skipped files and successful completions.
        """
        video_files = sorted(
            [f for f in os.listdir(self.video_dir) if f.lower().endswith(('.mp4', '.mov', '.avi'))],
            key=self._natural_sort_key
        )
        for video_file in video_files:
            match_name = os.path.splitext(video_file)[0]
            video_path = os.path.join(self.video_dir, video_file)
            H = self.matrices.get(match_name)
            if H is None:
                self.logger.warning(f"No homography matrix for {video_file}, skipping.")
                continue
            cap = self._setup_capture_header(video_path)
            if cap is None:
                continue
            total_frames = self._total_frames(cap)
            segments = self._get_segments(match_name)
            rally_list = self._loop(cap, segments, total_frames, H)
            self._write_data(rally_list, match_name)
        
        self.logger.info("Player positions extraction completed for all videos.")

            

        
