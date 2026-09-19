import pandas as pd
import subprocess
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch 
import sys
from src.core.utils.configs import PathResolver
from src.core.utils.logger import get_logger

class ShuttleDetector:
    def __init__(self):
        self.resolver = PathResolver()
        self.tracknet_path = self.resolver.get_path("models.tracknet")

    def _command(self, python_exe, video_path, seg_dir, seg, median_file, env):
        command = [
            python_exe, "predict.py",
            "--video_file", video_path,
            "--tracknet_file", "ckpts/TrackNet_best.pt",
            "--inpaintnet_file", "ckpts/InpaintNet_best.pt",
            "--save_dir", seg_dir,
            "--batch_size", str(self.resolver.get_config("tracknet.inference.batch_size")), 
            "--large_video",
            "--video_range", f"{seg[0]},{seg[1]}",
            "--median_file", median_file
        ]

        try:
            subprocess.run(
                command,
                cwd=self.tracknet_path,
                check=True,
                capture_output=True,
                text=True,
                env=env
            )
        except subprocess.CalledProcessError as e:
            return False, match_stem, f"Failed at segment {seg_index+1}: {e.stderr}"
    
    def run_tracknet_command(self, video_path, save_dir, gpu_id, thread_id):
        os.makedirs(save_dir, exist_ok=True)
        match_stem = os.path.splitext(os.path.basename(video_path))[0]
        
        # csv validation for segments file 
        segment_file = os.path.join(self.resolver.get_path("dataset_creation.segments_dir"), f"{match_stem}.json")
        segments = []
        if os.path.exists(segment_file):
            with open(segment_file, 'r') as f:
                data = json.load(f)
                segments = data.get("segments", [])
                
        if not segments:
            # Write an empty prediction file to satisfy missing dependencies downstream
            segment_dir = os.path.join(save_dir, "segment1")
            os.makedirs(segment_dir, exist_ok=True)
            df = pd.DataFrame(columns=['Frame', 'Visibility', 'X', 'Y'])
            df.to_csv(os.path.join(segment_dir, f'{match_stem}_ball.csv'), index=False)
            return True, match_stem, "Skipped due to empty segments file"

        env = os.environ.copy()
        if gpu_id != -1:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # Global median array saved safely out of the way for re-use
        median_file = os.path.join(save_dir, "metadata", f"{match_stem}_median.npz")
        
        python_exe = os.path.join(self.tracknet_path, "bin", "python")
        for seg_index, seg in enumerate(tqdm(segments, desc=f"{match_stem} (GPU {gpu_id})", position=thread_id, leave=False)):
            seg_dir = os.path.join(save_dir, f"segment{seg_index+1}")
            os.makedirs(seg_dir, exist_ok=True)
            
            self._command(python_exe, video_path, seg_dir, seg, median_file, env)

        return True, match_stem, None
    
class ShuttleExtractionPipeline:
    def __init__(self):
        self.resolver = PathResolver()
        self.logger = get_logger(__name__)
        self.video_dir = self.resolver.get_path("global.video_dir")
        self.segments_dir = self.resolver.get_path("dataset_creation.segments_dir")
        self.shuttle_tracks_dir = self.resolver.get_path("dataset_creation.shuttle_tracks_dir")
        self.max_parallel_matches = self.resolver.get_config("dataset_creation.max_parallel_matches")
        os.makedirs(self.shuttle_tracks_dir, exist_ok=True)
    
    def _natural_sort_key(self, s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    def get_available_gpus(self):
        """Returns list of GPU IDs available for use."""
        try:
            import torch
            count = torch.cuda.device_count()
            return list(range(count))
        except:
            # Fallback: assume single GPU
            return [0]

    def run_extraction(self):
        match_vids = sorted(
            [f for f in os.listdir(self.video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))],
            key=self._natural_sort_key
        )
        if torch.cuda.is_available():
            AVAILABLE_GPUS = self.get_available_gpus()
            print(f"🚀 Detected {len(AVAILABLE_GPUS)} GPU(s): {AVAILABLE_GPUS}")
        else:
            print("❌ No GPU detected. Running on CPU.")
            AVAILABLE_GPUS = []
        
        gpu_cycle = iter(AVAILABLE_GPUS * (len(match_vids) // len(AVAILABLE_GPUS) + 1)) if len(AVAILABLE_GPUS) > 0 else iter([-1] * len(match_vids))
        tasks = {}  # Map (future -> match info) for progress tracking
        shuttle_detector = ShuttleDetector()
        with ThreadPoolExecutor(max_workers=self.max_parallel_matches) as executor:
            
            # Submit all matches
            for thread_id, m in enumerate(match_vids):
                match_stem = os.path.splitext(m)[0]
                match_no = re.search(r'\d+', match_stem).group()
                video_path = os.path.join(self.video_dir, m)
                save_tracks_dir = os.path.join(self.shuttle_tracks_dir, match_stem)
                
                gpu_id = next(gpu_cycle)
                
                if os.path.exists(save_tracks_dir) and any(os.path.isdir(os.path.join(save_tracks_dir, d)) for d in os.listdir(save_tracks_dir) if d.startswith("segment")):
                    print(f" Skipping {match_stem} (already processed)")
                    continue

                # Position the progress bar based on logical thread slot
                slot_id = thread_id % self.max_parallel_matches
                future = executor.submit(shuttle_detector.run_tracknet_command,video_path=video_path,
                    save_dir=save_tracks_dir,
                    gpu_id=gpu_id,
                    thread_id=slot_id
                )
                tasks[future] = (match_stem, gpu_id)
            
            # Track completion
            completed = 0
            failed = []
            
            for future in as_completed(tasks):
                match_stem, gpu_id = tasks[future]
                success, returned_stem, stderr = future.result()
                completed += 1
                
                if success:
                    msg = f"✓ [{completed}/{len(match_vids)}] {match_stem} (GPU {gpu_id}) — Done"
                    if stderr: msg += f" ({stderr})"
                    tqdm.write(msg)
                else:
                    tqdm.write(f"✗ [{completed}/{len(match_vids)}] {match_stem} (GPU {gpu_id}) — FAILED\n  Error: {stderr}")
                    failed.append((match_stem, stderr))
            
            if failed:
                print(f"\n  {len(failed)} match(es) failed:")
                for match_stem, stderr in failed:
                    print(f"  - {match_stem}")
                    if stderr:
                        print(f"    Error: {stderr[:200]}")
        
        print("\n All matches processed!")
        if not failed:
            print("✓ No failures.")
