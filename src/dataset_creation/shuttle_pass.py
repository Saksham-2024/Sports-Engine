import subprocess
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cv2
import torch 
import sys
import yaml

with open('../../configs/configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

CWD         = os.path.dirname(os.path.abspath(__file__))
REPO_PATH   = os.path.join(configs['global']['project_root'], 'src', 'TrackNetV3')
VIDEO_DIR   = os.path.join(configs['global']['project_root'], configs['global']['video_dir'])
OUTPUT_PATH = os.path.join(configs['global']['project_root'], configs['dataset_creation']['shuttle_tracks_dir'])
SEGMENTS_PATH = os.path.join(configs['global']['project_root'], configs['dataset_creation']['segments_dir'])

# Detect available GPUs
def get_available_gpus():
    """Returns list of GPU IDs available for use."""
    try:
        import torch
        count = torch.cuda.device_count()
        return list(range(count))
    except:
        # Fallback: assume single GPU
        return [0]

if torch.cuda.is_available():
    AVAILABLE_GPUS = get_available_gpus()
    print(f"🚀 Detected {len(AVAILABLE_GPUS)} GPU(s): {AVAILABLE_GPUS}")
else:
    print("❌ No GPU detected. Running on CPU.")
    AVAILABLE_GPUS = []

WORKERS_PER_GPU = 4
MAX_PARALLEL_MATCHES = len(AVAILABLE_GPUS) * WORKERS_PER_GPU if len(AVAILABLE_GPUS) > 0 else 1

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def run_tracknet(video_path, save_dir, gpu_id, thread_id):
    """Run TrackNet iteratively per segment on a specific GPU. Returns (success, match_stem, stderr)."""
    os.makedirs(save_dir, exist_ok=True)
    match_stem = os.path.splitext(os.path.basename(video_path))[0]
    
    segment_file = os.path.join(SEGMENTS_PATH, f"{match_stem}.json")
    print(segment_file)
    segments = []
    if os.path.exists(segment_file):
        with open(segment_file, 'r') as f:
            data = json.load(f)
            segments = data.get("segments", [])
            
    if not segments:
        # Write an empty prediction file to satisfy missing dependencies downstream
        import pandas as pd
        segment_dir = os.path.join(save_dir, "segment1")
        os.makedirs(segment_dir, exist_ok=True)
        df = pd.DataFrame(columns=['Frame', 'Visibility', 'X', 'Y'])
        df.to_csv(os.path.join(segment_dir, f'{match_stem}_ball.csv'), index=False)
        return True, match_stem, "Skipped due to empty segments file"

    env = os.environ.copy()
    if gpu_id != -1:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id) # 🔑 Assign GPU

    # Global median array saved safely out of the way for re-use
    median_file = os.path.join(save_dir, "metadata", f"{match_stem}_median.npz")
    
    python_exe = os.path.join(REPO_PATH, "tracknet", "bin", "python")
    for seg_index, seg in enumerate(tqdm(segments, desc=f"{match_stem} (GPU {gpu_id})", position=thread_id, leave=False)):
        seg_dir = os.path.join(save_dir, f"segment{seg_index+1}")
        os.makedirs(seg_dir, exist_ok=True)
        
        command = [
            python_exe, "predict.py",
            "--video_file", video_path,
            "--tracknet_file", "ckpts/TrackNet_best.pt",
            "--inpaintnet_file", "ckpts/InpaintNet_best.pt",
            "--save_dir", seg_dir,
            "--batch_size", "8", 
            "--large_video",
            "--video_range", f"{seg[0]},{seg[1]}",
            "--median_file", median_file
        ]

        try:
            subprocess.run(
                command,
                cwd=REPO_PATH,
                check=True,
                capture_output=True,
                text=True,
                env=env
            )
        except subprocess.CalledProcessError as e:
            return False, match_stem, f"Failed at segment {seg_index+1}: {e.stderr}"

    return True, match_stem, None


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    match_vids = sorted(
        [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))],
        key=natural_sort_key
    )

    print(f" Found {len(match_vids)} matches to process")
    print(f"  Processing up to {MAX_PARALLEL_MATCHES} matches in parallel\n")
    print("\n" * MAX_PARALLEL_MATCHES)  # Reserve terminal layout slots for concurrent tqdm bars

    # Match GPU IDs in round-robin fashion
    gpu_cycle = iter(AVAILABLE_GPUS * (len(match_vids) // len(AVAILABLE_GPUS) + 1)) if len(AVAILABLE_GPUS) > 0 else iter([-1] * len(match_vids))

    tasks = {}  # Map (future -> match info) for progress tracking
    
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_MATCHES) as executor:
        
        # Submit all matches
        for thread_id, m in enumerate(match_vids):
            match_stem = os.path.splitext(m)[0]
            match_no = re.search(r'\d+', match_stem).group()
            video_path = os.path.join(VIDEO_DIR, m)
            save_tracks_dir = os.path.join(OUTPUT_PATH, match_stem)
            
            gpu_id = next(gpu_cycle)
            
            # Simple resume logic: skip if last segment exists
            # We don't know the exact count, but if segment1 exists, it's likely already processed or interrupted
            # Better: skip only if the folder exists and is not empty
            if os.path.exists(save_tracks_dir) and any(os.path.isdir(os.path.join(save_tracks_dir, d)) for d in os.listdir(save_tracks_dir) if d.startswith("segment")):
                print(f" Skipping {match_stem} (already processed)")
                continue

            # Position the progress bar based on logical thread slot
            slot_id = thread_id % MAX_PARALLEL_MATCHES
            future = executor.submit(run_tracknet, video_path, save_tracks_dir, gpu_id, slot_id)
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


if __name__ == "__main__":
    main()