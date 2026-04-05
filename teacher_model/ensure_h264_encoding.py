import os
import subprocess
from tqdm import tqdm #type: ignore

VIDEO_DIR = '../dataset/videos'  

def get_codec(filepath):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=codec_name',
         '-of', 'default=noprint_wrappers=1', filepath],
        capture_output=True, text=True
    )
    output = result.stdout.strip()
    if '=' in output:
        return output.split('=')[1].strip()
    return 'unknown'

def get_duration_seconds(filepath):
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=duration',
         '-of', 'default=noprint_wrappers=1', filepath],
        capture_output=True, text=True
    )
    output = result.stdout.strip()
    try:
        return float(output.split('=')[1].strip())
    except Exception:
        return None

def convert_to_h264(filepath):
    base, _ = os.path.splitext(filepath)
    temp_path = base + '_converting.mp4'
    filename  = os.path.basename(filepath)

    duration = get_duration_seconds(filepath)

    # Run ffmpeg with progress piped to stdout
    process = subprocess.Popen([
        'ffmpeg', '-i', filepath,
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'ultrafast',
        '-c:a', 'copy',
        '-y',
        '-progress', 'pipe:1',   # stream progress to stdout
        '-nostats',               # suppress default stats
        temp_path
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    # Parse ffmpeg progress lines and update tqdm
    pbar = tqdm(
        total=int(duration) if duration else None,
        unit='s',
        desc=f'  {filename}',
        bar_format='{l_bar}{bar}| {n:.0f}/{total}s [{elapsed}<{remaining}]',
        leave=True
    )

    current_time = 0
    for line in process.stdout:
        line = line.strip()
        if line.startswith('out_time_ms='):
            try:
                ms = int(line.split('=')[1])
                new_time = ms // 1_000_000   # microseconds → seconds
                if new_time > current_time:
                    pbar.update(new_time - current_time)
                    current_time = new_time
            except ValueError:
                pass

    process.wait()
    pbar.close()

    if process.returncode != 0:
        tqdm.write(f'  ✗ Failed: {filename}')
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

    os.replace(temp_path, filepath)
    tqdm.write(f'  ✓ Done — {filename}')
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
video_files = sorted([
    f for f in os.listdir(VIDEO_DIR)
    if f.endswith('.mp4') and not f.endswith('_converting.mp4')
])

if not video_files:
    print(f"No .mp4 files found in {VIDEO_DIR}")
    exit()

print(f"Found {len(video_files)} video(s) in {VIDEO_DIR}\n")

already_h264 = []
to_convert   = []

print("Checking codecs...")
for filename in tqdm(video_files, desc='Scanning', unit='file'):
    filepath = os.path.join(VIDEO_DIR, filename)
    codec    = get_codec(filepath)
    if codec == 'h264':
        already_h264.append(filename)
    else:
        to_convert.append((filename, codec))

print(f"\n  {len(already_h264)} already H264 — skipping")
for f in already_h264:
    print(f"    ✓ {f}")

print(f"\n  {len(to_convert)} need conversion:")
for f, c in to_convert:
    print(f"    • {f}  [{c.upper()}]")

if not to_convert:
    print("\nNothing to do.")
    exit()

print(f"\nConverting {len(to_convert)} file(s)...\n")

success = []
failed  = []

overall = tqdm(to_convert, desc='Overall', unit='video', position=0)
for filename, codec in overall:
    overall.set_postfix(file=filename)
    filepath = os.path.join(VIDEO_DIR, filename)
    ok = convert_to_h264(filepath)
    if ok:
        success.append(filename)
    else:
        failed.append(filename)

print(f"\n{'─'*50}")
print(f"✓ Converted : {len(success)}")
print(f"✗ Failed    : {len(failed)}")
if failed:
    print("Failed files:")
    for f in failed:
        print(f"  {f}")