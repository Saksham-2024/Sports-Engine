import pandas as pd
import subprocess
import os
import json
import time
import yt_dlp

REGISTRY_FILE = 'match_name_no_registry.csv'
VIDEO_DIR = '/home/saksham/projects and programming/BTech_Project/dataset/videos/'

def search_youtube(match_name):
    clean_name = match_name.replace('combined_master_', '').replace('.csv', '').replace('_', ' ')
    search_query = f"{clean_name} full match"
    print(f"  🔍 Searching: {search_query}")
    
    # We keep the search part in subprocess because it successfully dumps the 5 results to JSON
    command = [
        'yt-dlp', 
        f'ytsearch5:{search_query}', 
        '--dump-json', 
        '--no-playlist', 
        '--quiet'
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output_lines = result.stdout.strip().split('\n')
        
        videos = []
        for line in output_lines:
            if line:
                videos.append(json.loads(line))
                
        if not videos:
            print("  ❌ No videos found at all.")
            return ""

        # PASS 1: Look for Official Channels
        official_keywords = ['bwf', 'badminton europe', 'badmintonworld', 'astro arena', 'spotv']
        for v in videos:
            uploader = v.get('uploader', '').lower()
            if any(keyword in uploader for keyword in official_keywords):
                print(f"  🌟 Official Video Found! Channel: {v.get('uploader')}")
                return v.get('webpage_url', '')

        # PASS 2: If official is blocked, find ANY video longer than 30 minutes (1800 seconds)
        for v in videos:
            duration = v.get('duration', 0)
            if duration > 1800:
                print(f"  ⚠️ Official blocked. Using full match from fan channel: {v.get('uploader')}")
                return v.get('webpage_url', '')

        print("  ❌ Only short clips/highlights found in top 5.")
        return ""
            
    except Exception as e:
        print(f"  ❌ Search failed: {e}")
        return ""

def process_registry():
    os.makedirs(VIDEO_DIR, exist_ok=True)
    
    # Ignore the space in "name, match_id"
    df = pd.read_csv(REGISTRY_FILE, skipinitialspace=True)
    
    # Create the missing columns so row['youtube_link'] doesn't crash
    if 'youtube_link' not in df.columns:
        df['youtube_link'] = ""
    if 'local_video_path' not in df.columns:
        df['local_video_path'] = ""
        
    updated = True 
    
    for index, row in df.iterrows():
        match_no = row['match_id']
        match_name = row['name']
        
        # GUARANTEED SAFE NAMING: Maps exactly to the CSV ID
        local_path = f"{VIDEO_DIR}match_{match_no}.mp4"
        
        # Save the local path to the dataframe
        df.at[index, 'local_video_path'] = local_path
        
        youtube_link = row['youtube_link']
        
        print(f"\n--- Processing Match {match_no} ---")
        
        # 1. FIND LINK IF MISSING
        if pd.isna(youtube_link) or str(youtube_link).strip() == "":
            youtube_link = search_youtube(match_name)
            if youtube_link:
                print(f"  ✅ Found URL: {youtube_link}")
                df.at[index, 'youtube_link'] = youtube_link
                time.sleep(3) # Prevent IP ban from YouTube
            else:
                continue

        # 2. DOWNLOAD VIDEO
        if os.path.exists(local_path):
            print(f"  ⏭️ Already downloaded: {local_path}")
            continue
            
        print(f"  ⬇️ Downloading to {local_path}...")
        
        # Native yt_dlp implementation
        ydl_opts = {
            'format': 'bestvideo[height<=1080][ext=mp4]',
            'outtmpl': local_path,  # Uses the exact match_no path
            'noplaylist': True,
            'merge_output_format': 'mp4',
            'quiet': False          # Set to True if you want a cleaner terminal
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_link])
            print("  🎉 Download complete!")
        except Exception as e:
            print(f"  ❌ Download failed: {e}")

    # Save the updated links back to the CSV
    if updated:
        df.to_csv(REGISTRY_FILE, index=False)
        print(f"\n✅ Updated links saved to {REGISTRY_FILE}")

if __name__ == "__main__":
    process_registry()