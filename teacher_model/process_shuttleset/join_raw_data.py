import pandas as pd
import glob
import os
import re

# Base paths
BASE_DIR = '/home/saksham/projects and programming/BTech_Project/dataset/ShuttleSet/set'
OUTPUT_DIR = './combined_master_csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# The "Gold Standard" Badminton Translation Dictionary
TRANSLATION_DICT = {
    # Stroke Types (from image)
    '發球': 'short service',
    '長球': 'clear',
    '推球': 'push/rush',
    '殺球': 'smash',
    '擋小球': 'defensive shot',
    '平球': 'drive',
    '放小球': 'net shot',
    '挑球': 'lob',
    '切球': 'drop',
    '過渡擋球': 'crosscourt net shot',
    '防守回挑': 'net lift',
    
    # Outcomes / Flaws (Kept as a safety net in case these columns exist)
    '出界': 'Out', 
    '掛網': 'Net', 
    '未接': 'Unreturned', 
    '落地': 'Landed',
    '發球失誤': 'Service Error', 
    '掛網失誤': 'Net Error', 
    '出界失誤': 'Out Error',
    '未知': 'Unknown'
}

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

dirs = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

for d in dirs:
    match_dir = os.path.join(BASE_DIR, d)
    csv_files = sorted(glob.glob(os.path.join(match_dir, '*.csv')), key=natural_sort_key)
    
    if not csv_files:
        continue

    print(f"Combining and Translating Match {d}...")
    df_list = [pd.read_csv(file) for file in csv_files]
    master_df = pd.concat(df_list, ignore_index=True)

    # --- 1. THE AGGRESSIVE DICTIONARY PASS ---
    # This applies the translation to EVERY column instantly.
    # If "殺球" appears in 'type', 'win_reason', or a column we didn't expect, it gets translated.
    master_df.replace(TRANSLATION_DICT, inplace=True)

    # --- 2. CUMULATIVE POINT LOGIC ---
    if 'rally' in master_df.columns:
        cumulative_points = []
        current_offset = 0
        last_rally = -1
        
        for r in master_df['rally']:
            try:
                r_int = int(r)
            except (ValueError, TypeError):
                # Fallback for unexpected data types
                cumulative_points.append(r)
                continue
                
            # If the current rally number dips, a new set has started
            if last_rally != -1 and r_int < last_rally:
                current_offset += last_rally
                
            cumulative_points.append(r_int + current_offset)
            last_rally = r_int
            
        master_df['rally'] = cumulative_points

    # Save to output
    output_file = os.path.join(OUTPUT_DIR, f'combined_master_{d}.csv')
    master_df.to_csv(output_file, index=False)
    print(f"✅ Match {d} saved successfully!")

print("All matches processed with precise badminton terminology.")