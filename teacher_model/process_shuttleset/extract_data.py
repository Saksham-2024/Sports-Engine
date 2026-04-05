import pandas as pd
import os

# Configuration
INPUT_CSV_DIR = './combined_master_csv/'  
OUTPUT_CSV_NAME = 'filtered_data.csv'
REGISTRY_FILE = 'match_name_no_registry.csv'

def process_shuttleset_data():
    # 1. LOAD THE REGISTRY
    try:
        # skipinitialspace handles the "name, match_id" leading space bug
        reg_df = pd.read_csv(REGISTRY_FILE, skipinitialspace=True)
        # Sort by match_id just to be 100% mathematically certain it goes 1, 2, 3...
        reg_df = reg_df.sort_values(by='match_id')
        print(f"✅ Loaded registry with {len(reg_df)} assigned match IDs.")
    except Exception as e:
        print(f"❌ Failed to load registry file: {e}")
        return

    all_extracted_data = []

    # 2. PROCESS FILES IN STRICT REGISTRY ORDER
    for index, row in reg_df.iterrows():
        match_id = row['match_id']
        filename = row['name']
        file_path = os.path.join(INPUT_CSV_DIR, filename)
        
        # Check if this file actually exists in your folder before opening
        if not os.path.exists(file_path):
            print(f"  ⚠️ Skipping Match {match_id}: {filename} (File not found!)")
            continue
            
        print(f"Processing Match {match_id}: {filename}...")
        
        df = pd.read_csv(file_path)
        mapped_df = pd.DataFrame(index=df.index)

        # 3. ASSIGN THE PERMANENT ID
        mapped_df['match_no'] = match_id

        # 4. DIRECT MAPPINGS
        mapped_df['point_no'] = 'Point' + df['rally'].astype(str)
        mapped_df['stroke_num'] = df['ball_round'].astype(int)
        mapped_df['stroke_type'] = df['type']
        
        # 5. FRAME SEQUENCING (The LSTM conversion)
        mapped_df['stroke_begin'] = df['frame_num'].astype(int) - 15
        mapped_df['stroke_end'] = df['frame_num'].astype(int) + 14
        mapped_df.loc[mapped_df['stroke_begin'] < 0, 'stroke_begin'] = 0

        # 6. PLAYING SIDE
        mapped_df['playing_side'] = df['player'].map({'A': 1, 'B': 2})
        mapped_df['playing_side'] = mapped_df['playing_side'].fillna(1).astype(int)

        # 7. CAMERA ANGLE
        mapped_df['camera'] = 'normal' 

        all_extracted_data.append(mapped_df)

    if not all_extracted_data:
        print("❌ No data was extracted. Check your folder path and registry.")
        return

    # Combine all processed files
    final_df = pd.concat(all_extracted_data, ignore_index=True)
    
    # Sort the final dataframe to keep rows perfectly chronological
    final_df['_sort_key'] = final_df['point_no'].str.replace('Point', '', regex=False).astype(int)
    final_df = final_df.sort_values(by=['match_no', '_sort_key', 'stroke_num'])
    final_df = final_df.drop(columns=['_sort_key'])    
    final_df = final_df.dropna(subset=['stroke_end'])
    final_df.to_csv(OUTPUT_CSV_NAME, index=False)
    
    print(f"\n✅ Extraction complete! Saved {len(final_df)} strokes to {OUTPUT_CSV_NAME}")

if __name__ == "__main__":
    process_shuttleset_data()