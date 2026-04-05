import pandas as pd
import os

INPUT_FILE = 'clean_filtered_data.csv'
MERGE_TO_FILE = '../1-timestamps.csv'
MATCHES_TO_KEEP = 12

def merge_datasets():
    if not os.path.exists(MERGE_TO_FILE):
        print(f"❌ Could not find {MERGE_TO_FILE}")
        return
        
    master_df = pd.read_csv(MERGE_TO_FILE)
    max_existing_match = master_df['match_no'].max()
    print(f"Highest existing match number: {max_existing_match}")
    new_df = pd.read_csv(INPUT_FILE)
    new_df = new_df[new_df['match_no'] <= MATCHES_TO_KEEP].copy()
    new_df['match_no'] = new_df['match_no'] + max_existing_match
    new_df.to_csv(MERGE_TO_FILE, mode='a', index=False, header=False)
    
    new_max = new_df['match_no'].max()
    print(f"✅ Successfully appended {len(new_df)} rows to {MERGE_TO_FILE}")
    print(f" New matches are now registered as IDs {max_existing_match + 1} through {new_max}")

if __name__ == "__main__":
    merge_datasets()