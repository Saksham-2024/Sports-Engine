import pandas as pd

INPUT_FILE = 'filtered_data.csv'
OUTPUT_FILE = 'clean_filtered_data.csv'

STROKE_MERGE_DICT = {
    'lob': 'Lift',
    'lift': 'Lift',
    'defensive return lob': 'Lift',
    'push': 'Drive',
    'rush': 'Drive',
    'push/rush': 'Drive',
    'drive': 'Drive',
    'clear': 'Clear',
    'drop': 'Dropshot',
    'smash': 'Smash',
    'crosscourt net': 'Net-Shot',
    'net shot': 'Net-Shot',
    'wrist smash': 'Net-Kill',         
    'defensive shot': 'Block',
    '發長球': 'Serve',
    '發短球': 'Serve',
    'net lift': 'Net-Lift',
    '防守回挑': 'Net-Lift',
    '防守回抽': 'Net-Lift',
    'serve': 'Serve',
    'block': 'Block',
    'short service': 'Serve',

}

ALLOWED_STROKES = ['Block', 'Clear', 'Drive', 'Dropshot', 'Net-Kill', 'Net-Lift', 'Net-Shot', 'Serve', 'Smash', 'Lift']

def clean_and_filter_data():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    initial_rows = len(df)
    
    df['stroke_type_raw'] = df['stroke_type']
    
    for i in range(len(df)):
        if df.loc[i, 'stroke_num'] == 1:
            df.loc[i, 'stroke_type'] = 'Serve'
            
    df['stroke_type'] = df['stroke_type'].astype(str).str.lower().str.strip()    
    
    df['stroke_type'] = df['stroke_type'].replace(STROKE_MERGE_DICT)
    
    df_filtered = df[df['stroke_type'].isin(ALLOWED_STROKES)].copy()
    
    final_rows = len(df_filtered)
    dropped_rows = initial_rows - final_rows
    
    dropped_df = df[~df['stroke_type'].isin(ALLOWED_STROKES)]
    if not dropped_df.empty:
        print("\n⚠️ WARNING: The following raw labels were dropped:")
        print(dropped_df['stroke_type_raw'].value_counts())
    
    df_filtered.drop(columns=['stroke_type_raw'], errors='ignore').to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n--- Filter Report ---")
    print(f"Starting Rows: {initial_rows}")
    print(f"Dropped Rows:  {dropped_rows}")
    print(f"Final Rows:    {final_rows}")
    print(f"\nFinal Class Distribution:")
    print(df_filtered['stroke_type'].value_counts())
    print(f"✅ Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_and_filter_data()