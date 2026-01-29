import pandas as pd
df = pd.read_csv('features.csv')

for index, row in df.iterrows():
    if isinstance(row['stroke_type'], str) and "-Bh" in row['stroke_type']:
        df.loc[index, 'stroke_type'] = row['stroke_type'].replace("-Bh", "")
    
    if isinstance(row['prev_stroke_type'], str) and "-Bh" in row['prev_stroke_type']:
        df.loc[index, 'prev_stroke_type'] = row['prev_stroke_type'].replace("-Bh", "")
    
df.to_csv('features1.csv', index=False)

