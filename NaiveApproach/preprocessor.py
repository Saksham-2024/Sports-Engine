import numpy as np
import os
import pandas as pd
import json
import joblib
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('features1.csv')
output_dir = "processed_data/"
os.makedirs(output_dir, exist_ok=True)

categorical_cols = ["player1_pos", "player2_pos", "prev_stroke_type", "shuttle_hit_from", "shuttle_hit_to"]
numeric_cols = [
    'player1_x','player1_y',
    'player2_x','player2_y',
    'dist_players','dist_player1_net','dist_player2_net',
    'disp_p1_center_dx','disp_p1_center_dy',
    'disp_p2_center_dx','disp_p2_center_dy',
    'dist_p1_center','dist_p2_center',
    'vel_p1_dx','vel_p1_dy',
    'vel_p2_dx','vel_p2_dy'
]

cat_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str).fillna("NA"))
    cat_encoders[col] = le

encoded_cat_cols = [col + "_enc" for col in categorical_cols]
features = encoded_cat_cols + numeric_cols + [col + "_mask" for col in categorical_cols] + [col + "_mask" for col in numeric_cols]

df[numeric_cols] = df[numeric_cols].fillna(0)

le = LabelEncoder()
df['shot_label'] = le.fit_transform(df['stroke_type'].astype(str))

df['next_shot_label'] = (
    df.groupby(['match_id','rally_id'])['shot_label']
      .shift(-1)
      .fillna(-100)
      .astype(int)
)

def build_sequences(df, feature_cols):
    sequences = []
    for (mid, rid), grp in df.groupby(['match_id','rally_id']):
        grp = grp.sort_values('stroke_num')
        X = grp[feature_cols].to_numpy(dtype=np.float32)
        y = grp['next_shot_label'].to_numpy(dtype=np.int64)
        sequences.append((X, y))
    return sequences

sequences = build_sequences(df, features)

print(f"Total rallies: {len(sequences)}")
print("Example sequence lengths:", [seq[0].shape[0] for seq in sequences[:5]])
print(f"Features used: {features}")

joblib.dump(sequences, output_dir + "sequences.pkl")
joblib.dump(cat_encoders, output_dir + "categorical_encoders.pkl")
joblib.dump(le, output_dir + "shot_label_encoder.pkl")
with open(os.path.join(output_dir, "feature_list.json"), "w") as f:
    json.dump(features, f)

print("EXPORT COMPLETE:")
print(" - sequences.pkl")
print(" - categorical_encoders.pkl")
print(" - shot_label_encoder.pkl")
print(" - feature_list.json")