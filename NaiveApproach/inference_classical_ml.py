import pandas as pd
import joblib
import random
from sklearn.preprocessing import LabelEncoder

import yaml
import os

# Load Config
with open('configs.yaml', 'r') as f:
    config = yaml.safe_load(f)

src_dir = config['paths']['model_dir']
csv_path = config['files']['features_final']

df = pd.read_csv(csv_path)

models = {
    "DT": joblib.load(os.path.join(src_dir, config['models']['decision_tree'])),
    "RF": joblib.load(os.path.join(src_dir, config['models']['random_forest'])),
    "XGB": joblib.load(os.path.join(src_dir, config['models']['xgboost'])),
    "DT_CTX": joblib.load(os.path.join(src_dir, config['models']['decision_tree_with_context'])),
    "RF_CTX": joblib.load(os.path.join(src_dir, config['models']['random_forest_with_context'])),
    "XGB_CTX": joblib.load(os.path.join(src_dir, config['models']['xgboost_with_context'])),
}

shot_encoder = joblib.load(config['files']['shot_label_encoder'])

no_context_features = [
    'player1_x','player1_y','player2_x','player2_y',
    'dist_players','dist_player1_net','dist_player2_net',
    'disp_p1_center_dx','disp_p1_center_dy',
    'disp_p2_center_dx','disp_p2_center_dy',
    'dist_p1_center','dist_p2_center',
    'vel_p1_dx','vel_p1_dy','vel_p2_dx','vel_p2_dy',
    'player1_pos_enc','player2_pos_enc'
]

context_features = no_context_features + ['prev_stroke_type_enc']

stroke_encoder    = joblib.load(config['files']['stroke_type_encoder'])
prev_stroke_encoder = joblib.load(config['files']['prev_stroke_type_encoder'])
p1_pos_encoder    = joblib.load(config['files']['player1_pos_encoder'])
p2_pos_encoder    = joblib.load(config['files']['player2_pos_encoder'])

df['stroke_type_enc'] = stroke_encoder.transform(
    df['stroke_type'].astype(str)
)

df['prev_stroke_type_enc'] = prev_stroke_encoder.transform(
    df['prev_stroke_type'].astype(str)
)

df['player1_pos_enc'] = p1_pos_encoder.transform(
    df['player1_pos'].astype(str)
)

df['player2_pos_enc'] = p2_pos_encoder.transform(
    df['player2_pos'].astype(str)
)


idx = random.randint(0, len(df) - 1)
row = df.iloc[idx]

print(f"\nSample index: {idx}")
print(f"Current stroke type     : {row['stroke_type']}")
print(f"Previous stroke type    : {row['prev_stroke_type']}\n")

for name, model in models.items():
    if "CTX" in name:
        X = row[context_features].values.reshape(1, -1)
    else:
        X = row[no_context_features].values.reshape(1, -1)

    pred_id = model.predict(X)[0]
    pred_label = shot_encoder.inverse_transform([pred_id])[0]

    print(f"{name:<8} → {pred_label}")
