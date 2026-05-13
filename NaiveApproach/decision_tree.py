import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import seaborn as sns
import yaml

# Load Config
with open('configs.yaml', 'r') as f:
    config = yaml.safe_load(f)

SEED = config['hyperparameters']['general']['seed']
TEST_SIZE = config['hyperparameters']['decision_tree']['test_size']

# Decision Tree for shot selection based on Player Positions on a Badminton Court
df = pd.read_csv(config['files']['features_final'])

cols = ["player1_pos", "player2_pos", "player1_x", "player1_y", "player2_x", "player2_y", "dist_players",
         "dist_player1_net", "dist_player2_net", "disp_p1_center_dx", "disp_p1_center_dy", "disp_p2_center_dx",
         "disp_p2_center_dy", "dist_p1_center", "dist_p2_center", "vel_p1_dx", "vel_p1_dy" ,"vel_p2_dx" ,"vel_p2_dy",
         "shuttle_hit_from", "shuttle_hit_to", "stroke_type", "prev_stroke_type"]

encoders = {}
for col in ["player1_pos", "player2_pos", "stroke_type", "prev_stroke_type", "shuttle_hit_from", "shuttle_hit_to"]:    
    le = LabelEncoder() 
    df[col + "_enc"] = le.fit_transform(df[col].astype(str).fillna("NA"))
    encoders[col] = le

df.dropna()
features_without_context = ["player1_pos_enc", "player2_pos_enc", "player1_x", "player1_y", "player2_x", "player2_y",
                            "dist_players", "dist_player1_net", "dist_player2_net", "disp_p1_center_dx", "disp_p1_center_dy",
                            "disp_p2_center_dx", "disp_p2_center_dy", "dist_p1_center", "dist_p2_center", "vel_p1_dx", "vel_p1_dy",
                            "vel_p2_dx" ,"vel_p2_dy"]

features_with_context = features_without_context + ["prev_stroke_type_enc" , "shuttle_hit_from_enc", "shuttle_hit_to_enc"]
y = df["stroke_type_enc"]

X_no_context = df[features_without_context]
X_with_context = df[features_with_context]

vc = y.value_counts()
stratify_var = y if vc.min() >= 2 else None

X_train_no_context, X_test_no_context, y_train, y_test = train_test_split(
    X_no_context, y, test_size=TEST_SIZE, random_state=SEED, stratify=stratify_var
)

X_train_with_context, X_test_with_context, y_train_with_context, y_test_with_context = train_test_split(
    X_with_context, y, test_size=TEST_SIZE, random_state=SEED, stratify=stratify_var
)

if stratify_var is None:
    print("Warning: not stratifying because some classes have < 2 samples")

models_without_context = {
    "DecisionTree": DecisionTreeClassifier(
        max_depth=12, random_state=SEED, class_weight="balanced",
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=SEED,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=SEED
    )
}

models_with_context = {
    "DecisionTree": DecisionTreeClassifier(
        max_depth=12, random_state=SEED, class_weight="balanced",
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=SEED,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=SEED
    )
}

for name, model in models_without_context.items():
    model.fit(X_train_no_context, y_train)
    y_pred = model.predict(X_test_no_context)

    acc = accuracy_score(y_test, y_pred)
    print(f"Model: {name} (Without Context)")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoders["stroke_type"].classes_))
    print("Confusion Matrix:")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=encoders["stroke_type"].classes_,
        yticklabels=encoders["stroke_type"].classes_
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

for name, model in models_with_context.items():
    model.fit(X_train_with_context, y_train_with_context)
    y_pred = model.predict(X_test_with_context)

    acc = accuracy_score(y_test_with_context, y_pred)
    print(f"Model: {name} (With Context)")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test_with_context, y_pred, target_names=encoders["stroke_type"].classes_))
    print("Confusion Matrix:")
    
    cm = confusion_matrix(y_test_with_context, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=encoders["stroke_type"].classes_,
        yticklabels=encoders["stroke_type"].classes_
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

model_save_path = config['paths']['model_dir']
os.makedirs(model_save_path, exist_ok=True)

joblib.dump(models_without_context["DecisionTree"], os.path.join(model_save_path, config['models']['decision_tree']))
joblib.dump(models_without_context["RandomForest"], os.path.join(model_save_path, config['models']['random_forest']))
joblib.dump(models_without_context["XGBoost"], os.path.join(model_save_path, config['models']['xgboost']))

joblib.dump(models_with_context["DecisionTree"], os.path.join(model_save_path, config['models']['decision_tree_with_context']))
joblib.dump(models_with_context["RandomForest"], os.path.join(model_save_path, config['models']['random_forest_with_context']))
joblib.dump(models_with_context["XGBoost"], os.path.join(model_save_path, config['models']['xgboost_with_context']))

