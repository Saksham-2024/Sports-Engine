import numpy as np
import pandas as pd

data = np.load('4.2-features_tensor.npz', allow_pickle=True)
df = pd.read_csv('4.1-engineered_dataset.csv')

X_original = data['X']
y_original = data['y']
feature_names = data['feature_names']
classes = data['classes']

x_indices = [i for i, name in enumerate(feature_names) if name.endswith('_x')]
x_cols = [col for col in df.columns if col.endswith('_x')]

print(f"Found {len(x_cols)} X-coordinate columns to mirror (csv).")
print(f"Found {len(x_indices)} X-coordinate features to mirror (npz).")

X_mirrored = X_original.copy()
X_mirrored[:, :, x_indices] = X_mirrored[:, :, x_indices] * -1

X_doubled = np.concatenate((X_original, X_mirrored), axis=0)
y_doubled = np.concatenate((y_original, y_original), axis=0) # Labels stay exactly the same!

df_mirrored = df.copy()
df_mirrored[x_cols] = df_mirrored[x_cols] * -1
df_mirrored['match_no'] = df_mirrored['match_no'] + 1000 
df_final = pd.concat([df, df_mirrored], ignore_index=True)
df_final.to_csv('5.1-engineered_dataset_augmented.csv', index=False)

print(f"New Doubled Dataset (npz) Shape: {X_doubled.shape}")
print(f'New CSV size: {len(df_final)}')

np.savez_compressed('5.2-features_tensor_doubled.npz', 
                    X=X_doubled, 
                    y=y_doubled, 
                    feature_names=feature_names, 
                    classes=classes)

print("Saved as '5.1-engineered_dataset_augmented.csv' and '5.2-features_tensor_doubled.npz'")
