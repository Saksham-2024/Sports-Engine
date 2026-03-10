import pandas as pd
import numpy as np

df = pd.read_csv('2-pose_dataset(raw).csv')
feature_cols = [f'{axis}{j}' for j in range(11, 33) for axis in ['x', 'y', 'z', 'v']]

df[feature_cols] = df.groupby(['match_no', 'point_no', 'stroke_num'])[feature_cols].transform(
    lambda group: group.interpolate(method='linear', limit_direction='both')
)

df.to_csv('3-interpolated_dataset(raw).csv', index = False)