"""
STEP 2: Normalize to [0, 1] with NaN-Mask Awareness

Strategy:
  - Normalize valid (non-NaN) values to [0, 1]
  - Keep NaN as NaN (sentinel for "unknown")
  - Create binary masks for NaN positions
  - Output: normalized CSV + mask metadata
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
import yaml

with open('../../configs/configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

class BadmintonNormalizer:
    """
    Normalize badminton data to [0, 1] preserving NaN/masks.
    
    Bounds (from dataset analysis):
      X: [-1, 6.5] meters
      Y: [-1, 14.5] meters
      Z: [0, 9.95] meters (after cleaning)
    """
    
    def __init__(self,
                 x_bounds=(-1, 6.5),
                 y_bounds=(-1, 14.5),
                 z_bounds=(0, 9.95)):
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.z_min, self.z_max = z_bounds
        
        self.bounds = {
            'x': x_bounds,
            'y': y_bounds,
            'z': z_bounds
        }
        
        self.stats = {
            'rows_processed': 0,
            'coords_normalized': 0,
            'nans_preserved': 0,
            'clipped_values': 0
        }
    
    def normalize(self, value, coord_type):
        """
        Normalize single value to [0, 1].
        
        Args:
            value: float or np.nan
            coord_type: 'x', 'y', or 'z'
            
        Returns:
            normalized value in [0, 1] or np.nan
        """
        if pd.isna(value):
            return np.nan
        
        if coord_type == 'x':
            min_v, max_v = self.x_min, self.x_max
        elif coord_type == 'y':
            min_v, max_v = self.y_min, self.y_max
        elif coord_type == 'z':
            min_v, max_v = self.z_min, self.z_max
        else:
            raise ValueError(f"Unknown coord_type: {coord_type}")
        
        # Clip to bounds (handles any remaining outliers)
        clipped = np.clip(value, min_v, max_v)
        
        # Normalize to [0, 1]
        normalized = (clipped - min_v) / (max_v - min_v)
        
        return normalized
    
    def process_dataframe(self, df):
        """
        Normalize entire dataframe.
        
        Args:
            df: DataFrame with cleaned coordinates
            
        Returns:
            df_normalized: with values in [0, 1] or NaN
            mask_data: dict tracking NaN masks per agent
        """
        df_norm = df.copy()
        
        # Define coordinate columns and their types
        coord_cols = {
            'p1_x': 'x', 'p1_y': 'y', 'p1_z': 'z',
            'p2_x': 'x', 'p2_y': 'y', 'p2_z': 'z',
            'shuttle_x': 'x', 'shuttle_y': 'y', 'shuttle_z': 'z'
        }
        
        # Normalize each coordinate
        for col, coord_type in coord_cols.items():
            df_norm[col] = df[col].apply(lambda v: self.normalize(v, coord_type))
            
            # Count
            non_nan = df[col].notna().sum()
            self.stats['coords_normalized'] += non_nan
            self.stats['nans_preserved'] += df[col].isna().sum()
        
        self.stats['rows_processed'] = len(df)
        
        return df_norm, self.bounds
    
    def validate_normalized(self, df_norm):
        """
        Validate normalized data is in [0, 1] or NaN.
        
        Returns:
            validation report
        """
        coord_cols = [
            'p1_x', 'p1_y', 'p1_z',
            'p2_x', 'p2_y', 'p2_z',
            'shuttle_x', 'shuttle_y', 'shuttle_z'
        ]
        
        report = {
            'valid': True,
            'violations': [],
            'coord_stats': {}
        }
        
        for col in coord_cols:
            col_valid = df_norm[col].dropna()
            
            if len(col_valid) == 0:
                report['coord_stats'][col] = {
                    'min': None, 'max': None, 'count': 0, 'nans': len(df_norm) - 0
                }
                continue
            
            min_v = col_valid.min()
            max_v = col_valid.max()
            
            report['coord_stats'][col] = {
                'min': float(min_v),
                'max': float(max_v),
                'count': int(len(col_valid)),
                'nans': int(df_norm[col].isna().sum())
            }
            
            # Check bounds
            if min_v < -0.01 or max_v > 1.01:
                report['valid'] = False
                report['violations'].append(
                    f"{col}: out of [0,1] (min={min_v:.4f}, max={max_v:.4f})"
                )
        
        return report


def main():
    """Main normalization pipeline."""
    
    input_file = os.path.join(configs['global']['project_root'], configs['tranSPORTmer']['data']['cleaned_csv'])
    output_file = os.path.join(configs['global']['project_root'], configs['tranSPORTmer']['data']['normalized_csv'])
    stats_file = os.path.join(configs['global']['project_root'], configs['tranSPORTmer']['data']['normalization_stats'])
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("=" * 80)
    print("STEP 2: NORMALIZATION TO [0, 1]")
    print("=" * 80)
    
    # Load cleaned data
    print(f"\n[1/4] Loading cleaned data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"      Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Normalize
    print(f"\n[2/4] Normalizing coordinates to [0, 1]...")
    normalizer = BadmintonNormalizer()
    df_norm, bounds = normalizer.process_dataframe(df)
    
    print(f"      Processed {normalizer.stats['rows_processed']:,} rows")
    print(f"      Normalized coords: {normalizer.stats['coords_normalized']:,}")
    print(f"      Preserved NaNs: {normalizer.stats['nans_preserved']:,}")
    
    # Validate
    print(f"\n[3/4] Validating normalized data...")
    validation = normalizer.validate_normalized(df_norm)
    
    if validation['valid']:
        print(f"      ✓ All coordinates in valid [0, 1] range")
    else:
        print(f"      ✗ VALIDATION FAILED:")
        for violation in validation['violations']:
            print(f"        - {violation}")
    
    print(f"\n      Coordinate statistics:")
    for coord, stats_coord in sorted(validation['coord_stats'].items()):
        if stats_coord['count'] > 0:
            print(f"        {coord:15} | [{stats_coord['min']:.4f}, {stats_coord['max']:.4f}] "
                  f"| N={stats_coord['count']:8,} | NaN={stats_coord['nans']:,}")
        else:
            print(f"        {coord:15} | All NaN | N=0 | NaN={stats_coord['nans']:,}")
    
    # Save
    print(f"\n[4/4] Saving normalized data to {output_file}...")
    df_norm.to_csv(output_file, index=False)
    print(f"      ✓ Saved {len(df_norm):,} rows")
    
    # Save metadata
    stats_combined = {
        'bounds': bounds,
        'rows_processed': int(normalizer.stats['rows_processed']),
        'coords_normalized': int(normalizer.stats['coords_normalized']),
        'nans_preserved': int(normalizer.stats['nans_preserved']),
        'validation': validation
    }
    with open(stats_file, 'w') as f:
        json.dump(stats_combined, f, indent=2)
    print(f"      ✓ Stats saved to {stats_file}")
    
    print("\n" + "=" * 80)
    print("✓ NORMALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nNext step: 03_create_training_windows.py")


if __name__ == '__main__':
    main()
