import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
import yaml
from src.core.utils.configs import PathResolver
from src.core.utils.logger import get_logger

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


class NormalizeDatasetPipeline:
    """Main normalization pipeline."""
    def __init__(self):
        self.resolver = PathResolver()
        self.logger = get_logger(__name__)
        self.input_file = self.resolver.get_path("tranSPORTmer.data.cleaned_csv")
        self.output_file = self.resolver.get_path("tranSPORTmer.data.normalized_csv")
        self.stats_file = self.resolver.get_path("tranSPORTmer.data.normalization_stats")
    
    
    def _run(self):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        self.logger.info("Normalization to [0, 1]")
        
        # Load cleaned data
        self.logger.info(f"\n[1/4] Loading cleaned data from {self.input_file}...")
        df = pd.read_csv(self.input_file)
        self.logger.info(f"      Loaded {len(df):,} rows × {len(df.columns)} columns")
        
        # Normalize
        self.logger.info(f"\n[2/4] Normalizing coordinates to [0, 1]...")
        self.normalizer = BadmintonNormalizer()
        df_norm, bounds = self.normalizer.process_dataframe(df)
        
        self.logger.info(f"      Processed {self.normalizer.stats['rows_processed']:,} rows")
        self.logger.info(f"      Normalized coords: {self.normalizer.stats['coords_normalized']:,}")
        self.logger.info(f"      Preserved NaNs: {self.normalizer.stats['nans_preserved']:,}")
        
        # Validate
        self.logger.info(f"\n[3/4] Validating normalized data...")
        validation = self.normalizer.validate_normalized(df_norm)
        
        if validation['valid']:
            self.logger.info(f"      ✓ All coordinates in valid [0, 1] range")
        else:
            self.logger.critical(f"      ✗ VALIDATION FAILED:")
            for violation in validation['violations']:
                self.logger.warning(f"        - {violation}")
        
        self.logger.info(f"\n      Coordinate statistics:")
        for coord, stats_coord in sorted(validation['coord_stats'].items()):
            if stats_coord['count'] > 0:
                self.logger.info(f"        {coord:15} | [{stats_coord['min']:.4f}, {stats_coord['max']:.4f}] "
                    f"| N={stats_coord['count']:8,} | NaN={stats_coord['nans']:,}")
            else:
                self.logger.warning(f"        {coord:15} | All NaN | N=0 | NaN={stats_coord['nans']:,}")
        
        # Save
        self.logger.info(f"\n[4/4] Saving normalized data to {self.output_file}...")
        df_norm.to_csv(self.output_file, index=False)
        self.logger.info(f"      ✓ Saved {len(df_norm):,} rows")
        
        # Save metadata
        stats_combined = {
            'bounds': bounds,
            'rows_processed': int(self.normalizer.stats['rows_processed']),
            'coords_normalized': int(self.normalizer.stats['coords_normalized']),
            'nans_preserved': int(self.normalizer.stats['nans_preserved']),
            'validation': validation
        }
        with open(self.stats_file, 'w') as f:
            json.dump(stats_combined, f, indent=2)
        self.logger.info(f"      ✓ Stats saved to {self.stats_file}")
        print("✓ NORMALIZATION COMPLETE")


