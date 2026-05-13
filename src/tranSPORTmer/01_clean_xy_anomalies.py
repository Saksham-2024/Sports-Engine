"""
STEP 1: Clean XY Anomalies in Shuttle Coordinates

Strategy:
  - Bands ≤5 frames: Linear interpolation
  - Bands 5-10 frames: Mark as NaN + mask
  - Bands >10 frames: Split segment at boundary, pad with NaN
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
import yaml

with open('../../configs/configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

class XYAnomalyHandler:
    """Handle scattered XY anomalies in shuttle coordinates."""
    
    def __init__(self, 
                 x_bounds=(-1, 6.5),
                 y_bounds=(-1, 14.5),
                 interp_threshold=5,
                 mask_threshold=10):
        """
        Args:
            x_bounds: Reasonable bounds for X
            y_bounds: Reasonable bounds for Y
            interp_threshold: Band size ≤ this → interpolate
            mask_threshold: Band size > this → split segment
        """
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.interp_threshold = interp_threshold
        self.mask_threshold = mask_threshold
        
        self.stats = {
            'segments_processed': 0,
            'anomalies_interpolated': 0,
            'anomalies_masked': 0,
            'segments_split': 0,
            'frames_interpolated': 0,
            'frames_masked': 0,
            'segments_created': 0
        }
    
    def is_anomalous(self, x, y):
        """Check if XY coordinate is anomalous."""
        if pd.isna(x) or pd.isna(y):
            return True  # NaN is treated as anomaly
        return (x < self.x_min or x > self.x_max) or (y < self.y_min or y > self.y_max)
    
    def find_anomaly_bands(self, x_series, y_series):
        """
        Identify consecutive anomalous frames.
        
        Returns:
            List of (start_idx, end_idx, size) tuples
        """
        anomaly_mask = [self.is_anomalous(x, y) for x, y in zip(x_series, y_series)]
        
        bands = []
        in_band = False
        start_idx = None
        
        for i, is_anom in enumerate(anomaly_mask):
            if is_anom and not in_band:
                # Start of band
                start_idx = i
                in_band = True
            elif not is_anom and in_band:
                # End of band
                bands.append((start_idx, i, i - start_idx))
                in_band = False
        
        # Handle band that extends to end
        if in_band:
            bands.append((start_idx, len(anomaly_mask), len(anomaly_mask) - start_idx))
        
        return bands
    
    def interpolate_band(self, x_series, y_series, start, end):
        """Linearly interpolate over anomaly band."""
        x_series = x_series.copy()
        y_series = y_series.copy()
        
        # Ensure we have valid values to interpolate from
        if start == 0:
            # Band at start, use next valid value
            if end < len(x_series):
                x_series.iloc[start:end] = x_series.iloc[end]
                y_series.iloc[start:end] = y_series.iloc[end]
        elif end >= len(x_series):
            # Band at end, use last valid value
            x_series.iloc[start:end] = x_series.iloc[start - 1]
            y_series.iloc[start:end] = y_series.iloc[start - 1]
        else:
            # Band in middle, interpolate between valid values
            x_before = x_series.iloc[start - 1]
            x_after = x_series.iloc[end]
            y_before = y_series.iloc[start - 1]
            y_after = y_series.iloc[end]
            
            steps = np.arange(end - start + 1) / (end - start)
            x_interp = x_before + steps[:-1] * (x_after - x_before)
            y_interp = y_before + steps[:-1] * (y_after - y_before)
            
            x_series.iloc[start:end] = x_interp
            y_series.iloc[start:end] = y_interp
        
        return x_series, y_series
    
    def process_segment(self, seg_df, seg_id, match_id):
        """
        Process single segment for XY anomalies.
        
        Returns:
            List of processed segment dataframes (may be 1 or multiple if split)
        """
        seg_df = seg_df.copy().reset_index(drop=True)
        
        # Find anomaly bands
        bands = self.find_anomaly_bands(seg_df['shuttle_x'], seg_df['shuttle_y'])
        
        if not bands:
            # No anomalies
            return [seg_df]
        
        # Process bands from end to start (to preserve indices)
        for start, end, size in reversed(bands):
            if size <= self.interp_threshold:
                # Interpolate
                seg_df['shuttle_x'], seg_df['shuttle_y'] = self.interpolate_band(
                    seg_df['shuttle_x'], seg_df['shuttle_y'], start, end
                )
                self.stats['anomalies_interpolated'] += 1
                self.stats['frames_interpolated'] += size
            
            elif size <= self.mask_threshold:
                # Mask as NaN
                seg_df.loc[start:end-1, 'shuttle_x'] = np.nan
                seg_df.loc[start:end-1, 'shuttle_y'] = np.nan
                seg_df.loc[start:end-1, 'shuttle_z'] = np.nan
                self.stats['anomalies_masked'] += 1
                self.stats['frames_masked'] += size
            
            else:
                # Split segment (large gap)
                # Will handle in main loop after all band processing
                pass
        
        # Check if any large bands remain (these need segment splitting)
        final_bands = self.find_anomaly_bands(seg_df['shuttle_x'], seg_df['shuttle_y'])
        large_bands = [b for b in final_bands if b[2] > self.mask_threshold]
        
        if large_bands:
            # Split at each large band boundary
            segments_split = []
            last_end = 0
            
            for start, end, size in large_bands:
                # Keep segment up to anomaly start
                if start > last_end:
                    seg_part = seg_df.iloc[last_end:start].copy()
                    segments_split.append(seg_part)
                
                # Skip the anomaly band (start to end)
                last_end = end
                self.stats['segments_split'] += 1
            
            # Keep remaining segment after last anomaly
            if last_end < len(seg_df):
                seg_part = seg_df.iloc[last_end:].copy()
                segments_split.append(seg_part)
            
            # Renumber segment IDs for split segments
            for i, seg in enumerate(segments_split):
                seg['segment_idx'] = f"{seg_id}_split{i}"
            
            self.stats['segments_created'] += len(segments_split)
            return segments_split
        
        return [seg_df]
    
    def process_dataframe(self, df):
        """
        Process entire dataframe.
        
        Returns:
            Cleaned dataframe with NaN-masked anomalies
        """
        results = []
        
        for match_id in df['match_id'].unique():
            match_df = df[df['match_id'] == match_id]
            
            for seg_id in match_df['segment_idx'].unique():
                seg_df = match_df[match_df['segment_idx'] == seg_id]
                
                # Process segment
                seg_results = self.process_segment(seg_df, seg_id, match_id)
                results.extend(seg_results)
                
                self.stats['segments_processed'] += 1
        
        # Concatenate all results
        df_cleaned = pd.concat(results, ignore_index=True)
        
        return df_cleaned, self.stats


def main():
    """Main preprocessing pipeline."""
    
    input_file = os.path.join(configs['global']['project_root'], configs['dataset_creation']['transformer_dataset_csv'])
    output_file = os.path.join(configs['global']['project_root'], configs['tranSPORTmer']['data']['cleaned_csv'])
    stats_file = os.path.join(configs['global']['project_root'], configs['tranSPORTmer']['data']['cleaning_stats'])
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("=" * 80)
    print("STEP 1: XY ANOMALY CLEANING")
    print("=" * 80)
    
    # Load
    print(f"\n[1/3] Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"      Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Process
    print(f"\n[2/3] Cleaning XY anomalies...")
    print(f"      Strategy: interpolate ≤5 | mask 5-10 | split >10 frames")
    handler = XYAnomalyHandler(
        x_bounds=(-1, 6.5),
        y_bounds=(-1, 14.5),
        interp_threshold=5,
        mask_threshold=10
    )
    df_cleaned, stats = handler.process_dataframe(df)
    
    print(f"\n      Results:")
    print(f"        Segments processed: {stats['segments_processed']}")
    print(f"        Anomalies interpolated: {stats['anomalies_interpolated']} "
          f"({stats['frames_interpolated']} frames)")
    print(f"        Anomalies masked: {stats['anomalies_masked']} "
          f"({stats['frames_masked']} frames)")
    print(f"        Segments split: {stats['segments_split']} "
          f"({stats['segments_created']} new segments created)")
    
    # Validate
    print(f"\n[3/3] Saving cleaned data to {output_file}...")
    df_cleaned.to_csv(output_file, index=False)
    print(f"      ✓ Saved {len(df_cleaned):,} rows")
    
    # Save stats
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"      ✓ Stats saved to {stats_file}")
    
    print("\n" + "=" * 80)
    print("✓ XY ANOMALY CLEANING COMPLETE")
    print("=" * 80)
    print(f"\nNext step: normalize_with_masks.py")


if __name__ == '__main__':
    main()
