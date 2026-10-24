"""
Water Telemetry Preprocessing and Feature Engineering Pipeline
Handles missing sensor data, computes hydraulic mass balances, minimum night flows,
and energy metrics for the Strathmore / County WSP Digital Twin.
"""

import pandas as pd
import numpy as np

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "water_hourly_2023_2025.csv"
DEFAULT_GROUND_TRUTH_PATH = BASE_DIR / "fault_log_ground_truth.csv"

def load_and_clean_data(csv_path=None):
    if csv_path is None:
        csv_path = DEFAULT_DATA_PATH
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 1. Flag dropouts (where multiple sensors are NaN at once)
    numeric_cols = [c for c in df.columns if c not in ['timestamp', 'backwash_flag']]
    df['nan_count'] = df[numeric_cols].isnull().sum(axis=1)
    df['is_logger_dropout'] = (df['nan_count'] >= 5).astype(int)

    # 2. Imputation:
    # Linear interpolation for sequential hourly rows, followed by forward and backward fill
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    df['backwash_flag'] = df['backwash_flag'].fillna(0).astype(int)

    # 3. Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['is_night_mnf'] = df['hour'].isin([2, 3, 4]).astype(int)

    # 4. Hydraulic mass balance features
    df['sum_bldg_flow_m3h'] = df['bldgA_flow_m3h'] + df['bldgB_flow_m3h'] + df['bldgC_flow_m3h']
    df['main_loss_m3h'] = df['treatment_outlet_m3h'] - df['sum_bldg_flow_m3h']
    df['main_loss_pct'] = np.where(
        df['treatment_outlet_m3h'] > 0.1,
        (df['main_loss_m3h'] / df['treatment_outlet_m3h']) * 100.0,
        0.0
    )

    # Treatment plant mass balance & efficiency
    df['plant_consumption_m3h'] = df['treatment_inlet_m3h'] - df['treatment_outlet_m3h']
    df['filter_recovery_pct'] = np.where(
        df['treatment_inlet_m3h'] > 0.1,
        (df['treatment_outlet_m3h'] / df['treatment_inlet_m3h']) * 100.0,
        96.0
    )

    # 5. Borehole diagnostics & specific energy
    # Specific energy (kWh / m3)
    df['bh1_total_m3'] = df['bh1_flow_m3h']
    df['bh2_total_m3'] = df['bh2_flow_m3h']
    df['total_raw_flow_m3h'] = df['bh1_flow_m3h'] + df['bh2_flow_m3h']

    df['bh1_kwh_per_m3'] = np.where(df['bh1_flow_m3h'] > 0.05, df['bh1_pump_kw'] / df['bh1_flow_m3h'], np.nan)
    df['bh2_kwh_per_m3'] = np.where(df['bh2_flow_m3h'] > 0.05, df['bh2_pump_kw'] / df['bh2_flow_m3h'], np.nan)
    
    df['bh1_min_per_m3'] = np.where(df['bh1_flow_m3h'] > 0.05, df['bh1_runtime_min'] / df['bh1_flow_m3h'], np.nan)
    df['bh2_min_per_m3'] = np.where(df['bh2_flow_m3h'] > 0.05, df['bh2_runtime_min'] / df['bh2_flow_m3h'], np.nan)

    # 6. Filter post-backwash dP baseline tracking
    # Detect the floor dP after a backwash occurs
    df['post_backwash_hours'] = 0
    hours_since_bw = 999
    post_bw_hours_col = []
    for flag in df['backwash_flag']:
        if flag == 1:
            hours_since_bw = 0
        else:
            hours_since_bw += 1
        post_bw_hours_col.append(hours_since_bw)
    df['hours_since_backwash'] = post_bw_hours_col

    # Post-backwash dP baseline (dP measured within 1-2 hours after backwash)
    # Forward-fill this baseline to monitor floor drift over time
    df['recent_post_bw_dp'] = np.where(
        df['hours_since_backwash'].isin([1, 2]),
        df['filter_dp_kpa'],
        np.nan
    )
    df['filter_dp_floor_kpa'] = df['recent_post_bw_dp'].ffill().bfill()

    return df

if __name__ == '__main__':
    df = load_and_clean_data()
    print(f"Preprocessed shape: {df.shape}")
    print(f"Sample mass balance loss mean: {df['main_loss_m3h'].mean():.3f} m3/h")
