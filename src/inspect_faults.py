import pandas as pd
import numpy as np

df = pd.read_csv('water_hourly_2023_2025.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
gt = pd.read_csv('fault_log_ground_truth.csv')

print("--- Inspecting Ground Truth Signatures ---")
for idx, row in gt.iterrows():
    if row['start'] == 'random':
        continue
    t_start = pd.to_datetime(row['start'])
    t_end = pd.to_datetime(row['end'])
    
    # Slice window and preceding baseline window
    baseline_start = t_start - pd.Timedelta(days=7)
    df_base = df[(df['timestamp'] >= baseline_start) & (df['timestamp'] < t_start)]
    df_fault = df[(df['timestamp'] >= t_start) & (df['timestamp'] <= t_end)]
    
    print(f"\n[{idx}] {row['asset']} | {row['fault_type']}")
    print(f"    Window: {row['start']} -> {row['end']} ({len(df_fault)} hrs)")
    print(f"    Expected: {row['expected_signature']}")
    
    if "Building A" in row['asset']:
        print(f"    Bldg A flow base night (02-04h): {df_base[df_base['timestamp'].dt.hour.isin([2,3,4])]['bldgA_flow_m3h'].mean():.3f} -> fault night: {df_fault[df_fault['timestamp'].dt.hour.isin([2,3,4])]['bldgA_flow_m3h'].mean():.3f}")
    elif "Building B" in row['asset']:
        print(f"    Bldg B flow base: {df_base['bldgB_flow_m3h'].mean():.3f}, pressure: {df_base['bldgB_pressure_bar'].mean():.3f}")
        print(f"    Bldg B flow fault: {df_fault['bldgB_flow_m3h'].mean():.3f}, pressure: {df_fault['bldgB_pressure_bar'].mean():.3f}")
    elif "Building C" in row['asset']:
        print(f"    Bldg C flow base: {df_base['bldgC_flow_m3h'].mean():.3f}, pressure: {df_base['bldgC_pressure_bar'].mean():.3f}")
        print(f"    Bldg C flow fault: {df_fault['bldgC_flow_m3h'].mean():.3f}, pressure: {df_fault['bldgC_pressure_bar'].mean():.3f}")
    elif "Filter" in row['asset']:
        print(f"    Filter dp base: {df_base['filter_dp_kpa'].mean():.1f} -> fault: {df_fault['filter_dp_kpa'].mean():.1f}, max: {df_fault['filter_dp_kpa'].max():.1f}")
        print(f"    Treated turb base: {df_base['treated_turbidity_ntu'].mean():.3f} -> fault: {df_fault['treated_turbidity_ntu'].mean():.3f}")
    elif "Distribution" in row['asset']:
        base_loss = (df_base['treatment_outlet_m3h'] - (df_base['bldgA_flow_m3h'] + df_base['bldgB_flow_m3h'] + df_base['bldgC_flow_m3h'])).mean()
        fault_loss = (df_fault['treatment_outlet_m3h'] - (df_fault['bldgA_flow_m3h'] + df_fault['bldgB_flow_m3h'] + df_fault['bldgC_flow_m3h'])).mean()
        print(f"    Main loss base: {base_loss:.3f} -> fault: {fault_loss:.3f}")
    elif "Borehole 1" in row['asset']:
        print(f"    BH1 flow base: {df_base['bh1_flow_m3h'].mean():.3f} -> fault: {df_fault['bh1_flow_m3h'].mean():.3f}")
        print(f"    BH2 flow base: {df_base['bh2_flow_m3h'].mean():.3f} -> fault: {df_fault['bh2_flow_m3h'].mean():.3f}")
    elif "Borehole 2" in row['asset']:
        print(f"    BH2 flow base: {df_base['bh2_flow_m3h'].mean():.3f} -> fault: {df_fault['bh2_flow_m3h'].mean():.3f}")
        print(f"    BH2 water level base: {df_base['bh2_water_level_m'].mean():.2f} -> fault: {df_fault['bh2_water_level_m'].mean():.2f}")
        bh2_base_eff = (df_base['bh2_pump_kw'] / df_base['bh2_flow_m3h'].clip(lower=0.1)).mean()
        bh2_fault_eff = (df_fault['bh2_pump_kw'] / df_fault['bh2_flow_m3h'].clip(lower=0.1)).mean()
        print(f"    BH2 specific energy base: {bh2_base_eff:.3f} -> fault: {bh2_fault_eff:.3f}")
    elif "turbidity sensor" in row['asset']:
        print(f"    Treated turb std base: {df_base['treated_turbidity_ntu'].std():.4f} -> fault std: {df_fault['treated_turbidity_ntu'].std():.4f}")
