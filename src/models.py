"""
Water Analytics & Anomaly Detection Model Suite
Implements physics-informed statistical & machine learning models for:
1. Leak & Burst Detection (MNF creep, Pressure-Flow bursts, DMA transmission mass balance)
2. Pipe Clog / Valve constriction detection
3. Water Treatment & Quality Sentinel (Filter fouling baseline drift, Turbidity risk, Coagulant dosing)
4. Borehole & Pump Health (Specific energy drift, Screen clogging, Pump outage failover)
5. Sensor Integrity Guardian (Stuck flatline sensor, Comms dropouts)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

try:
    from ml_models import WaterMLPipeline, TimeSeriesDemandForecaster, PredictiveBreakdownClassifier
except ImportError:
    from .ml_models import WaterMLPipeline, TimeSeriesDemandForecaster, PredictiveBreakdownClassifier

class WaterAnalyticsEngine:
    def __init__(self):
        self.dosing_stock_conc_g_per_l = 100.0 # 10% alum solution = 100 g/L

    def run_all_models(self, df: pd.DataFrame, include_ml: bool = False) -> pd.DataFrame:
        """Runs the entire suite of diagnostic and predictive models on the preprocessed telemetry."""
        data = df.copy()

        # 1. Sensor Guardian: Stuck Sensor & Dropout
        data = self._detect_sensor_faults(data)

        # 2. Minimum Night Flow (MNF) & Creeping Leaks
        data = self._detect_creeping_leaks_mnf(data)

        # 3. Fast Bursts & Clogs (Pressure-Flow Residuals)
        data = self._detect_bursts_and_clogs(data)

        # 4. Transmission Main Mass Balance (Unmetered Leaks)
        data = self._detect_main_transmission_leaks(data)

        # 5. Filtration Sentinel: dP baseline creep & Turbidity breakthrough
        data = self._detect_filter_fouling(data)

        # 6. Automated Coagulant Dosing Advisory
        data = self._compute_coagulant_dosing_advisory(data)

        # 7. Borehole Health & Pump Outages
        data = self._detect_borehole_anomalies(data)

        # 8. Unified Master Alert Synthesizer
        data = self._synthesize_alerts(data)

        # 9. Optional ML Time-Series & Breakdown Pipeline
        if include_ml:
            pipeline = WaterMLPipeline()
            pipeline.fit(data)
            data = pipeline.predict(data)

        return data

    def _detect_sensor_faults(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detects frozen/stuck sensors (zero variance) and comms dropouts."""
        # Rolling standard deviation over 12 hours for treated turbidity
        roll_std_turb = df['treated_turbidity_ntu'].rolling(12, min_periods=6).std()
        # A flatline sensor has almost zero variance
        df['alert_stuck_turbidity_sensor'] = ((roll_std_turb < 0.005) & (df['treated_turbidity_ntu'] > 0.1)).astype(int)
        
        # Comms dropout flag is inherited from preprocessing
        df['alert_comms_dropout'] = df['is_logger_dropout']
        return df

    def _detect_creeping_leaks_mnf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Minimum Night Flow (MNF) analysis (02:00-04:00).
        Computes 14-day rolling baseline of night flow for each building.
        Elevated night flow signals creeping leaks.
        """
        for bldg in ['A', 'B', 'C']:
            flow_col = f'bldg{bldg}_flow_m3h'
            
            # Extract night flow only
            night_mask = df['is_night_mnf'] == 1
            night_series = df.loc[night_mask, flow_col]
            
            # Compute 14-day rolling minimum and 30-day baseline
            rolling_mnf_min = night_series.rolling(14 * 3, min_periods=7, center=False).quantile(0.15)
            # Align back with dataframe index
            df[f'bldg{bldg}_mnf_baseline'] = rolling_mnf_min
            df[f'bldg{bldg}_mnf_baseline'] = df[f'bldg{bldg}_mnf_baseline'].ffill().bfill()

            # Leak threshold: night flow consistently exceeding baseline by margin
            margin = 0.30 if bldg in ['A', 'C'] else 0.50
            excess_night_flow = np.maximum(0, df[flow_col] - df[f'bldg{bldg}_mnf_baseline'])
            
            # Rolling 24h average excess night flow
            excess_smoothed = excess_night_flow.rolling(72, min_periods=24).mean()
            df[f'alert_leak_slow_{bldg}'] = (
                (excess_smoothed > margin) & 
                (df[f'bldg{bldg}_pressure_bar'] < df[f'bldg{bldg}_pressure_bar'].rolling(168).mean() + 0.1)
            ).astype(int)

        return df

    def _detect_bursts_and_clogs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pressure-Flow joint residual anomaly detector.
        - Pipe Burst: Flow z-score >> +3 AND Pressure drop z-score << -2.5
        - Pipe Clog: Flow z-score << -2.5 AND Pressure z-score >> +1.5
        """
        for bldg in ['A', 'B', 'C']:
            q_col = f'bldg{bldg}_flow_m3h'
            p_col = f'bldg{bldg}_pressure_bar'

            # 7-day rolling mean & std for flow and pressure
            q_mean = df[q_col].rolling(168, min_periods=24).mean()
            q_std = df[q_col].rolling(168, min_periods=24).std().clip(lower=0.05)
            p_mean = df[p_col].rolling(168, min_periods=24).mean()
            p_std = df[p_col].rolling(168, min_periods=24).std().clip(lower=0.05)

            q_z = (df[q_col] - q_mean) / q_std
            p_z = (df[p_col] - p_mean) / p_std

            # Fast burst detection
            df[f'alert_burst_{bldg}'] = (
                (q_z > 3.0) & (p_z < -2.0) & (df[q_col] > 2.0)
            ).astype(int)

            # Clog / Blockage detection: Flow drops significantly while pressure rises
            # Gradual or sudden restriction
            df[f'alert_clog_{bldg}'] = (
                (q_z < -2.0) & (p_z > 1.2)
            ).astype(int)

        return df

    def _detect_main_transmission_leaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        District Metered Area (DMA) Mass Balance.
        Residual = treatment_outlet - sum(bldgA + bldgB + bldgC).
        Normal background loss is ~0.05 m3/h. Unmetered main bursts spike by +1.0 m3/h.
        """
        # Smooth the mass balance over 24 hours to filter meter timestamp jitter
        loss_smoothed = df['main_loss_m3h'].rolling(48, min_periods=12).mean()
        # Normal baseline ~0.05 - 0.2 m3/h. Alert if sustained loss > 0.8 m3/h
        df['alert_main_leak'] = (loss_smoothed > 0.75).astype(int)
        return df

    def _detect_filter_fouling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter condition assessment:
        1. Media fouling: post-backwash dP baseline creeps from 18 kPa up to 48 kPa.
        2. Turbidity breakthrough: treated turbidity exceeds WHO 0.5 NTU standard.
        """
        # Floor creep: normal clean floor is ~18-20 kPa. Over 28 kPa is warning, over 35 kPa is severe clog.
        df['filter_fouling_level'] = np.clip((df['filter_dp_floor_kpa'] - 18.0) / (48.0 - 18.0), 0.0, 1.0)
        df['alert_filter_clog'] = (df['filter_dp_floor_kpa'] > 28.0).astype(int)
        
        # Treated turbidity breakthrough (> 0.5 NTU WHO threshold)
        df['alert_turbidity_exceedance'] = (df['treated_turbidity_ntu'] > 0.50).astype(int)
        return df

    def _compute_coagulant_dosing_advisory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates recommended coagulant dose (Alum / Polyaluminium Chloride) in mg/L and dosing pump rate (mL/min).
        Standard water engineering power-law relationship:
        Dose (mg/L) = 4.5 * (raw_turbidity ** 0.6) + 5.0
        """
        # Recommended dose in mg/L (ppm)
        raw_turb = df['raw_turbidity_ntu'].clip(lower=0.5)
        df['rec_coagulant_dose_mg_l'] = 4.5 * np.power(raw_turb, 0.6) + 5.0

        # Flow into plant in m3/h -> L/h = m3/h * 1000
        # Required mass of active coagulant (mg/h) = Dose (mg/L) * Flow (L/h)
        # In stock solution (100 g/L = 100,000 mg/L = 100 mg/mL):
        # Dosing pump rate (mL/min) = [Dose (mg/L) * Flow (m3/h) * 1000] / [100,000 * 60]
        # Simplifies to: Dose (mg/L) * Flow (m3/h) / 6.0
        inlet_flow = df['treatment_inlet_m3h'].clip(lower=0.0)
        df['rec_dosing_flow_ml_min'] = (df['rec_coagulant_dose_mg_l'] * inlet_flow) / 6.0
        return df

    def _detect_borehole_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects:
        1. Borehole Pump Outage: runtime commanded or expected during scheduled rotation but flow = 0.
        2. Screen Clogging & Aquifer Yield Collapse:
           Specific energy (kWh/m3) and drawdown per m3/h rising persistently.
        """
        # 1. Pump Outage Detection
        # When one pump runs > 50 min/hr while the other has flow=0 during expected duty
        df['alert_bh1_outage'] = ((df['bh1_flow_m3h'] < 0.05) & (df['bh2_runtime_min'] > 45.0)).astype(int)
        df['alert_bh2_outage'] = ((df['bh2_flow_m3h'] < 0.05) & (df['bh1_runtime_min'] > 45.0)).astype(int)

        # 2. Screen Clogging in Borehole 2:
        # Runtime per m3 up, specific energy up, drawdown up
        # 14-day rolling median of specific energy for BH2 when operating
        bh2_active = df[df['bh2_flow_m3h'] > 0.5]['bh2_kwh_per_m3']
        rolling_bh2_kwh = bh2_active.rolling(14 * 24, min_periods=48).median()
        df['bh2_kwh_m3_smooth'] = rolling_bh2_kwh
        df['bh2_kwh_m3_smooth'] = df['bh2_kwh_m3_smooth'].ffill().bfill()

        # Normal specific energy for BH2 is ~0.35-0.42 kWh/m3. Screen clogging pushes it > 0.55 kWh/m3.
        df['alert_bh2_screen_clog'] = (
            (df['bh2_kwh_m3_smooth'] > 0.52) & 
            (df['bh2_water_level_m'] > 42.0)
        ).astype(int)

        return df

    def _synthesize_alerts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combines all alerts into severity classes (CRITICAL, WARNING, NORMAL)."""
        critical_cols = [
            'alert_burst_A', 'alert_burst_B', 'alert_burst_C',
            'alert_main_leak', 'alert_bh1_outage', 'alert_bh2_outage',
            'alert_turbidity_exceedance'
        ]
        warning_cols = [
            'alert_leak_slow_A', 'alert_leak_slow_B', 'alert_leak_slow_C',
            'alert_clog_A', 'alert_clog_B', 'alert_clog_C',
            'alert_filter_clog', 'alert_bh2_screen_clog',
            'alert_stuck_turbidity_sensor', 'alert_comms_dropout'
        ]

        df['has_critical_alert'] = (df[critical_cols].sum(axis=1) > 0).astype(int)
        df['has_warning_alert'] = (df[warning_cols].sum(axis=1) > 0).astype(int)
        
        df['system_status'] = 'NORMAL'
        df.loc[df['has_warning_alert'] == 1, 'system_status'] = 'WARNING'
        df.loc[df['has_critical_alert'] == 1, 'system_status'] = 'CRITICAL'

        return df

if __name__ == '__main__':
    from preprocessing import load_and_clean_data
    df = load_and_clean_data()
    engine = WaterAnalyticsEngine()
    processed_df = engine.run_all_models(df)
    print("Models executed successfully!")
    print(f"System status distribution:\n{processed_df['system_status'].value_counts()}")
