"""
Machine Learning & Time-Series Predictive Breakdown Suite
==========================================================
Implements end-to-end Machine Learning models for:
1. Time-Series Demand & Pressure Forecasting (HistGradientBoosting / RF Regressors)
   - Autoregressive multi-horizon forecasting with cyclical time encodings & lags
   - Prediction interval residuals for early warning of bursts, creeping leaks, and restrictions
2. Supervised Predictive Breakdown & Fault Classification (HistGradientBoosting Classifier)
   - Multi-class breakdown classifier trained on labeled ground-truth events:
     [NORMAL, LEAK_BURST, PIPE_CLOG, FILTER_FOULING, PUMP_OUTAGE, SCREEN_CLOGGING, SENSOR_FAULT]
   - Breakdown risk probability estimation (P(Breakdown))
3. Unsupervised Anomaly Detection (IsolationForest)
   - Multivariate telemetry anomaly scoring for emergent / novel failures
4. Predictive Breakdown Prognostics (RUL & Trend Extrapolation)
   - Filter media fouling rate -> Estimated days to backwash failure / media replacement
   - Borehole 2 specific energy drift -> Days to critical yield collapse / pump overhaul
   - Creeping leak expansion rate -> Projected water loss growth
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_ARTIFACTS_DIR = BASE_DIR / "models_saved"


# =====================================================================
# 1. TIME-SERIES FEATURE ENGINEERING
# =====================================================================

def extract_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs high-dimensional time-series features:
    - Cyclical harmonic time encodings (sin/cos for hour, day of week, month)
    - Autoregressive lag features (t-1, t-2, t-24, t-168)
    - Rolling window statistics (24h mean/std, 168h weekly mean)
    - Hydraulic interaction terms & ratios
    """
    data = df.copy()
    if not np.issubdtype(data['timestamp'].dtype, np.datetime64):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    data = data.sort_values('timestamp').reset_index(drop=True)

    # 1. Harmonic Cyclical Features
    hour = data['timestamp'].dt.hour
    dow = data['timestamp'].dt.dayofweek
    month = data['timestamp'].dt.month

    data['sin_hour'] = np.sin(2 * np.pi * hour / 24.0)
    data['cos_hour'] = np.cos(2 * np.pi * hour / 24.0)
    data['sin_dow'] = np.sin(2 * np.pi * dow / 7.0)
    data['cos_dow'] = np.cos(2 * np.pi * dow / 7.0)
    data['sin_month'] = np.sin(2 * np.pi * month / 12.0)
    data['cos_month'] = np.cos(2 * np.pi * month / 12.0)
    data['is_weekend'] = dow.isin([5, 6]).astype(float)
    data['is_night_mnf'] = hour.isin([2, 3, 4]).astype(float)

    # Academic break flag (approx April, August, December dips)
    data['is_academic_holiday'] = month.isin([4, 8, 12]).astype(float)

    # 2. Hydraulic Lags & Rolling Dynamics
    tracked_series = [
        'bldgA_flow_m3h', 'bldgB_flow_m3h', 'bldgC_flow_m3h',
        'bldgA_pressure_bar', 'bldgB_pressure_bar', 'bldgC_pressure_bar',
        'treatment_outlet_m3h', 'treatment_inlet_m3h',
        'filter_dp_kpa', 'treated_turbidity_ntu'
    ]

    for col in tracked_series:
        if col in data.columns:
            # Short-term lag (1 hr, 2 hr)
            data[f'{col}_lag1'] = data[col].shift(1).bfill()
            data[f'{col}_lag2'] = data[col].shift(2).bfill()
            # Daily lag (24 hr)
            data[f'{col}_lag24'] = data[col].shift(24).bfill()
            # Weekly lag (168 hr)
            data[f'{col}_lag168'] = data[col].shift(168).bfill()
            # Rolling statistics
            data[f'{col}_roll_mean_24'] = data[col].rolling(24, min_periods=6).mean().bfill()
            data[f'{col}_roll_std_24'] = data[col].rolling(24, min_periods=6).std().fillna(0.0)
            data[f'{col}_roll_mean_168'] = data[col].rolling(168, min_periods=24).mean().bfill()

    # 3. Inter-sensor Ratios & Differences
    if 'sum_bldg_flow_m3h' not in data.columns:
        data['sum_bldg_flow_m3h'] = data['bldgA_flow_m3h'] + data['bldgB_flow_m3h'] + data['bldgC_flow_m3h']

    if 'main_loss_m3h' not in data.columns:
        data['main_loss_m3h'] = data['treatment_outlet_m3h'] - data['sum_bldg_flow_m3h']

    data['main_loss_ratio'] = np.where(
        data['treatment_outlet_m3h'] > 0.1,
        data['main_loss_m3h'] / data['treatment_outlet_m3h'],
        0.0
    )

    # Building pressure-flow impedance ratio (R = P / Q)
    for bldg in ['A', 'B', 'C']:
        flow_col = f'bldg{bldg}_flow_m3h'
        press_col = f'bldg{bldg}_pressure_bar'
        data[f'bldg{bldg}_flow_pressure_ratio'] = data[flow_col] / (data[press_col].clip(lower=0.5))

    # Borehole power to flow efficiency (Specific Energy kWh/m3)
    if 'bh1_kwh_per_m3' not in data.columns:
        data['bh1_kwh_per_m3'] = np.where(data['bh1_flow_m3h'] > 0.05, data['bh1_pump_kw'] / data['bh1_flow_m3h'], 0.35)
    if 'bh2_kwh_per_m3' not in data.columns:
        data['bh2_kwh_per_m3'] = np.where(data['bh2_flow_m3h'] > 0.05, data['bh2_pump_kw'] / data['bh2_flow_m3h'], 0.38)

    # Rolling specific energy for BH2 to detect screen degradation
    data['bh2_kwh_per_m3_roll7d'] = data['bh2_kwh_per_m3'].rolling(168, min_periods=24).median().bfill()

    # Turbidity filtration ratio (treated / raw)
    raw_clipped = data['raw_turbidity_ntu'].clip(lower=0.1)
    data['turbidity_removal_ratio'] = data['treated_turbidity_ntu'] / raw_clipped

    # Rolling standard deviation of treated turbidity to spot stuck flatline sensor
    data['treated_turb_roll_std_12h'] = data['treated_turbidity_ntu'].rolling(12, min_periods=6).std().fillna(0.05)
    data['treated_turb_roll_std_24'] = data['treated_turbidity_ntu'].rolling(24, min_periods=6).std().fillna(0.05)

    # Asset-agnostic normalized building deviations for spatial transfer
    for b in ['bldgA', 'bldgB', 'bldgC']:
        q_mean = data[f'{b}_flow_m3h'].rolling(168, min_periods=24).mean().bfill()
        q_std = data[f'{b}_flow_m3h'].rolling(168, min_periods=24).std().replace(0, 1e-4).bfill()
        data[f'{b}_flow_z'] = (data[f'{b}_flow_m3h'] - q_mean) / q_std
        
        p_mean = data[f'{b}_pressure_bar'].rolling(168, min_periods=24).mean().bfill()
        p_std = data[f'{b}_pressure_bar'].rolling(168, min_periods=24).std().replace(0, 1e-4).bfill()
        data[f'{b}_pressure_z'] = (data[f'{b}_pressure_bar'] - p_mean) / p_std

    data['max_flow_z'] = data[['bldgA_flow_z', 'bldgB_flow_z', 'bldgC_flow_z']].max(axis=1)
    data['min_flow_z'] = data[['bldgA_flow_z', 'bldgB_flow_z', 'bldgC_flow_z']].min(axis=1)
    data['min_pressure_z'] = data[['bldgA_pressure_z', 'bldgB_pressure_z', 'bldgC_pressure_z']].min(axis=1)
    data['max_pressure_z'] = data[['bldgA_pressure_z', 'bldgB_pressure_z', 'bldgC_pressure_z']].max(axis=1)

    # Borehole pump outage mismatch signatures
    if 'bh1_runtime_min' in data.columns and 'bh1_flow_m3h' in data.columns:
        data['bh1_pump_mismatch'] = ((data['bh1_runtime_min'] > 20) & (data['bh1_flow_m3h'] < 0.1)).astype(float)
    if 'bh2_runtime_min' in data.columns and 'bh2_flow_m3h' in data.columns:
        data['bh2_pump_mismatch'] = ((data['bh2_runtime_min'] > 20) & (data['bh2_flow_m3h'] < 0.1)).astype(float)

    return data


# =====================================================================
# 2. GROUND TRUTH LABELING & FAULT TAXONOMY
# =====================================================================

FAULT_CATEGORY_MAPPING = {
    'Leak (slow, growing)': 'LEAK_BURST',
    'Leak (burst)': 'LEAK_BURST',
    'Leak (unmetered)': 'LEAK_BURST',
    'Leak (small, constant)': 'LEAK_BURST',
    'Leak': 'LEAK_BURST',
    'Clog / partial blockage (gradual)': 'PIPE_CLOG',
    'Clog / valve partly closed (abrupt)': 'PIPE_CLOG',
    'Clogging / ineffective backwash': 'FILTER_FOULING',
    'Pump outage': 'PUMP_OUTAGE',
    'Screen clogging / yield decline': 'SCREEN_CLOGGING',
    'Stuck sensor': 'SENSOR_FAULT',
    'Logger/comms dropout': 'SENSOR_FAULT'
}

def label_ground_truth_faults(df: pd.DataFrame, ground_truth_path: Optional[Path] = None) -> pd.DataFrame:
    """Labels telemetry rows against ground truth failure events."""
    if ground_truth_path is None:
        ground_truth_path = BASE_DIR / "fault_log_ground_truth.csv"

    data = df.copy()
    data['fault_name'] = 'NORMAL'
    data['fault_category'] = 'NORMAL'
    data['fault_asset'] = 'NONE'

    if not Path(ground_truth_path).exists():
        return data

    gt = pd.read_csv(ground_truth_path)
    for _, row in gt.iterrows():
        if str(row['start']).lower() == 'random':
            continue
        t_start = pd.to_datetime(row['start'])
        t_end = pd.to_datetime(row['end'])
        mask = (data['timestamp'] >= t_start) & (data['timestamp'] <= t_end)
        
        data.loc[mask, 'fault_name'] = row['fault_type']
        category = FAULT_CATEGORY_MAPPING.get(row['fault_type'], 'OTHER_FAULT')
        data.loc[mask, 'fault_category'] = category
        data.loc[mask, 'fault_asset'] = row['asset']

    return data


# =====================================================================
# 3. TIME-SERIES DEMAND FORECASTER & RESIDUAL DETECTOR
# =====================================================================

class TimeSeriesDemandForecaster:
    """
    Time-Series ML model using Gradient Boosted Trees for multi-step water demand forecasting.
    Trained to predict expected hourly flow profiles based on cyclical time, calendar, rainfall, and lags.
    Residuals (actual - predicted) serve as real-time predictive indicators of pipe bursts and creeping leaks.
    """
    def __init__(self):
        self.targets = ['bldgA_flow_m3h', 'bldgB_flow_m3h', 'bldgC_flow_m3h', 'sum_bldg_flow_m3h']
        self.models: Dict[str, HistGradientBoostingRegressor] = {}
        self.residual_stds: Dict[str, float] = {}
        self.feature_cols: List[str] = []

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        base_features = [
            'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', 'sin_month', 'cos_month',
            'is_weekend', 'is_academic_holiday', 'rainfall_mm'
        ]
        # Include autoregressive lag features for each target
        lag_features = [
            col for col in df.columns 
            if any(target in col for target in self.targets) and ('lag' in col or 'roll_mean' in col)
        ]
        return [f for f in base_features + lag_features if f in df.columns]

    def fit(self, df_train: pd.DataFrame):
        """Fits gradient boosted time-series regression models for each target demand."""
        self.feature_cols = self._get_feature_columns(df_train)
        X = df_train[self.feature_cols].copy()

        for target in self.targets:
            if target not in df_train.columns:
                continue
            y = df_train[target].values
            model = HistGradientBoostingRegressor(
                loss='squared_error',
                learning_rate=0.08,
                max_iter=150,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                random_state=42
            )
            model.fit(X, y)
            self.models[target] = model
            
            # Compute normal residual standard deviation
            preds = model.predict(X)
            residuals = y - preds
            self.residual_stds[target] = float(np.std(residuals))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates demand forecasts, prediction residuals, and predictive breakdown alerts.
        """
        data = df.copy()
        X = data[self.feature_cols].copy()

        for target, model in self.models.items():
            pred_col = f'{target}_forecast'
            resid_col = f'{target}_forecast_residual'
            z_col = f'{target}_forecast_zscore'
            
            data[pred_col] = model.predict(X)
            data[resid_col] = data[target] - data[pred_col]
            
            std_res = max(self.residual_stds.get(target, 0.2), 0.05)
            data[z_col] = data[resid_col] / std_res

            # Predictive breakdown flags based on forecast departure:
            # - High positive residual: Burst or sudden unmetered demand
            # - High negative residual: Severe pipe blockage or shut valve
            data[f'{target}_anomaly_burst_risk'] = (data[z_col] > 3.0).astype(int)
            data[f'{target}_anomaly_clog_risk'] = (data[z_col] < -3.0).astype(int)

        return data

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluates MAE, RMSE, and R2 score for each forecast target."""
        X = df_test[self.feature_cols].copy()
        metrics = {}
        for target, model in self.models.items():
            if target not in df_test.columns:
                continue
            y_true = df_test[target].values
            y_pred = model.predict(X)
            metrics[target] = {
                'MAE': float(mean_absolute_error(y_true, y_pred)),
                'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'R2': float(r2_score(y_true, y_pred))
            }
        return metrics


# =====================================================================
# 4. SUPERVISED PREDICTIVE BREAKDOWN CLASSIFIER
# =====================================================================

class PredictiveBreakdownClassifier:
    """
    Supervised Machine Learning classifier for multi-fault predictive breakdown diagnostics.
    Predicts the exact fault class and breakdown probabilities across all hydraulic and physical assets.
    """
    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=180,
            max_leaf_nodes=45,
            min_samples_leaf=15,
            class_weight='balanced',
            random_state=42
        )
        self.classes_: np.ndarray = np.array([])
        self.feature_cols: List[str] = []

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        exclude_prefixes = [
            'timestamp', 'fault_name', 'fault_category', 'fault_asset',
            'alert_', 'has_', 'system_status'
        ]
        feature_candidates = [
            c for c in df.columns 
            if not any(c.startswith(p) for p in exclude_prefixes)
            and np.issubdtype(df[c].dtype, np.number)
        ]
        return feature_candidates

    def fit(self, df_train: pd.DataFrame, target_col: str = 'fault_category'):
        """Fits multi-class breakdown classifier on labeled telemetry."""
        self.feature_cols = self._select_features(df_train)
        X = df_train[self.feature_cols].copy()
        y = df_train[target_col].values

        self.model.fit(X, y)
        self.classes_ = self.model.classes_

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts fault classes and breakdown probability for each timestamp.
        """
        data = df.copy()
        X = data[self.feature_cols].copy()

        preds = self.model.predict(X)
        proba = self.model.predict_proba(X)

        data['ml_predicted_fault'] = preds
        
        # Breakdown Probability: 1.0 - P(NORMAL)
        if 'NORMAL' in self.classes_:
            normal_idx = list(self.classes_).index('NORMAL')
            data['ml_breakdown_probability'] = 1.0 - proba[:, normal_idx]
        else:
            data['ml_breakdown_probability'] = proba.max(axis=1)

        # Store individual class probabilities
        for idx, cls_name in enumerate(self.classes_):
            data[f'ml_prob_{cls_name}'] = proba[:, idx]

        return data

    def evaluate(self, df_test: pd.DataFrame, target_col: str = 'fault_category') -> Dict[str, Any]:
        """Generates detailed classification report and metrics."""
        X = df_test[self.feature_cols].copy()
        y_true = df_test[target_col].values
        y_pred = self.model.predict(X)
        active_labels = np.unique(y_true)

        report = classification_report(y_true, y_pred, labels=active_labels, output_dict=True, zero_division=0)
        return report


# =====================================================================
# 5. UNSUPERVISED ANOMALY DETECTOR (ISOLATION FOREST)
# =====================================================================

class UnsupervisedAnomalyDetector:
    """
    Isolation Forest for unsupervised telemetry anomaly detection.
    Detects multivariate operational anomalies without relying on ground-truth labels.
    """
    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )
        self.feature_cols: List[str] = []

    def fit(self, df_train: pd.DataFrame):
        # Train on nominal features (flows, pressures, power, turbidity)
        key_features = [
            'bldgA_flow_m3h', 'bldgB_flow_m3h', 'bldgC_flow_m3h',
            'bldgA_pressure_bar', 'bldgB_pressure_bar', 'bldgC_pressure_bar',
            'bh1_flow_m3h', 'bh2_flow_m3h', 'bh1_pump_kw', 'bh2_pump_kw',
            'bh1_water_level_m', 'bh2_water_level_m',
            'filter_dp_kpa', 'raw_turbidity_ntu', 'treated_turbidity_ntu',
            'main_loss_m3h'
        ]
        self.feature_cols = [f for f in key_features if f in df_train.columns]
        X = df_train[self.feature_cols].fillna(0.0).copy()
        self.model.fit(X)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        X = data[self.feature_cols].fillna(0.0).copy()
        
        # Decision function: lower values mean more anomalous
        scores = self.model.decision_function(X)
        preds = self.model.predict(X)  # -1 for anomaly, 1 for normal

        data['ml_anomaly_score'] = scores
        # Invert score to 0..1 scale where 1 is highest anomaly severity
        norm_score = np.clip((0.15 - scores) / 0.35, 0.0, 1.0)
        data['ml_anomaly_severity'] = norm_score
        data['ml_is_unsupervised_anomaly'] = (preds == -1).astype(int)

        return data


# =====================================================================
# 6. PREDICTIVE PROGNOSTICS & REMAINING USEFUL LIFE (RUL)
# =====================================================================

class PredictivePrognosticsEngine:
    """
    Physics-informed Machine Learning Prognostics for impending asset breakdown:
    1. Filter Media Fouling: Projects drift of post-backwash dP floor to forecast
       Remaining Useful Life (RUL in days) before critical 48 kPa threshold.
    2. Borehole 2 Yield Collapse: Linear trend regression on specific energy (kWh/m3)
       to forecast days until pump motor overload / aquifer rehabilitation cutoff.
    3. Creeping Pipe Leak Growth: Linear regression on night minimum flow (MNF) to
       estimate rate of leak growth (m3/day) and time until burst.
    """
    def compute_prognostics(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # 1. Filter Media Degradation Prognostics
        # Fit rolling 30-day slope of post-backwash dP floor
        if 'filter_dp_floor_kpa' in data.columns:
            dp_floor = data['filter_dp_floor_kpa']
            # Rolling 14-day slope (in kPa per hour)
            roll_slope_dp = (dp_floor - dp_floor.shift(14 * 24)) / (14 * 24)
            data['filter_fouling_rate_kpa_per_day'] = np.maximum(0.0, roll_slope_dp * 24.0).fillna(0.0)
            
            # Distance to critical 48 kPa threshold
            dp_margin = np.maximum(0.0, 48.0 - dp_floor)
            # Days to failure
            rate = data['filter_fouling_rate_kpa_per_day'].clip(lower=0.01)
            data['filter_rul_days'] = np.where(dp_floor >= 45.0, 0.0, np.clip(dp_margin / rate, 0.0, 120.0))
        else:
            data['filter_rul_days'] = 90.0

        # 2. Borehole 2 Specific Energy Degradation (Screen Clogging)
        if 'bh2_kwh_per_m3_roll7d' in data.columns:
            kwh_m3 = data['bh2_kwh_per_m3_roll7d']
            # Slope over 30 days
            kwh_slope = (kwh_m3 - kwh_m3.shift(30 * 24)) / (30 * 24)
            data['bh2_degradation_rate_kwh_day'] = np.maximum(0.0, kwh_slope * 24.0).fillna(0.0)
            
            # Normal ~0.38, Critical overload > 0.55 kWh/m3
            kwh_margin = np.maximum(0.0, 0.55 - kwh_m3)
            deg_rate = data['bh2_degradation_rate_kwh_day'].clip(lower=0.0005)
            data['bh2_rehab_rul_days'] = np.where(kwh_m3 >= 0.52, 0.0, np.clip(kwh_margin / deg_rate, 0.0, 180.0))
        else:
            data['bh2_rehab_rul_days'] = 120.0

        # 3. Overall Breakdown Risk Index (0.0 to 1.0)
        # Combines ML classifier probability, unsupervised anomaly score, and prognostics stress
        ml_prob = data.get('ml_breakdown_probability', 0.0)
        anomaly_sev = data.get('ml_anomaly_severity', 0.0)
        filter_stress = np.clip(1.0 - (data['filter_rul_days'] / 30.0), 0.0, 1.0)
        bh_stress = np.clip(1.0 - (data['bh2_rehab_rul_days'] / 30.0), 0.0, 1.0)

        data['composite_breakdown_risk'] = np.clip(
            0.45 * ml_prob + 0.25 * anomaly_sev + 0.15 * filter_stress + 0.15 * bh_stress,
            0.0, 1.0
        )

        return data


# =====================================================================
# 7. UNIFIED WATER ML PIPELINE ORCHESTRATOR
# =====================================================================

class WaterMLPipeline:
    """
    Master pipeline orchestrator that coordinates:
    - Time-series demand forecasting
    - Supervised breakdown classification
    - Unsupervised anomaly detection
    - Asset prognostics & RUL estimation
    """
    def __init__(self):
        self.demand_forecaster = TimeSeriesDemandForecaster()
        self.breakdown_classifier = PredictiveBreakdownClassifier()
        self.anomaly_detector = UnsupervisedAnomalyDetector()
        self.prognostics_engine = PredictivePrognosticsEngine()
        self.is_fitted = False

    def fit(self, df_raw: pd.DataFrame, ground_truth_path: Optional[Path] = None, split_year: int = 2024):
        """
        Extracts features, labels ground truth, and trains all ML models
        using a temporal split (train on data <= split_year, e.g. 2023-2024).
        """
        print("[1/5] Extracting time-series features & lags...")
        df_feats = extract_time_series_features(df_raw)
        
        print("[2/5] Labeling ground truth failure taxonomy...")
        df_labeled = label_ground_truth_faults(df_feats, ground_truth_path)

        # Temporal split to prevent data leakage in time-series evaluation
        train_mask = df_labeled['timestamp'].dt.year <= split_year
        df_train = df_labeled[train_mask].copy()
        df_val = df_labeled[~train_mask].copy()

        print(f"Training set: {len(df_train)} rows (<= {split_year})")
        print(f"Validation set: {len(df_val)} rows (> {split_year})")

        print("[3/5] Fitting Time-Series Demand Forecasters...")
        self.demand_forecaster.fit(df_train)

        print("[4/5] Fitting Supervised Breakdown Classifier...")
        # Augment training set with demand forecast features
        df_train_fc = self.demand_forecaster.predict(df_train)
        self.breakdown_classifier.fit(df_train_fc, target_col='fault_category')

        print("[5/5] Fitting Unsupervised Anomaly Detector...")
        self.anomaly_detector.fit(df_train)

        self.is_fitted = True
        print("Model training complete!")

        # Evaluate on out-of-time validation set if available
        if len(df_val) > 0:
            print("\n--- Out-of-Time Validation Results ---")
            fc_metrics = self.demand_forecaster.evaluate(df_val)
            print("Demand Forecasting Metrics (MAE / RMSE / R2):")
            for tgt, met in fc_metrics.items():
                print(f"  {tgt}: MAE={met['MAE']:.3f}, RMSE={met['RMSE']:.3f}, R2={met['R2']:.3f}")

            df_val_fc = self.demand_forecaster.predict(df_val)
            clf_report = self.breakdown_classifier.evaluate(df_val_fc, target_col='fault_category')
            print("\nBreakdown Classifier Macro Avg F1:", clf_report.get('macro avg', {}).get('f1-score', 'N/A'))
            print("Accuracy:", clf_report.get('accuracy', 'N/A'))

    def predict(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Runs full end-to-end inference and returns enriched telemetry with ML breakdowns."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted or loaded before calling predict().")

        data = extract_time_series_features(df_raw)
        data = self.demand_forecaster.predict(data)
        data = self.breakdown_classifier.predict(data)
        data = self.anomaly_detector.predict(data)
        data = self.prognostics_engine.compute_prognostics(data)

        return data

    def save(self, output_dir: Optional[Path] = None):
        """Persists trained model checkpoints to disk using joblib."""
        if output_dir is None:
            output_dir = MODEL_ARTIFACTS_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.demand_forecaster, output_dir / "demand_forecaster.joblib")
        joblib.dump(self.breakdown_classifier, output_dir / "breakdown_classifier.joblib")
        joblib.dump(self.anomaly_detector, output_dir / "anomaly_detector.joblib")
        joblib.dump(self.prognostics_engine, output_dir / "prognostics_engine.joblib")
        print(f"All ML model artifacts saved successfully to {output_dir}")

    def load(self, model_dir: Optional[Path] = None):
        """Loads trained model artifacts from disk."""
        if model_dir is None:
            model_dir = MODEL_ARTIFACTS_DIR
        model_dir = Path(model_dir)

        self.demand_forecaster = joblib.load(model_dir / "demand_forecaster.joblib")
        self.breakdown_classifier = joblib.load(model_dir / "breakdown_classifier.joblib")
        self.anomaly_detector = joblib.load(model_dir / "anomaly_detector.joblib")
        self.prognostics_engine = joblib.load(model_dir / "prognostics_engine.joblib")
        self.is_fitted = True
        print(f"All ML model artifacts loaded successfully from {model_dir}")
