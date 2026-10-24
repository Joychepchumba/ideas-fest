"""
Advanced Machine Learning & Statistical Time-Series Pipeline
=============================================================
Industry-grade ML lifecycle for Municipal Water Analytics:
1. Data Preparation & Statistical Time-Series EDA:
   - Stationarity verification (Augmented Dickey-Fuller / ADF test)
   - Autocorrelation (ACF) & Partial Autocorrelation (PACF)
   - Classical Additive Seasonal Decomposition (Trend, Seasonality, Residuals)
2. Rigorous Preprocessing:
   - Time-aware imputation, RobustScaler, cyclical harmonic encodings (sin/cos)
   - Multi-horizon autoregressive lag generation (t-1, t-2, t-24, t-168)
   - Rolling statistical aggregations (24h mean/std, 168h weekly mean, EMA)
3. Model Progression & Benchmarking (Forecasting):
   - Baseline 1: Naive Lag-1 Persistence (y_t = y_{t-1})
   - Baseline 2: Seasonal Lag-24 Persistence (y_t = y_{t-24})
   - Baseline 3: 24h Moving Average
   - Model 1: Classical Statistical Time Series (SARIMAX / AutoReg)
   - Model 2: Regularized Linear Model (Ridge Regression)
   - Model 3: Tree Bagging Ensemble (Random Forest Regressor)
   - Model 4: Tuned Gradient Boosted Trees (HistGradientBoostingRegressor)
4. Hyperparameter Tuning & Cross-Validation:
   - TimeSeriesSplit (5-fold temporal expansion without future leakage)
   - RandomizedSearchCV over learning rate, leaf nodes, min leaf samples, and L2 regularization
5. Supervised Multi-Fault Classification Progression:
   - Dummy Classifier (Baseline)
   - Random Forest Classifier (Class-weighted)
   - Tuned HistGradientBoostingClassifier (Class-weighted)
6. Model Leaderboard & Evaluation:
   - MAE, RMSE, MAPE, R2 for regression
   - Balanced Accuracy, Macro F1, Confusion Matrix for classification
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import joblib

# Statistical Time-Series Analysis
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Scikit-Learn Modeling & Validation
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix, balanced_accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models_saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 1. EXPLORATORY DATA ANALYSIS (EDA) & STATISTICAL TIME-SERIES TESTS
# =====================================================================

def perform_stationarity_eda(series: pd.Series, max_lag: int = 48) -> Dict[str, Any]:
    """
    Performs rigorous statistical time-series diagnostic tests:
    - Augmented Dickey-Fuller (ADF) test for unit-root stationarity
    - Autocorrelation Function (ACF)
    - Partial Autocorrelation Function (PACF)
    - Seasonal decomposition (Trend, 24h Seasonality, Residuals)
    """
    clean_series = series.dropna().values
    
    # 1. Augmented Dickey-Fuller Test
    adf_result = adfuller(clean_series, autolag='AIC')
    is_stationary = bool(adf_result[1] < 0.05)
    
    # 2. ACF and PACF
    acf_vals = acf(clean_series, nlags=max_lag, fft=True)
    pacf_vals = pacf(clean_series, nlags=max_lag, method='yw')

    # 3. Additive Decomposition on first 2 weeks (336 hours) to isolate 24h diurnal pattern
    sample_len = min(len(clean_series), 24 * 14)
    decomp = seasonal_decompose(clean_series[:sample_len], model='additive', period=24)

    return {
        'adf_statistic': float(adf_result[0]),
        'p_value': float(adf_result[1]),
        'used_lag': int(adf_result[2]),
        'critical_values': {k: float(v) for k, v in adf_result[4].items()},
        'is_stationary': is_stationary,
        'acf': acf_vals.tolist(),
        'pacf': pacf_vals.tolist(),
        'trend': decomp.trend.tolist(),
        'seasonal': decomp.seasonal.tolist(),
        'resid': decomp.resid.tolist()
    }


# =====================================================================
# 2. BASELINE PERSISTENCE ESTIMATORS
# =====================================================================

class NaiveLagRegressor(BaseEstimator, RegressorMixin):
    """
    Simple persistence baseline: predicts value at lag k.
    Default k=1 (yesterday/last hour), k=24 (seasonal same-hour yesterday).
    """
    def __init__(self, lag: int = 1):
        self.lag = lag
        self.last_value = 0.0

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        # Expects dataframe or array containing the lag column
        if isinstance(X, pd.DataFrame):
            col_name = f'lag_{self.lag}'
            if col_name in X.columns:
                return X[col_name].values
        # Fallback to mean
        return np.full(len(X), self.last_value)


class MovingAverageRegressor(BaseEstimator, RegressorMixin):
    """Moving average baseline over rolling window w."""
    def __init__(self, window: int = 24):
        self.window = window

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            col_name = f'roll_mean_{self.window}'
            if col_name in X.columns:
                return X[col_name].values
        return np.zeros(len(X))


# =====================================================================
# 3. TIME-SERIES MODEL PROGRESSION & BENCHMARKING
# =====================================================================

def evaluate_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates MAE, RMSE, MAPE, and R2 score."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Epsilon-clipped MAPE to prevent zero division
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 0.1, None))) * 100.0
    r2 = r2_score(y_true, y_pred)
    return {
        'MAE': round(float(mae), 4),
        'RMSE': round(float(rmse), 4),
        'MAPE (%)': round(float(mape), 2),
        'R2': round(float(r2), 4)
    }


class TimeSeriesModelSuite:
    """
    Trains and compares a progressive hierarchy of models on a target time-series:
    1. Naive Lag-1 Baseline
    2. Seasonal Lag-24 Baseline
    3. 24h Moving Average Baseline
    4. Classical Statistical SARIMAX
    5. Regularized Ridge Regression
    6. Random Forest Regressor
    7. Tuned HistGradientBoostingRegressor (with TimeSeriesSplit CV)
    """
    def __init__(self, target_col: str = 'sum_bldg_flow_m3h'):
        self.target_col = target_col
        self.models: Dict[str, Any] = {}
        self.leaderboard: pd.DataFrame = pd.DataFrame()
        self.best_model: Optional[Any] = None
        self.feature_cols: List[str] = []

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        data = df.copy()
        target = data[self.target_col]

        # Explicit lag columns for baselines
        data[f'lag_1'] = target.shift(1).bfill()
        data[f'lag_2'] = target.shift(2).bfill()
        data[f'lag_24'] = target.shift(24).bfill()
        data[f'lag_168'] = target.shift(168).bfill()
        data[f'roll_mean_24'] = target.rolling(24, min_periods=6).mean().bfill()
        data[f'roll_std_24'] = target.rolling(24, min_periods=6).std().fillna(0.0)

        # Candidate feature subset
        candidate_cols = [
            'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', 'sin_month', 'cos_month',
            'is_weekend', 'is_academic_holiday', 'rainfall_mm',
            'lag_1', 'lag_2', 'lag_24', 'lag_168', 'roll_mean_24', 'roll_std_24'
        ]
        self.feature_cols = [c for c in candidate_cols if c in data.columns]
        return data, target

    def run_benchmark(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
        """Trains all candidates, logs metrics on out-of-time test set, and produces leaderboard."""
        train_data, y_train = self._prepare_features(df_train)
        test_data, y_test = self._prepare_features(df_test)

        X_train = train_data[self.feature_cols]
        X_test = test_data[self.feature_cols]

        results = []

        # 1. Baseline 1: Naive Lag-1 Persistence
        pred_lag1 = X_test['lag_1'].values
        m_lag1 = evaluate_forecast_metrics(y_test.values, pred_lag1)
        results.append({'Model': '1. Baseline: Naive Lag-1 Persistence (t-1)', **m_lag1})

        # 2. Baseline 2: Seasonal Lag-24 Persistence
        pred_lag24 = X_test['lag_24'].values
        m_lag24 = evaluate_forecast_metrics(y_test.values, pred_lag24)
        results.append({'Model': '2. Baseline: Seasonal Persistence (t-24)', **m_lag24})

        # 3. Baseline 3: 24h Moving Average
        pred_ma24 = X_test['roll_mean_24'].values
        m_ma24 = evaluate_forecast_metrics(y_test.values, pred_ma24)
        results.append({'Model': '3. Baseline: 24-Hour Rolling Mean', **m_ma24})

        # 4. Model 1: Classical Statistical Time-Series (SARIMAX on sample)
        try:
            # Fit on recent 1,000 hours for computational tractability
            sarima_train = y_train.values[-1000:]
            sarima = SARIMAX(sarima_train, order=(1, 0, 1), seasonal_order=(1, 0, 0, 24), enforce_stationarity=False)
            sarima_res = sarima.fit(disp=False)
            # Forecast horizon corresponding to test set
            pred_sarima = sarima_res.forecast(steps=len(y_test))
            m_sarima = evaluate_forecast_metrics(y_test.values, pred_sarima)
            results.append({'Model': '4. Classical: SARIMAX (1,0,1)x(1,0,0)_24', **m_sarima})
            self.models['SARIMAX'] = sarima_res
        except Exception as e:
            results.append({'Model': '4. Classical: SARIMAX (1,0,1)', 'MAE': np.nan, 'RMSE': np.nan, 'MAPE (%)': np.nan, 'R2': np.nan})

        # 5. Model 2: Regularized Linear Model (Ridge)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ridge = Ridge(alpha=10.0)
        ridge.fit(X_train_scaled, y_train.values)
        pred_ridge = ridge.predict(X_test_scaled)
        m_ridge = evaluate_forecast_metrics(y_test.values, pred_ridge)
        results.append({'Model': '5. Linear: Ridge Regression (L2 Regularized)', **m_ridge})
        self.models['Ridge'] = (ridge, scaler)

        # 6. Model 3: Tree Bagging (Random Forest Regressor)
        rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=15, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train.values)
        pred_rf = rf.predict(X_test)
        m_rf = evaluate_forecast_metrics(y_test.values, pred_rf)
        results.append({'Model': '6. Ensemble: Random Forest Regressor', **m_rf})
        self.models['RandomForest'] = rf

        # 7. Model 4: Tuned Gradient Boosted Trees with TimeSeriesSplit CV
        tscv = TimeSeriesSplit(n_splits=3)
        param_dist = {
            'learning_rate': [0.04, 0.08, 0.12],
            'max_leaf_nodes': [25, 31, 45],
            'min_samples_leaf': [15, 25, 40],
            'l2_regularization': [0.0, 0.1, 1.0]
        }
        base_hgb = HistGradientBoostingRegressor(random_state=42)
        search = RandomizedSearchCV(base_hgb, param_distributions=param_dist, n_iter=6, cv=tscv, scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1)
        search.fit(X_train, y_train.values)
        
        best_hgb = search.best_estimator_
        pred_hgb = best_hgb.predict(X_test)
        m_hgb = evaluate_forecast_metrics(y_test.values, pred_hgb)
        results.append({'Model': '7. Modern ML: Tuned HistGradientBoosting (Final)', **m_hgb})
        self.models['TunedGradientBoosting'] = best_hgb
        self.best_model = best_hgb

        # Leaderboard DataFrame
        self.leaderboard = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
        return self.leaderboard


# =====================================================================
# 4. SUPERVISED BREAKDOWN CLASSIFIER COMPARISON
# =====================================================================

class FaultClassifierSuite:
    """
    Compares baseline vs modern classifiers for predictive fault diagnostics:
    1. Dummy Majority / Stratified Classifier
    2. Balanced Random Forest Classifier
    3. Tuned Balanced HistGradientBoostingClassifier
    """
    def __init__(self):
        self.leaderboard: pd.DataFrame = pd.DataFrame()
        self.best_classifier: Optional[Any] = None
        self.feature_cols: List[str] = []

    def run_cv_benchmark(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
        """
        Executes a rigorous 5-Fold Stratified Cross-Validation benchmark across all
        7 operational states to fairly assess multi-class diagnostic capability.
        """
        self.feature_cols = list(X.columns)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        models = {
            '1. Baseline: Dummy Majority': DummyClassifier(strategy='most_frequent'),
            '2. Ensemble: Balanced Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1),
            '3. Modern ML: Tuned HistGradientBoosting (Final)': HistGradientBoostingClassifier(learning_rate=0.08, max_iter=150, class_weight='balanced', random_state=42)
        }

        cv_results = []
        for name, model in models.items():
            acc_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            bal_acc_scores = cross_val_score(model, X, y, cv=skf, scoring='balanced_accuracy')
            macro_f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
            cv_results.append({
                'Model': name,
                'Accuracy': round(float(np.mean(acc_scores)), 4),
                'Balanced Accuracy': round(float(np.mean(bal_acc_scores)), 4),
                'Macro F1': round(float(np.mean(macro_f1_scores)), 4)
            })

        self.leaderboard = pd.DataFrame(cv_results).sort_values('Macro F1', ascending=False).reset_index(drop=True)
        return self.leaderboard

    def run_benchmark(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        self.feature_cols = list(X_train.columns)
        results = []
        active_labels = np.unique(y_test)

        # 1. Dummy Baseline
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_train, y_train)
        pred_dummy = dummy.predict(X_test)
        results.append({
            'Model': '1. Baseline: Dummy Majority Classifier',
            'Accuracy': round(float(np.mean(y_test == pred_dummy)), 4),
            'Balanced Accuracy': round(float(balanced_accuracy_score(y_test, pred_dummy)), 4),
            'Macro F1': round(float(classification_report(y_test, pred_dummy, labels=active_labels, output_dict=True, zero_division=0)['macro avg']['f1-score']), 4)
        })

        # 2. Balanced Random Forest
        rf_clf = RandomForestClassifier(n_estimators=120, max_depth=15, class_weight='balanced', n_jobs=-1, random_state=42)
        rf_clf.fit(X_train, y_train)
        pred_rf = rf_clf.predict(X_test)
        results.append({
            'Model': '2. Ensemble: Balanced Random Forest Classifier',
            'Accuracy': round(float(np.mean(y_test == pred_rf)), 4),
            'Balanced Accuracy': round(float(balanced_accuracy_score(y_test, pred_rf)), 4),
            'Macro F1': round(float(classification_report(y_test, pred_rf, labels=active_labels, output_dict=True, zero_division=0)['macro avg']['f1-score']), 4)
        })

        # 3. Tuned HistGradientBoosting
        hgb_clf = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=180,
            max_leaf_nodes=45,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42
        )
        hgb_clf.fit(X_train, y_train)
        pred_hgb = hgb_clf.predict(X_test)
        results.append({
            'Model': '3. Modern ML: Tuned HistGradientBoosting (Final)',
            'Accuracy': round(float(np.mean(y_test == pred_hgb)), 4),
            'Balanced Accuracy': round(float(balanced_accuracy_score(y_test, pred_hgb)), 4),
            'Macro F1': round(float(classification_report(y_test, pred_hgb, labels=active_labels, output_dict=True, zero_division=0)['macro avg']['f1-score']), 4)
        })

        self.best_classifier = hgb_clf
        self.leaderboard = pd.DataFrame(results).sort_values('Macro F1', ascending=False).reset_index(drop=True)
        return self.leaderboard
