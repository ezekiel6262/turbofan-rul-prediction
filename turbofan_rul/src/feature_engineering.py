"""
feature_engineering.py
=======================
Feature selection and scaling utilities for CMAPSS data.

Usage
-----
from src.feature_engineering import select_features, scale_features

X_train, X_test, feature_cols = select_features(train, test)
X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List


SENSOR_COLS = [f's{i}' for i in range(1, 22)]


def select_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    corr_threshold: float = 0.3,
    std_threshold: float = 0.5,
    include_cycle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Select informative sensors using two criteria:
      1. Remove near-constant sensors (std < std_threshold)
      2. Keep sensors with |Pearson r| >= corr_threshold vs RUL

    Parameters
    ----------
    train : pd.DataFrame  — training set with RUL column
    test  : pd.DataFrame  — test set with RUL column
    corr_threshold : float  — minimum absolute Pearson correlation
    std_threshold  : float  — minimum standard deviation to keep sensor
    include_cycle  : bool   — whether to include cycle count as a feature

    Returns
    -------
    X_train      : np.ndarray
    X_test       : np.ndarray
    feature_cols : list of selected column names
    """
    # Step 1: remove flat sensors
    std_vals = train[SENSOR_COLS].std()
    flat = std_vals[std_vals < std_threshold].index.tolist()
    if flat:
        print(f"Dropped {len(flat)} flat sensors: {flat}")

    # Step 2: correlation filter
    candidate_sensors = [s for s in SENSOR_COLS if s not in flat]
    corr = train[candidate_sensors + ['RUL']].corr()['RUL'].drop('RUL')
    useful = corr[abs(corr) >= corr_threshold].index.tolist()

    feature_cols = useful.copy()
    if include_cycle:
        feature_cols.append('cycle')

    print(f"Selected {len(feature_cols)} features: {feature_cols}")
    print("\nCorrelation with RUL for selected sensors:")
    for s in useful:
        print(f"  {s:4s}  r = {corr[s]:+.3f}")

    X_train = train[feature_cols].values
    X_test  = test[feature_cols].values

    return X_train, X_test, feature_cols


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Apply Min-Max scaling (0–1) fitted on training data only.

    Parameters
    ----------
    X_train : np.ndarray
    X_test  : np.ndarray

    Returns
    -------
    X_train_sc : np.ndarray  — scaled training features
    X_test_sc  : np.ndarray  — scaled test features (same scaler)
    scaler     : MinMaxScaler — fitted scaler (save for deployment)
    """
    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print(f"Features scaled to [0, 1]. Shape: {X_train_sc.shape}")
    return X_train_sc, X_test_sc, scaler


def make_sequence_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window: int = 30,
    clip: int = 125,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences for LSTM input.

    For each engine, create overlapping windows of `window` cycles.
    The label for each window is the RUL at the last step in the window.

    Parameters
    ----------
    df          : pd.DataFrame with engine_id, cycle, features, and RUL
    feature_cols: list of feature column names
    window      : number of time steps per sequence
    clip        : RUL clip value

    Returns
    -------
    X : np.ndarray of shape (n_sequences, window, n_features)
    y : np.ndarray of shape (n_sequences,)
    """
    X_list, y_list = [], []

    for eid, grp in df.groupby('engine_id'):
        grp = grp.sort_values('cycle')
        feats = grp[feature_cols].values
        rul   = grp['RUL'].values

        for i in range(len(feats) - window + 1):
            X_list.append(feats[i:i + window])
            y_list.append(rul[i + window - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"Sequence windows: X={X.shape}  y={y.shape}")
    return X, y
