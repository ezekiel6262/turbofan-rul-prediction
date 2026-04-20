"""
evaluate.py
===========
Model evaluation utilities.

Usage
-----
from src.evaluate import evaluate_model, print_results_table
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSE, MAE, and R² for a set of predictions.

    Parameters
    ----------
    y_true : np.ndarray — ground truth RUL values
    y_pred : np.ndarray — predicted RUL values

    Returns
    -------
    dict with keys: rmse, mae, r2
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {'rmse': round(rmse, 4), 'mae': round(mae, 4), 'r2': round(r2, 4)}


def print_results_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Print a formatted table of model results.

    Parameters
    ----------
    results : dict of {model_name: {'train': metrics, 'test': metrics}}
    """
    rows = []
    for model_name, stages in results.items():
        row = {'Model': model_name}
        for stage, metrics in stages.items():
            row[f'{stage}_RMSE'] = metrics['rmse']
            row[f'{stage}_R2']   = metrics['r2']
            row[f'{stage}_MAE']  = metrics['mae']
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    return df


def score_summary(name: str, y_tr: np.ndarray, p_tr: np.ndarray,
                  y_te: np.ndarray, p_te: np.ndarray) -> dict:
    """Helper: return train+test metrics for one model."""
    return {
        'Model': name,
        'Train_RMSE': round(float(np.sqrt(mean_squared_error(y_tr, p_tr))), 4),
        'Test_RMSE':  round(float(np.sqrt(mean_squared_error(y_te, p_te))), 4),
        'Train_R2':   round(float(r2_score(y_tr, p_tr)), 4),
        'Test_R2':    round(float(r2_score(y_te, p_te)), 4),
        'Train_MAE':  round(float(mean_absolute_error(y_tr, p_tr)), 4),
        'Test_MAE':   round(float(mean_absolute_error(y_te, p_te)), 4),
    }
