"""
data_loader.py
==============
Load and preprocess NASA CMAPSS FD001 data.

Usage
-----
from src.data_loader import load_cmapss

train, test, rul_test = load_cmapss(data_dir='data/', clip=125)
"""

import numpy as np
import pandas as pd
from pathlib import Path


COLUMN_NAMES = (
    ['engine_id', 'cycle', 'op1', 'op2', 'op3'] +
    [f's{i}' for i in range(1, 22)]
)

SENSOR_DESCRIPTIONS = {
    's1':  'Total temperature at fan inlet (°R)',
    's2':  'Total temperature at LPC outlet (°R)',
    's3':  'Total temperature at HPC outlet (°R)',
    's4':  'Total temperature at LPT outlet (°R)',
    's5':  'Pressure at fan inlet (psia)',
    's6':  'Total pressure in bypass-duct (psia)',
    's7':  'Total pressure at HPC outlet (psia)',
    's8':  'Physical fan speed (rpm)',
    's9':  'Physical core speed (rpm)',
    's10': 'Engine pressure ratio (P50/P2)',
    's11': 'Static pressure at HPC outlet (psia)',
    's12': 'Ratio of fuel flow to Ps30 (pps/psi)',
    's13': 'Corrected fan speed (rpm)',
    's14': 'Corrected core speed (rpm)',
    's15': 'Bypass ratio',
    's16': 'Burner fuel-air ratio',
    's17': 'Bleed enthalpy',
    's18': 'Required fan speed (rpm)',
    's19': 'Required fan conversion speed (rpm)',
    's20': 'High-pressure turbine cool air flow (lbm/s)',
    's21': 'Low-pressure turbine cool air flow (lbm/s)',
}


def load_cmapss(data_dir: str = 'data/', clip: int = 125):
    """
    Load NASA CMAPSS FD001 dataset and add clipped RUL labels.

    Parameters
    ----------
    data_dir : str
        Path to directory containing train_FD001.txt, test_FD001.txt,
        and RUL_FD001.txt.
    clip : int
        Maximum RUL value (clip at this value). Default 125 cycles.
        Engines are healthy and sensors flat before ~125 cycles from
        failure; clipping focuses the model on the detectable window.

    Returns
    -------
    train : pd.DataFrame  — training set with RUL column
    test  : pd.DataFrame  — test set with RUL column
    rul_test : pd.Series  — ground truth RUL for each test engine
    """
    data_dir = Path(data_dir)

    # Load files
    train = pd.read_csv(
        data_dir / 'train_FD001.txt',
        sep=' ', names=COLUMN_NAMES
    )
    test = pd.read_csv(
        data_dir / 'test_FD001.txt',
        sep=' ', names=COLUMN_NAMES
    )
    rul_test = pd.read_csv(
        data_dir / 'RUL_FD001.txt',
        names=['RUL']
    )['RUL']

    # Add RUL to training set
    train['RUL'] = (
        train
        .groupby('engine_id')['cycle']
        .transform(lambda x: x.max() - x)
        .clip(upper=clip)
    )

    # Add approximate RUL to test set
    test_max = (
        test.groupby('engine_id')['cycle']
        .max()
        .reset_index()
        .rename(columns={'cycle': 'max_cycle'})
    )
    test = test.merge(test_max, on='engine_id')
    test['RUL'] = (test['max_cycle'] - test['cycle']).clip(upper=clip)
    test.drop(columns='max_cycle', inplace=True)

    print(f"Train: {train.shape[0]:,} rows | {train['engine_id'].nunique()} engines")
    print(f"Test : {test.shape[0]:,} rows  | {test['engine_id'].nunique()} engines")
    print(f"RUL range (train): {train['RUL'].min():.0f} – {train['RUL'].max():.0f} cycles")

    return train, test, rul_test


def get_sensor_info() -> pd.DataFrame:
    """Return a DataFrame with sensor names and descriptions."""
    return pd.DataFrame(
        list(SENSOR_DESCRIPTIONS.items()),
        columns=['Sensor', 'Description']
    )
