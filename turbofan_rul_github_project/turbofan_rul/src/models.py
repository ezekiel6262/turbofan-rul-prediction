"""
models.py
=========
Model definitions and training helpers for turbofan RUL prediction.

Classical ML models (scikit-learn) and LSTM (TensorFlow/Keras).
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any


# ─────────────────────────────────────────────
# Classical ML
# ─────────────────────────────────────────────

def get_classical_models() -> Dict[str, Any]:
    """
    Return a dictionary of scikit-learn regression models.

    Returns
    -------
    dict : {model_name: unfitted_model}
    """
    return {
        'Linear Regression': LinearRegression(),
        'Lasso Regression':  Lasso(alpha=0.05, max_iter=5000),
        'Decision Tree':     DecisionTreeRegressor(
                                 random_state=42,
                                 max_depth=8
                             ),
        'Random Forest':     RandomForestRegressor(
                                 n_estimators=100,
                                 max_depth=10,
                                 random_state=42,
                                 n_jobs=-1
                             ),
        'SVR':               SVR(
                                 kernel='rbf',
                                 C=100,
                                 gamma='scale',
                                 epsilon=1.0
                             ),
    }


def train_classical_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, Any]:
    """
    Fit all models in the dict and return fitted versions.

    Parameters
    ----------
    models  : dict from get_classical_models()
    X_train : np.ndarray — scaled feature matrix
    y_train : np.ndarray — RUL labels

    Returns
    -------
    fitted_models : dict of {name: fitted_model}
    """
    fitted = {}
    for name, model in models.items():
        print(f"Training {name}...", end=' ', flush=True)
        model.fit(X_train, y_train)
        fitted[name] = model
        print("done")
    return fitted


# ─────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────

def build_lstm(n_features: int, units_1: int = 64, units_2: int = 32,
               dropout: float = 0.2, learning_rate: float = 0.001):
    """
    Build a two-layer stacked LSTM for RUL regression.

    Architecture:
        LSTM(units_1, return_sequences=True)
        → Dropout
        → LSTM(units_2)
        → Dropout
        → Dense(16, relu)
        → Dense(1)

    Parameters
    ----------
    n_features    : int   — number of input features
    units_1       : int   — LSTM units in first layer (default 64)
    units_2       : int   — LSTM units in second layer (default 32)
    dropout       : float — dropout rate (default 0.2)
    learning_rate : float — Adam learning rate (default 0.001)

    Returns
    -------
    model : compiled Keras model
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            units_1,
            return_sequences=True,
            input_shape=(None, n_features)
        ),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(units_2),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='mse',
        metrics=['mae']
    )

    model.summary()
    return model


def train_lstm(model, X_train: np.ndarray, y_train: np.ndarray,
               epochs: int = 50, batch_size: int = 256,
               validation_split: float = 0.15, patience: int = 8):
    """
    Train the LSTM model with early stopping.

    Parameters
    ----------
    model            : compiled Keras model from build_lstm()
    X_train          : np.ndarray shape (samples, timesteps, features)
    y_train          : np.ndarray shape (samples,)
    epochs           : int   — max training epochs
    batch_size       : int   — mini-batch size
    validation_split : float — fraction of training data for validation
    patience         : int   — early stopping patience

    Returns
    -------
    history : Keras History object
    """
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6
        ),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nTraining complete. Best val_loss at epoch "
          f"{np.argmin(history.history['val_loss']) + 1}")
    return history
