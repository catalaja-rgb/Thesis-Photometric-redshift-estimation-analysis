import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


@dataclass
class MLResult:
    z_pred: np.ndarray
    metrics: Dict[str, float]


def make_features(x: np.ndarray, x_err: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Simple, robust features that work even with negative fluxes:
      phi_b = x_b / sigma_b
    Shape: (N, B)
    """
    x = np.asarray(x, float)
    x_err = np.asarray(x_err, float)
    return x / np.clip(x_err, eps, None)


def compute_metrics(z_true: np.ndarray, z_pred: np.ndarray) -> Dict[str, float]:
    z_true = np.asarray(z_true, float)
    z_pred = np.asarray(z_pred, float)

    dz = (z_pred - z_true) / (1.0 + z_true)
    mae = float(np.mean(np.abs(z_pred - z_true)))
    bias = float(np.mean(dz))
    scatter = float(np.std(dz))
    outlier = float(np.mean(np.abs(dz) > 0.15))
    return {"mae": mae, "bias": bias, "scatter": scatter, "outlier": outlier}


def fit_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 25,
    weights: str = "distance",
) -> np.ndarray:
    """
    kNN regression baseline.
    """
    model = KNeighborsRegressor(n_neighbors=k, weights=weights)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int = 300,
    random_state: int = 0,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 2,
) -> np.ndarray:
    """
    Random Forest regression baseline.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def run_empirical_models(
    x: np.ndarray,
    x_err: np.ndarray,
    z: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 0,
    knn_k: int = 25,
) -> Tuple[MLResult, MLResult]:
    """
    Train/test split, run kNN and RF, return predictions + metrics.
    """
    X = make_features(x, x_err)
    y = np.asarray(z, float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    z_knn = fit_knn(X_train, y_train, X_test, k=knn_k)
    z_rf = fit_random_forest(X_train, y_train, X_test, random_state=random_state)

    res_knn = MLResult(z_pred=z_knn, metrics=compute_metrics(y_test, z_knn))
    res_rf = MLResult(z_pred=z_rf, metrics=compute_metrics(y_test, z_rf))
    return res_knn, res_rf
