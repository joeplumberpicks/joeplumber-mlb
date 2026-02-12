from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class NumpyLogisticRegression:
    lr: float = 0.05
    epochs: int = 2000
    reg_lambda: float = 1e-3

    coef_: np.ndarray | None = None
    intercept_: float = 0.0
    medians_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def _fit_preprocess(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        X_imp = np.where(np.isnan(X), med, X)

        mean = X_imp.mean(axis=0)
        std = X_imp.std(axis=0)
        std = np.where(std <= 1e-12, 1.0, std)

        self.medians_ = med
        self.mean_ = mean
        self.std_ = std
        return (X_imp - mean) / std

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.medians_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Model preprocessing is not fitted")
        X = np.asarray(X, dtype=float)
        X_imp = np.where(np.isnan(X), self.medians_, X)
        return (X_imp - self.mean_) / self.std_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyLogisticRegression":
        Xs = self._fit_preprocess(X)
        y = np.asarray(y, dtype=float)

        n, p = Xs.shape
        w = np.zeros(p, dtype=float)
        b = 0.0

        for _ in range(self.epochs):
            z = Xs @ w + b
            p_hat = _sigmoid(z)
            err = p_hat - y

            grad_w = (Xs.T @ err) / n + self.reg_lambda * w
            grad_b = float(err.mean())

            w -= self.lr * grad_w
            b -= self.lr * grad_b

        self.coef_ = w
        self.intercept_ = b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not fitted")
        Xs = self.transform(X)
        p1 = _sigmoid(Xs @ self.coef_ + self.intercept_)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])
