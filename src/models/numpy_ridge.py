from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class NumpyRidgeRegression:
    alpha: float = 1.0

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyRidgeRegression":
        Xs = self._fit_preprocess(X)
        y = np.asarray(y, dtype=float)

        n, p = Xs.shape
        xtx = Xs.T @ Xs
        reg = self.alpha * np.eye(p)
        xty = Xs.T @ y

        coef = np.linalg.solve(xtx + reg, xty)
        intercept = float(y.mean() - (Xs @ coef).mean())

        self.coef_ = coef
        self.intercept_ = intercept
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not fitted")
        Xs = self.transform(X)
        return Xs @ self.coef_ + self.intercept_
