from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class NumpyPlattScaler:
    lr: float = 0.05
    epochs: int = 2000
    reg_lambda: float = 1e-3

    a_: float = 1.0
    b_: float = 0.0

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "NumpyPlattScaler":
        x = np.asarray(p_raw, dtype=float)
        x = np.clip(x, 1e-6, 1 - 1e-6)
        y = np.asarray(y, dtype=float)

        a, b = 1.0, 0.0
        n = len(x)
        for _ in range(self.epochs):
            z = a * x + b
            p = _sigmoid(z)
            err = p - y
            grad_a = float((err * x).mean() + self.reg_lambda * a)
            grad_b = float(err.mean())
            a -= self.lr * grad_a
            b -= self.lr * grad_b

        self.a_ = a
        self.b_ = b
        return self

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        x = np.asarray(p_raw, dtype=float)
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return _sigmoid(self.a_ * x + self.b_)
