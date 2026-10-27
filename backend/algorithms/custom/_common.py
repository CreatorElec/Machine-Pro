"""自研算法公共工具: 极简标准化器与 softmax。

仅本包内部使用, 用于保证基于梯度下降的自研模型数值稳定。
"""
from __future__ import annotations

import numpy as np


class Standardizer:
    """按列零均值单位方差标准化 (内部使用)。"""

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def softmax(logits: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax (按行)。"""
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)
