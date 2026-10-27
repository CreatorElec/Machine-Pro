"""自研高斯朴素贝叶斯 (纯 numpy 实现)。"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ._common import softmax


class MyGaussianNB(BaseEstimator, ClassifierMixin):
    """高斯朴素贝叶斯: 假设各特征在类内服从独立正态分布。"""

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = float(var_smoothing)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        epsilon = self.var_smoothing * (X.var(axis=0).max() if X.size else 0.0)

        theta, var, prior = [], [], []
        for c in self.classes_:
            Xc = X[y == c]
            theta.append(Xc.mean(axis=0))
            var.append(Xc.var(axis=0) + epsilon)
            prior.append(Xc.shape[0] / X.shape[0])
        self.theta_ = np.array(theta)
        self.var_ = np.array(var)
        self.class_prior_ = np.array(prior)
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        jll = np.zeros((X.shape[0], len(self.classes_)))
        for k in range(len(self.classes_)):
            var = self.var_[k]
            ll = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            ll -= 0.5 * np.sum(((X - self.theta_[k]) ** 2) / var, axis=1)
            jll[:, k] = ll + np.log(self.class_prior_[k])
        return jll

    def predict_proba(self, X) -> np.ndarray:
        return softmax(self._joint_log_likelihood(np.asarray(X, dtype=float)))

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self._joint_log_likelihood(np.asarray(X, dtype=float)), axis=1)]
