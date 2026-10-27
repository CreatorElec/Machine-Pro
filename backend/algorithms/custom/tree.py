"""自研 CART 决策树 (分类 + 回归, 纯 numpy 实现)。

特征重要性采用"加权不纯度下降": importance[f] += (n_node / N) * (impurity - weighted_child_impurity),
最后归一化到和为 1。
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

_MAX_BINS = 24


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "proba")

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.proba = None


def _candidate_thresholds(col: np.ndarray) -> np.ndarray:
    uniq = np.unique(col)
    if uniq.size <= 1:
        return np.empty(0)
    if uniq.size > _MAX_BINS:
        uniq = np.unique(np.quantile(uniq, np.linspace(0.0, 1.0, _MAX_BINS)))
    return (uniq[:-1] + uniq[1:]) / 2.0


def _impurity(proba: np.ndarray, criterion: str) -> float:
    p = proba[proba > 0]
    if p.size == 0:
        return 0.0
    if criterion == "entropy":
        return float(-(p * np.log2(p)).sum())
    return float(1.0 - (proba ** 2).sum())


class _TreeBuilder:
    def __init__(self, X, y, mode, criterion, max_depth, min_samples_split, min_samples_leaf):
        self.X = X
        self.y = y
        self.mode = mode                     # "clf" / "reg"
        self.criterion = criterion
        self.max_depth = max_depth           # None = 无限制
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features = X.shape[1]
        self.n_total = X.shape[0]
        self.importances = np.zeros(self.n_features)
        self.n_classes = int(y.max()) + 1 if mode == "clf" else 0

    def _stats(self, y):
        if self.mode == "clf":
            counts = np.bincount(y, minlength=self.n_classes).astype(float)
            proba = counts / counts.sum()
            return float(np.argmax(counts)), proba, _impurity(proba, self.criterion)
        return float(y.mean()), None, float(np.var(y))

    def build(self, idx, depth):
        y = self.y[idx]
        value, proba, impurity = self._stats(y)
        node = _Node()
        node.value = value
        node.proba = proba

        over_depth = self.max_depth is not None and depth >= self.max_depth
        if over_depth or idx.size < self.min_samples_split or idx.size < 2 * self.min_samples_leaf or impurity <= 1e-12:
            return node

        X = self.X[idx]
        best_gain, best_f, best_thr = 0.0, None, None
        for f in range(self.n_features):
            col = X[:, f]
            for thr in _candidate_thresholds(col):
                left = col <= thr
                nL, nR = int(left.sum()), int((~left).sum())
                if nL < self.min_samples_leaf or nR < self.min_samples_leaf:
                    continue
                _, _, impL = self._stats(y[left])
                _, _, impR = self._stats(y[~left])
                gain = impurity - (nL * impL + nR * impR) / idx.size
                if gain > best_gain:
                    best_gain, best_f, best_thr = gain, f, float(thr)

        if best_f is None or best_gain <= 1e-12:
            return node

        self.importances[best_f] += idx.size * best_gain / self.n_total
        left_mask = X[:, best_f] <= best_thr
        node.feature = best_f
        node.threshold = best_thr
        node.left = self.build(idx[left_mask], depth + 1)
        node.right = self.build(idx[~left_mask], depth + 1)
        return node


def _traverse(node: _Node, row: np.ndarray) -> _Node:
    while node.feature is not None:
        node = node.left if row[node.feature] <= node.threshold else node.right
    return node


class MyDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """CART 分类树, 支持 gini / entropy 分裂准则。"""

    def __init__(self, max_depth=None, criterion: str = "gini", min_samples_split: int = 2,
                 min_samples_leaf: int = 1):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        encoded = np.searchsorted(self.classes_, y)
        builder = _TreeBuilder(X, encoded, "clf", self.criterion,
                               self.max_depth, self.min_samples_split, self.min_samples_leaf)
        self.tree_ = builder.build(np.arange(X.shape[0]), 0)
        imp = builder.importances
        total = imp.sum()
        self.feature_importances_ = imp / total if total > 0 else imp
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        proba = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        for i, row in enumerate(X):
            proba[i] = _traverse(self.tree_, row).proba
        return proba

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class MyDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    """CART 回归树, 以方差下降 (平方误差) 作为分裂准则。"""

    def __init__(self, max_depth=None, min_samples_split: int = 2, min_samples_leaf: int = 1):
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        builder = _TreeBuilder(X, y, "reg", "squared_error", self.max_depth,
                               self.min_samples_split, self.min_samples_leaf)
        self.tree_ = builder.build(np.arange(X.shape[0]), 0)
        imp = builder.importances
        total = imp.sum()
        self.feature_importances_ = imp / total if total > 0 else imp
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([_traverse(self.tree_, row).value for row in X], dtype=float)
