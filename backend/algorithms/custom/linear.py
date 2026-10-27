"""自研线性/近邻类算法 (纯 numpy 实现)。

统一遵循 sklearn 风格接口: fit / predict / predict_proba,
并暴露 classes_ 与 coef_ 以便复用平台的评估与特征重要性逻辑。

基于梯度下降的模型 (逻辑回归 / 线性 SVM) 在内部对特征做标准化以保证数值稳定,
标准化统计量保存于 _scaler_, 预测时复用同一变换。
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from ._common import Standardizer, softmax


# --------------------------------------------------------------------------- #
# K 近邻
# --------------------------------------------------------------------------- #
class MyKNNClassifier(BaseEstimator, ClassifierMixin):
    """K 近邻分类器: 多数投票 / 距离加权投票。"""

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", metric: str = "minkowski"):
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)
        self._class_index = {c: i for i, c in enumerate(self.classes_)}
        return self

    def _distances(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - self.X_[None, :, :]
        if self.metric == "manhattan":
            return np.abs(diff).sum(axis=2)
        return np.sqrt((diff ** 2).sum(axis=2))

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        D = self._distances(X)
        k = max(1, min(self.n_neighbors, D.shape[1]))
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        proba = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        for i in range(X.shape[0]):
            neigh_d = D[i, idx[i]]
            w = 1.0 / (neigh_d + 1e-12) if self.weights == "distance" else np.ones(k)
            for ni, wi in zip(idx[i], w):
                proba[i, self._class_index[self.y_[ni]]] += wi
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class MyKNNRegressor(BaseEstimator, RegressorMixin):
    """K 近邻回归器: 近邻目标值的均值 / 距离加权均值。"""

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", metric: str = "minkowski"):
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=float)
        return self

    def _distances(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - self.X_[None, :, :]
        if self.metric == "manhattan":
            return np.abs(diff).sum(axis=2)
        return np.sqrt((diff ** 2).sum(axis=2))

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        D = self._distances(X)
        k = max(1, min(self.n_neighbors, D.shape[1]))
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        out = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            neigh_d = D[i, idx[i]]
            vals = self.y_[idx[i]]
            if self.weights == "distance":
                w = 1.0 / (neigh_d + 1e-12)
                out[i] = float(np.sum(w * vals) / np.sum(w))
            else:
                out[i] = float(np.mean(vals))
        return out


# --------------------------------------------------------------------------- #
# 逻辑回归 (多项 softmax + 梯度下降)
# --------------------------------------------------------------------------- #
class MyLogisticRegression(BaseEstimator, ClassifierMixin):
    """多项逻辑回归, 全批量梯度下降 + L2 正则 (softmax 输出)。"""

    def __init__(self, C: float = 1.0, max_iter: int = 500, learning_rate: float = 1.0):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)

    def fit(self, X, y):
        self._scaler_ = Standardizer().fit(X)
        Xs = self._scaler_.transform(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n, d = Xs.shape
        K = len(self.classes_)
        class_index = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_index[v] for v in y])
        Y = np.zeros((n, K))
        Y[np.arange(n), y_idx] = 1.0

        W = np.zeros((d, K))
        b = np.zeros(K)
        # 目标: (1/n) * 交叉熵 + (1/(2*C*n)) * ||W||^2, 使 C 的作用与 sklearn 一致
        l2 = 1.0 / (max(self.C, 1e-12) * max(n, 1))
        lr = self.learning_rate
        emit_every = max(1, self.max_iter // 20)

        for it in range(self.max_iter):
            probs = softmax(Xs @ W + b)
            grad_W = Xs.T @ (probs - Y) / n + l2 * W
            grad_b = (probs - Y).mean(axis=0)
            W -= lr * grad_W
            b -= lr * grad_b
            if (it + 1) % emit_every == 0:
                loss = float(-np.mean(np.log(probs[np.arange(n), y_idx] + 1e-12)) + 0.5 * l2 * np.sum(W ** 2))
                _emit(self, it + 1, loss)

        self.coef_ = W.T            # (K, d)
        self.intercept_ = b
        return self

    def predict_proba(self, X) -> np.ndarray:
        Xs = self._scaler_.transform(X)
        return softmax(Xs @ self.coef_.T + self.intercept_)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# --------------------------------------------------------------------------- #
# 线性回归 / 岭回归 (闭式解)
# --------------------------------------------------------------------------- #
class MyLinearRegression(BaseEstimator, RegressorMixin):
    """普通最小二乘线性回归 (lstsq 正规方程)。"""

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        A = np.hstack([X, np.ones((X.shape[0], 1))]) if self.fit_intercept else X
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.coef_ = sol[: X.shape[1]]
        self.intercept_ = float(sol[-1]) if self.fit_intercept else 0.0
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


class MyRidge(BaseEstimator, RegressorMixin):
    """岭回归 (L2 正则闭式解, 数据中心化)。"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean()
        Xc = X - self.x_mean_
        yc = y - self.y_mean_
        d = X.shape[1]
        self.coef_ = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(d), Xc.T @ yc)
        self.intercept_ = float(self.y_mean_ - self.x_mean_ @ self.coef_)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


# --------------------------------------------------------------------------- #
# 线性支持向量机 (合页损失 + 次梯度下降, 一对多)
# --------------------------------------------------------------------------- #
class MyLinearSVM(BaseEstimator, ClassifierMixin):
    """线性核 SVM, 合页损失 + L2 正则, 一对多 (OvR) 策略。"""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, learning_rate: float = 0.001):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)

    def fit(self, X, y):
        self._scaler_ = Standardizer().fit(X)
        Xs = self._scaler_.transform(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        d = Xs.shape[1]
        lr = self.learning_rate

        W = np.zeros((K, d))
        b = np.zeros(K)
        for k, cls in enumerate(self.classes_):
            t = np.where(y == cls, 1.0, -1.0)
            w = np.zeros(d)
            bk = 0.0
            for _ in range(self.max_iter):
                margins = t * (Xs @ w + bk)
                active = margins < 1.0
                # 目标: 0.5*||w||^2 + C * sum(hinge)
                grad_w = w - self.C * (Xs[active] * t[active, None]).sum(axis=0)
                grad_b = -self.C * t[active].sum()
                w -= lr * grad_w
                bk -= lr * grad_b
            W[k] = w
            b[k] = bk
            _emit(self, k + 1, float(np.mean(np.maximum(0.0, 1.0 - t * (Xs @ w + bk)))))

        self.coef_ = W                              # (K, d)
        self.intercept_ = b

        # Platt 缩放: 将每个 OvR 决策值校准为类概率 (供 predict_proba / ROC 使用)
        F = Xs @ W.T + b
        self._platt_A_ = np.zeros(K)
        self._platt_B_ = np.zeros(K)
        for k in range(K):
            self._platt_A_[k], self._platt_B_[k] = _fit_platt(
                F[:, k], (y == self.classes_[k]).astype(float)
            )
        return self

    def decision_function(self, X) -> np.ndarray:
        Xs = self._scaler_.transform(X)
        return Xs @ self.coef_.T + self.intercept_

    def predict_proba(self, X) -> np.ndarray:
        # 用 Platt 缩放把各 OvR 决策值映射为校准概率, 再归一化到和为 1
        F = self.decision_function(X)
        proba = np.empty_like(F, dtype=float)
        for k in range(F.shape[1]):
            proba[:, k] = _sigmoid(self._platt_A_[k] * F[:, k] + self._platt_B_[k])
        total = proba.sum(axis=1, keepdims=True)
        return proba / np.where(total <= 1e-12, 1.0, total)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid 函数 σ(z) = 1 / (1 + e^{-z})。"""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _fit_platt(f: np.ndarray, t: np.ndarray, max_iter: int = 100) -> tuple[float, float]:
    """Platt 缩放: 拟合 P(t=1) = σ(A·f + B), 用 Newton 法最大化对数似然。

    f: 训练样本的 OvR 决策值 (n,);  t: 0/1 目标 (是否属于该类)。
    返回 (A, B)。
    """
    f = np.asarray(f, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    prior = float(np.clip(t.mean(), 1e-6, 1.0 - 1e-6))
    A, B = 0.0, float(np.log(prior / (1.0 - prior)))

    for _ in range(max_iter):
        p = _sigmoid(A * f + B)
        g0 = float(np.sum((p - t) * f))   # ∂L/∂A (交叉熵损失梯度)
        g1 = float(np.sum(p - t))         # ∂L/∂B
        w = p * (1.0 - p)
        h00 = float(np.sum(w * f * f))
        h01 = float(np.sum(w * f))
        h11 = float(np.sum(w))
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        # Newton 步: Δ = -H^{-1}·g
        dA = -(h11 * g0 - h01 * g1) / det
        dB = -(-h01 * g0 + h00 * g1) / det
        A += dA
        B += dB
        if abs(dA) < 1e-6 and abs(dB) < 1e-6:
            break
    return A, B


def _emit(model, iteration: int, loss: float) -> None:
    """若模型设置了 progress_cb, 则回调 (用于 WebSocket 推送逐轮损失)。"""
    cb = getattr(model, "progress_cb", None)
    if cb is not None:
        try:
            cb(iteration, float(loss))
        except Exception:  # noqa: BLE001
            pass
