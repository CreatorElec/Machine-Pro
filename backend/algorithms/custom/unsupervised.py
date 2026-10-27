"""自研无监督算法: KMeans (k-means++ 初始化) 与 PCA (SVD 分解)。"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin


class MyKMeans(BaseEstimator, ClusterMixin):
    """K-Means 聚类, k-means++ 初始化 + Lloyd 迭代。"""

    def __init__(self, n_clusters: int = 3, max_iter: int = 300, n_init: int = 10,
                 random_state: int = 42, tol: float = 1e-6):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.random_state = int(random_state)
        self.tol = float(tol)

    def _kmeans_plusplus(self, X: np.ndarray, rng) -> np.ndarray:
        n = X.shape[0]
        centers = [X[rng.integers(n)]]
        for _ in range(1, self.n_clusters):
            d2 = np.min([((X - c) ** 2).sum(axis=1) for c in centers], axis=0)
            probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
            centers.append(X[rng.choice(n, p=probs)])
        return np.array(centers)

    def _lloyd(self, X: np.ndarray, centers: np.ndarray):
        for _ in range(self.max_iter):
            D = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = D.argmin(axis=1)
            new_centers = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                for k in range(self.n_clusters)
            ])
            if np.allclose(new_centers, centers, atol=self.tol):
                centers = new_centers
                break
            centers = new_centers
        D = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = D.argmin(axis=1)
        inertia = float(D[np.arange(X.shape[0]), labels].sum())
        return labels, inertia, centers

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(max(1, self.n_init)):
            labels, inertia, centers = self._lloyd(X, self._kmeans_plusplus(X, rng))
            if best is None or inertia < best[1]:
                best = (labels, inertia, centers)
        self.labels_, self.inertia_, self.cluster_centers_ = best
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        D = ((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
        return D.argmin(axis=1)


class MyPCA(BaseEstimator, TransformerMixin):
    """主成分分析, 基于中心化后的 SVD 分解。"""

    def __init__(self, n_components: int = 2):
        self.n_components = int(n_components)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = int(min(self.n_components, Vt.shape[0]))
        self.components_ = Vt[:k]
        n = X.shape[0]
        explained = (S[:k] ** 2) / max(n - 1, 1)
        total = (S ** 2).sum() / max(n - 1, 1)
        self.explained_variance_ = explained
        self.explained_variance_ratio_ = explained / total if total > 0 else np.zeros(k)
        return self

    def transform(self, X) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)
