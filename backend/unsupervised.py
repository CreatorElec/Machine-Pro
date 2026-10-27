"""无监督任务编排: 聚类评估 (silhouette / inertia / 肘部法则) 与 PCA 降维。

仅依赖特征列 (无目标列), 返回绘图与评估数据。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from algorithms import coerce_params, get_unsupervised_algorithm
from pipeline import clean_feature_matrix
from schemas import UnsupervisedRequest

_MAX_SCATTER = 800


def _subsample(rng: np.random.Generator, n: int) -> np.ndarray:
    if n <= _MAX_SCATTER:
        return np.arange(n)
    return np.sort(rng.choice(n, _MAX_SCATTER, replace=False))


def _to_2d(Xs: np.ndarray, random_state: int) -> np.ndarray:
    """把特征矩阵压到 2D 用于散点图 (超过 2 维时用 PCA)。"""
    if Xs.shape[1] <= 2:
        return Xs[:, :2] if Xs.shape[1] == 2 else np.c_[Xs[:, 0], np.zeros(len(Xs))]
    return PCA(n_components=2, random_state=random_state).fit_transform(Xs)


def run_unsupervised(df: pd.DataFrame, req: UnsupervisedRequest) -> Dict[str, Any]:
    """执行聚类或降维, 返回评估指标与可视化数据。"""
    missing = [c for c in req.features if c not in df.columns]
    if missing:
        raise ValueError(f"特征列不存在: {missing}")
    if len(req.features) < 1:
        raise ValueError("请至少选择一个特征列")

    X = clean_feature_matrix(df, req.features)
    Xs = StandardScaler().fit_transform(X) if req.standardize else X.to_numpy(dtype=float)

    alg = get_unsupervised_algorithm(req.algorithm)
    params = coerce_params(alg, req.params)
    est = alg.estimator(**params)

    result: Dict[str, Any] = {
        "algorithm": alg.key,
        "algorithm_name": alg.name,
        "source": alg.source,
        "params": params,
        "standardize": req.standardize,
        "n_samples": int(Xs.shape[0]),
        "n_features": int(Xs.shape[1]),
        "features": list(X.columns),
    }

    rng = np.random.default_rng(req.random_state)

    # ---------------- PCA 降维 ----------------
    if alg.base_key == "pca":
        est.fit(Xs)
        evr = [round(float(v), 4) for v in est.explained_variance_ratio_]
        result["kind"] = "pca"
        result["explained_variance_ratio"] = evr
        result["cumulative_variance"] = [
            round(float(v), 4) for v in np.cumsum(est.explained_variance_ratio_)
        ]
        coords = est.transform(Xs) if hasattr(est, "transform") else None
        if coords is not None and coords.shape[1] >= 2:
            idx = _subsample(rng, coords.shape[0])
            result["scatter"] = [
                {"x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4)}
                for i in idx
            ]
            result["x_name"] = f"PC1 ({evr[0] * 100:.1f}%)"
            result["y_name"] = f"PC2 ({evr[1] * 100:.1f}%)" if len(evr) > 1 else "PC2"
        return result

    # ---------------- 聚类 ----------------
    labels = est.fit_predict(Xs) if hasattr(est, "fit_predict") else est.fit(Xs).labels_
    labels = np.asarray(labels)
    uniq = sorted(set(labels.tolist()))
    n_clusters = len([u for u in uniq if u != -1])

    result["kind"] = "cluster"
    result["n_clusters"] = n_clusters
    result["cluster_sizes"] = {str(u): int((labels == u).sum()) for u in uniq}
    if hasattr(est, "inertia_"):
        result["inertia"] = round(float(est.inertia_), 4)

    # silhouette (排除噪声点 -1; 至少 2 个簇且非全部同簇)
    if -1 in uniq:
        mask = labels != -1
        X_eval, lab_eval = Xs[mask], labels[mask]
    else:
        X_eval, lab_eval = Xs, labels
    n_labels = len(set(lab_eval.tolist()))
    if 2 <= n_labels < len(lab_eval):
        result["silhouette"] = round(float(silhouette_score(X_eval, lab_eval)), 4)
    else:
        result["silhouette"] = None

    coords = _to_2d(Xs, req.random_state)
    idx = _subsample(rng, Xs.shape[0])
    result["scatter"] = [
        {"x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4),
         "cluster": int(labels[i])}
        for i in idx
    ]
    result["x_name"] = "特征 1" if Xs.shape[1] <= 2 else "PC1"
    result["y_name"] = "特征 2" if Xs.shape[1] <= 2 else "PC2"

    # K-Means 肘部法则数据 (K=1..8 的 inertia)
    if alg.base_key == "kmeans":
        elbow = []
        for k in range(1, 9):
            km = alg.estimator(n_clusters=k, n_init=5, max_iter=200,
                               random_state=req.random_state)
            km.fit(Xs)
            elbow.append({"k": k, "inertia": round(float(km.inertia_), 2)})
        result["elbow"] = elbow

    return result
