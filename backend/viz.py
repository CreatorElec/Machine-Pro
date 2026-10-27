"""可视化绘图数据生成 (仅返回点坐标/曲线数据, 不生成图片)。

- roc_curve_data:        ROC 曲线 (二分类 + 多分类 OvR) 与 AUC
- learning_curve_data:   学习曲线 (训练/验证分数随样本量变化)
- decision_boundary_data: 决策边界 (2 特征直接用; 多于 2 特征先 PCA 降到 2 维)
- pca_scatter_data:      PCA 2D 散点 (target 可选, 探索性分析)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler, label_binarize

from algorithms import AlgorithmDef, coerce_params, get_algorithm
from datasets import detect_task
from pipeline import build_model, clean_feature_matrix, preprocess, validate_columns
from schemas import VizRequest

_MAX_CURVE_POINTS = 200   # 曲线降采样上限
_MAX_SCATTER = 800        # 散点降采样上限
_GRID_SIZE = 100          # 决策边界网格边长


def _prepare(df: pd.DataFrame, req: VizRequest):
    """公共准备: 任务识别 + 列校验 + 清洗 + 算法与参数解析。"""
    if req.target is None:
        raise ValueError("该可视化需要指定目标列")
    if not req.algorithm:
        raise ValueError("该可视化需要指定算法")
    task = req.task
    if task == "auto":
        task = detect_task(df[req.target])
    if task not in ("classification", "regression"):
        raise ValueError("该可视化仅支持分类/回归任务")
    validate_columns(df, req.features, req.target)
    X, y, classes, _, _ = preprocess(df, req.features, req.target, task)
    alg = get_algorithm(task, req.algorithm)
    params = coerce_params(alg, req.params)
    return task, X, y, classes, alg, params


def _thin(xs: np.ndarray, ys: np.ndarray) -> Tuple[List[float], List[float]]:
    """曲线降采样, 限制返回点数。"""
    n = len(xs)
    if n <= _MAX_CURVE_POINTS:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, _MAX_CURVE_POINTS).astype(int)
    return [round(float(v), 5) for v in xs[idx]], [round(float(v), 5) for v in ys[idx]]


def _subsample(rng: np.random.Generator, n: int) -> np.ndarray:
    if n <= _MAX_SCATTER:
        return np.arange(n)
    return np.sort(rng.choice(n, _MAX_SCATTER, replace=False))


# --------------------------------------------------------------------------- #
# ROC 曲线
# --------------------------------------------------------------------------- #
def roc_curve_data(df: pd.DataFrame, req: VizRequest) -> Dict[str, Any]:
    task, X, y, classes, alg, params = _prepare(df, req)
    if task != "classification":
        raise ValueError("ROC 曲线仅支持分类任务")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=req.test_size, random_state=req.random_state, stratify=y
    )
    model = build_model(alg, params, req.standardize)
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    else:
        raise ValueError(f"算法 '{alg.name}' 不输出概率/决策分数, 无法绘制 ROC 曲线")
    scores = np.asarray(scores, dtype=float)

    model_classes = list(getattr(model, "classes_", range(len(classes))))
    n_classes = len(model_classes)

    curves: List[Dict[str, Any]] = []
    if n_classes == 2:
        pos_scores = scores[:, 1] if scores.ndim == 2 else scores
        fpr, tpr, _ = roc_curve(y_test, pos_scores, pos_label=model_classes[1])
        fx, tx = _thin(fpr, tpr)
        label = classes[1] if classes else str(model_classes[1])
        curves.append({"label": f"{label} (正类)", "fpr": fx, "tpr": tx,
                       "auc": round(float(auc(fpr, tpr)), 4)})
    else:
        if scores.ndim != 2 or scores.shape[1] != n_classes:
            raise ValueError(f"算法 '{alg.name}' 的多分类决策分数不支持 OvR ROC")
        Y = label_binarize(y_test, classes=model_classes)
        for i, c in enumerate(model_classes):
            if Y[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(Y[:, i], scores[:, i])
            fx, tx = _thin(fpr, tpr)
            label = classes[i] if classes else str(c)
            curves.append({"label": f"{label} vs 其余", "fpr": fx, "tpr": tx,
                           "auc": round(float(auc(fpr, tpr)), 4)})

    return {"kind": "roc", "algorithm": alg.key, "algorithm_name": alg.name,
            "curves": curves}


# --------------------------------------------------------------------------- #
# 学习曲线
# --------------------------------------------------------------------------- #
def learning_curve_data(df: pd.DataFrame, req: VizRequest) -> Dict[str, Any]:
    task, X, y, _, alg, params = _prepare(df, req)
    model = build_model(alg, params, req.standardize)

    if task == "classification":
        cv = StratifiedKFold(3, shuffle=True, random_state=req.random_state)
        scoring = "accuracy"
    else:
        cv = KFold(3, shuffle=True, random_state=req.random_state)
        scoring = "r2"

    sizes, train_scores, val_scores = learning_curve(
        model, np.asarray(X, dtype=float), np.asarray(y),
        cv=cv, scoring=scoring, train_sizes=np.linspace(0.2, 1.0, 5),
        shuffle=True, random_state=req.random_state, error_score=np.nan,
    )
    return {
        "kind": "learning_curve",
        "algorithm": alg.key,
        "algorithm_name": alg.name,
        "scoring": scoring,
        "train_sizes": [int(s) for s in sizes],
        "train_mean": [round(float(v), 4) for v in np.nanmean(train_scores, axis=1)],
        "train_std": [round(float(v), 4) for v in np.nanstd(train_scores, axis=1)],
        "val_mean": [round(float(v), 4) for v in np.nanmean(val_scores, axis=1)],
        "val_std": [round(float(v), 4) for v in np.nanstd(val_scores, axis=1)],
    }


# --------------------------------------------------------------------------- #
# 决策边界
# --------------------------------------------------------------------------- #
def decision_boundary_data(df: pd.DataFrame, req: VizRequest) -> Dict[str, Any]:
    task, X, y, classes, alg, params = _prepare(df, req)
    if task != "classification":
        raise ValueError("决策边界仅支持分类任务")

    used_pca = len(req.features) != 2
    if used_pca:
        Xs = StandardScaler().fit_transform(np.asarray(X, dtype=float))
        pca = PCA(n_components=2, random_state=req.random_state)
        X2 = pca.fit_transform(Xs)
        evr = pca.explained_variance_ratio_
        x_name = f"PC1 ({evr[0] * 100:.1f}%)"
        y_name = f"PC2 ({evr[1] * 100:.1f}%)"
        model = build_model(alg, params, False)
    else:
        X2 = np.asarray(X, dtype=float)
        x_name, y_name = req.features[0], req.features[1]
        model = build_model(alg, params, req.standardize)

    X_train, X_test, y_train, y_test = train_test_split(
        X2, y, test_size=req.test_size, random_state=req.random_state, stratify=y
    )
    model.fit(X_train, y_train)
    accuracy = round(float(np.mean(model.predict(X_test) == y_test)), 4)

    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, _GRID_SIZE),
        np.linspace(y_min, y_max, _GRID_SIZE),
    )
    gz = model.predict(np.c_[gx.ravel(), gy.ravel()]).astype(int)

    rng = np.random.default_rng(req.random_state)
    idx = _subsample(rng, len(X2))
    points = [
        {"x": round(float(X2[i, 0]), 4), "y": round(float(X2[i, 1]), 4), "label": int(y[i])}
        for i in idx
    ]

    return {
        "kind": "decision_boundary",
        "algorithm": alg.key,
        "algorithm_name": alg.name,
        "used_pca": used_pca,
        "x_name": x_name,
        "y_name": y_name,
        "classes": classes,
        "accuracy_2d": accuracy,
        "grid": {
            "x": [round(float(v), 4) for v in gx[0]],
            "y": [round(float(v), 4) for v in gy[:, 0]],
            "z": gz.tolist(),
        },
        "points": points,
    }


# --------------------------------------------------------------------------- #
# PCA 2D 散点 (探索性, target 可选)
# --------------------------------------------------------------------------- #
def pca_scatter_data(df: pd.DataFrame, req: VizRequest) -> Dict[str, Any]:
    missing = [c for c in req.features if c not in df.columns]
    if missing:
        raise ValueError(f"特征列不存在: {missing}")

    X = clean_feature_matrix(df, req.features)
    Xs = StandardScaler().fit_transform(X) if req.standardize else X.to_numpy(dtype=float)

    pca = PCA(n_components=2, random_state=req.random_state)
    coords = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_

    labels: Optional[List[Any]] = None
    label_task: Optional[str] = None
    if req.target and req.target in df.columns:
        label_task = detect_task(df[req.target])
        col = df[req.target]
        valid = col.notna()
        keep = valid.to_numpy()
        coords = coords[keep]
        col = col[valid]
        if label_task == "classification":
            labels = [str(v) for v in col.tolist()]
        else:
            labels = [round(float(v), 4) for v in col.tolist()]

    rng = np.random.default_rng(req.random_state)
    idx = _subsample(rng, coords.shape[0])
    points = []
    for i in idx:
        p: Dict[str, Any] = {"x": round(float(coords[i, 0]), 4),
                             "y": round(float(coords[i, 1]), 4)}
        if labels is not None:
            p["label"] = labels[i]
        points.append(p)

    return {
        "kind": "pca_scatter",
        "x_name": f"PC1 ({evr[0] * 100:.1f}%)",
        "y_name": f"PC2 ({evr[1] * 100:.1f}%)",
        "explained_variance_ratio": [round(float(v), 4) for v in evr],
        "label_task": label_task,
        "points": points,
    }
