"""机器学习训练与评估流水线。

对外提供:
- preprocess / clean_feature_matrix: 特征清洗与编码
- train_and_evaluate: 单模型训练 + 评估 (REST /api/train)
- fit_and_evaluate: 可复用的单模型执行器 (供多算法对比与 WebSocket 复用)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from algorithms import AlgorithmDef, coerce_params, get_algorithm
from datasets import detect_task
from schemas import TrainRequest


# --------------------------------------------------------------------------- #
# 特征清洗
# --------------------------------------------------------------------------- #
def clean_feature_matrix(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """仅清洗特征 (数值列中位数填充, 类别列众数填充 + one-hot), 返回 float 矩阵。

    无监督路径复用此函数 (无目标列)。
    """
    X = df[features].copy()
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    for c in numeric_cols:
        med = X[c].median()
        X[c] = X[c].fillna(med if pd.notna(med) else 0.0)

    if categorical_cols:
        for c in categorical_cols:
            X[c] = X[c].astype(str).replace({"nan": "缺失", "None": "缺失", "": "缺失"})
        X = pd.get_dummies(X, columns=categorical_cols)

    return X.astype(float)


def preprocess(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    task: str,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[List[str]], List[str], List[str]]:
    """特征清洗 + 目标编码。

    返回: (X, y, 类别名列表, 数值列, 类别列)
    """
    X = df[features].copy()
    y = df[target].copy()

    valid = y.notna()
    X, y = X[valid], y[valid]
    if X.empty:
        raise ValueError("目标列缺失值过多, 没有可训练的数据")

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    for c in numeric_cols:
        med = X[c].median()
        X[c] = X[c].fillna(med if pd.notna(med) else 0.0)

    if categorical_cols:
        for c in categorical_cols:
            X[c] = X[c].astype(str).replace({"nan": "缺失", "None": "缺失", "": "缺失"})
        X = pd.get_dummies(X, columns=categorical_cols)

    X = X.astype(float)

    classes: Optional[List[str]] = None
    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        classes = [str(c) for c in le.classes_]
    else:
        y = y.astype(float).to_numpy()

    return X, y, classes, numeric_cols, categorical_cols


# --------------------------------------------------------------------------- #
# 模型构建与重要性
# --------------------------------------------------------------------------- #
def build_model(alg: AlgorithmDef, params: Dict[str, Any], standardize: bool):
    """构建估计器; standardize=True 时在前面接 StandardScaler。"""
    est = alg.estimator(**params)
    if standardize:
        return make_pipeline(StandardScaler(), est)
    return est


def _final_estimator(model: Any) -> Any:
    return model.steps[-1][1] if hasattr(model, "steps") else model


def _feature_importance(model: Any, feature_names: List[str]) -> List[Dict[str, Any]]:
    """提取模型的特征重要性 (树模型 / 线性模型), 需为最终估计器。"""
    est = _final_estimator(model)
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float).ravel()
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float)
        imp = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        imp = imp.ravel()
    else:
        return []

    order = np.argsort(imp)[::-1][:15]
    return [
        {"feature": feature_names[i], "importance": round(float(imp[i]), 6)}
        for i in order
        if imp[i] > 1e-9
    ]


def _label(value: Any, classes: Optional[List[str]], task: str) -> Any:
    if task == "classification":
        return classes[int(value)]
    return round(float(value), 4)


def _classification_metrics(y_true, y_pred) -> Dict[str, Any]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }


def _regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    return {
        "mse": round(float(mean_squared_error(y_true, y_pred)), 4),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def cross_validate_model(
    X: pd.DataFrame,
    y: np.ndarray,
    alg: AlgorithmDef,
    params: Dict[str, Any],
    task: str,
    folds: int,
    random_state: int,
    standardize: bool,
) -> Dict[str, Any]:
    """K 折交叉验证, 返回各指标 mean / std。"""
    est = build_model(alg, params, standardize)
    if task == "classification":
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        scoring = {"accuracy": "accuracy", "precision_macro": "precision_macro",
                   "recall_macro": "recall_macro", "f1_macro": "f1_macro"}
    else:
        cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        scoring = {"r2": "r2", "neg_mse": "neg_mean_squared_error", "neg_mae": "neg_mean_absolute_error"}

    scores = cross_validate(est, np.asarray(X, dtype=float), np.asarray(y), cv=cv, scoring=scoring,
                            error_score="raise")
    out: Dict[str, Any] = {"folds": folds}
    for key, values in scores.items():
        if not key.startswith("test_"):
            continue
        name = key[len("test_"):]
        out[name] = {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4)}
    return out


# --------------------------------------------------------------------------- #
# 训练执行器
# --------------------------------------------------------------------------- #
def fit_and_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    classes: Optional[List[str]],
    task: str,
    algorithm_key: str,
    raw_params: Dict[str, Any],
    test_size: float,
    random_state: int,
    standardize: bool = False,
    cv_folds: int = 0,
    progress_cb: Optional[Callable[[int, float], None]] = None,
    model_hook: Optional[Callable[[Any], None]] = None,
) -> Dict[str, Any]:
    """在已清洗好的 (X, y) 上训练单个模型并返回评估结果。"""
    alg = get_algorithm(task, algorithm_key)
    params = coerce_params(alg, raw_params)

    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    model = build_model(alg, params, standardize)
    if progress_cb is not None:
        setattr(_final_estimator(model), "progress_cb", progress_cb)

    model.fit(X_train, y_train)
    if model_hook is not None:
        model_hook(model)

    y_pred = model.predict(X_test)

    result: Dict[str, Any] = {
        "task": task,
        "algorithm": alg.key,
        "algorithm_name": alg.name,
        "algorithm_description": alg.description,
        "source": alg.source,
        "params": params,
        "standardize": standardize,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "features": list(X.columns),
        "n_features_used": int(X.shape[1]),
    }

    if task == "classification":
        result["classes"] = classes
        result["metrics"] = _classification_metrics(y_test, y_pred)
        result["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        result["classification_report"] = classification_report(
            y_test, y_pred, target_names=classes, output_dict=True, zero_division=0
        )
    else:
        result["metrics"] = _regression_metrics(y_test, y_pred)

    result["feature_importance"] = _feature_importance(model, X.columns.tolist())
    result["predictions"] = [
        {"index": i, "true": _label(y_test[i], classes, task), "pred": _label(y_pred[i], classes, task)}
        for i in range(min(10, len(y_test)))
    ]

    if cv_folds and cv_folds >= 2:
        result["cv"] = cross_validate_model(X, y, alg, params, task, cv_folds, random_state, standardize)

    return result


def validate_columns(df: pd.DataFrame, features: List[str], target: str) -> None:
    """校验特征列与目标列是否存在。"""
    missing_features = [c for c in features if c not in df.columns]
    if missing_features:
        raise ValueError(f"特征列不存在: {missing_features}")
    if target not in df.columns:
        raise ValueError(f"目标列 '{target}' 不存在")


def train_and_evaluate(df: pd.DataFrame, req: TrainRequest) -> Dict[str, Any]:
    """执行完整训练流程并返回评估结果 (REST 入口)。"""
    task = req.task
    if task == "auto":
        task = detect_task(df[req.target])

    validate_columns(df, req.features, req.target)
    X, y, classes, _, _ = preprocess(df, req.features, req.target, task)

    return fit_and_evaluate(
        X, y, classes, task, req.algorithm, req.params,
        req.test_size, req.random_state, req.standardize, req.cv_folds,
    )
