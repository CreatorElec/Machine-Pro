"""机器学习训练与评估流水线。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from algorithms import AlgorithmDef, coerce_params, get_algorithm
from datasets import detect_task
from schemas import TrainRequest


def preprocess(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    task: str,
) -> tuple[pd.DataFrame, np.ndarray, Optional[List[str]], List[str], List[str]]:
    """特征清洗与编码。

    返回: (X, y, 类别名列表, 数值列, 类别列)
    """
    X = df[features].copy()
    y = df[target].copy()

    # 丢弃目标缺失的行
    valid = y.notna()
    X, y = X[valid], y[valid]
    if X.empty:
        raise ValueError("目标列缺失值过多, 没有可训练的数据")

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # 数值列: 中位数填充
    for c in numeric_cols:
        med = X[c].median()
        X[c] = X[c].fillna(med if pd.notna(med) else 0.0)

    # 类别列: 字符串化 + 众数填充 + one-hot 编码
    if categorical_cols:
        for c in categorical_cols:
            X[c] = X[c].astype(str).replace({"nan": "缺失", "None": "缺失", "": "缺失"})
        X = pd.get_dummies(X, columns=categorical_cols)

    X = X.astype(float)

    # 目标列编码
    classes: Optional[List[str]] = None
    if task == "classification":
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        classes = [str(c) for c in le.classes_]
        y = y_encoded
    else:
        y = y.astype(float).to_numpy()

    return X, y, classes, numeric_cols, categorical_cols


def _feature_importance(model: Any, feature_names: List[str]) -> List[Dict[str, Any]]:
    """提取模型的特征重要性 (树模型 / 线性模型)。"""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float).ravel()
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
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


def train_and_evaluate(df: pd.DataFrame, req: TrainRequest) -> Dict[str, Any]:
    """执行完整训练流程并返回评估结果。"""
    task = req.task
    if task == "auto":
        task = detect_task(df[req.target])

    # 校验列名
    missing_features = [c for c in req.features if c not in df.columns]
    if missing_features:
        raise ValueError(f"特征列不存在: {missing_features}")
    if req.target not in df.columns:
        raise ValueError(f"目标列 '{req.target}' 不存在")

    X, y, classes, _, _ = preprocess(df, req.features, req.target, task)

    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=req.test_size, random_state=req.random_state, stratify=stratify
    )

    alg: AlgorithmDef = get_algorithm(task, req.algorithm)
    params = coerce_params(alg, req.params)
    model = alg.estimator(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result: Dict[str, Any] = {
        "task": task,
        "algorithm": alg.key,
        "algorithm_name": alg.name,
        "algorithm_description": alg.description,
        "params": params,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "features": req.features,
        "n_features_used": X.shape[1],
    }

    if task == "classification":
        result["classes"] = classes
        result["metrics"] = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        }
        result["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        result["classification_report"] = classification_report(
            y_test, y_pred, target_names=classes, output_dict=True, zero_division=0
        )
    else:
        result["metrics"] = {
            "mse": round(float(mean_squared_error(y_test, y_pred)), 4),
            "rmse": round(float(root_mean_squared_error(y_test, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "r2": round(float(r2_score(y_test, y_pred)), 4),
        }

    result["feature_importance"] = _feature_importance(model, X.columns.tolist())

    result["predictions"] = [
        {"index": i, "true": _label(y_test[i], classes, task), "pred": _label(y_pred[i], classes, task)}
        for i in range(min(10, len(y_test)))
    ]

    return result
