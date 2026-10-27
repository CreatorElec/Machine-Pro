"""网格搜索调参 (GridSearchCV)。

- standardize=True 时使用 make_pipeline(StandardScaler(), 模型),
  参数名自动加管道前缀, 避免数据泄漏;
- 分类用 StratifiedKFold, 回归用 KFold。
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

from algorithms import coerce_params, get_algorithm
from algorithms.base import AlgorithmDef, _coerce_value
from datasets import detect_task
from pipeline import build_model, preprocess, validate_columns
from schemas import TuneRequest


def _coerce_grid(alg: AlgorithmDef, raw_grid: Dict[str, Any]) -> Dict[str, List[Any]]:
    """校验并转换网格参数值 (字符串 -> 正确类型)。"""
    pmap = {p.name: p for p in alg.params}
    grid: Dict[str, List[Any]] = {}
    for name, values in raw_grid.items():
        p = pmap.get(name)
        if p is None:
            raise ValueError(f"算法 '{alg.key}' 没有参数 '{name}'")
        if not isinstance(values, list):
            values = [values]
        grid[name] = [_coerce_value(p, v) for v in values]
    return grid


def grid_search(df: pd.DataFrame, req: TuneRequest) -> Dict[str, Any]:
    """执行网格搜索, 返回最佳参数与全部候选的交叉验证表现。"""
    task = req.task
    if task == "auto":
        task = detect_task(df[req.target])
    if task not in ("classification", "regression"):
        raise ValueError("网格搜索仅支持分类/回归任务")

    validate_columns(df, req.features, req.target)
    X, y, _, _, _ = preprocess(df, req.features, req.target, task)

    alg = get_algorithm(task, req.algorithm)
    grid = _coerce_grid(alg, req.param_grid)
    if not grid:
        raise ValueError("param_grid 不能为空")

    base_params = coerce_params(alg, {})
    model = build_model(alg, base_params, req.standardize)
    if hasattr(model, "steps"):  # Pipeline: 参数名需加最终步骤前缀
        prefix = model.steps[-1][0] + "__"
        param_grid = {prefix + k: v for k, v in grid.items()}
    else:
        param_grid = grid

    if task == "classification":
        cv = StratifiedKFold(req.cv_folds, shuffle=True, random_state=req.random_state)
        scoring = "accuracy"
    else:
        cv = KFold(req.cv_folds, shuffle=True, random_state=req.random_state)
        scoring = "r2"

    gs = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, refit=True)
    gs.fit(np.asarray(X, dtype=float), np.asarray(y))

    rows = []
    for i, params in enumerate(gs.cv_results_["params"]):
        clean = {k.split("__")[-1]: v for k, v in params.items()}
        rows.append({
            "params": clean,
            "mean_score": round(float(gs.cv_results_["mean_test_score"][i]), 4),
            "std_score": round(float(gs.cv_results_["std_test_score"][i]), 4),
            "rank": int(gs.cv_results_["rank_test_score"][i]),
        })
    rows.sort(key=lambda r: (r["rank"], -r["mean_score"]))

    best_params = {k.split("__")[-1]: v for k, v in gs.best_params_.items()}
    return {
        "task": task,
        "algorithm": alg.key,
        "algorithm_name": alg.name,
        "source": alg.source,
        "scoring": scoring,
        "cv_folds": req.cv_folds,
        "standardize": req.standardize,
        "best_params": best_params,
        "best_score": round(float(gs.best_score_), 4),
        "n_candidates": len(rows),
        "results": rows,
    }
