"""多算法对比: 在同一数据集/同一划分上训练多个算法并排名。

被 REST /api/compare 与 WebSocket /ws/train 复用;
progress_cb 接收事件字典 (可直接推送给前端):
- {"type": "progress", "index", "total", "algorithm", "algorithm_name", "source"}
- {"type": "loss", "algorithm", "iter", "loss"}      (自研 GD 模型逐轮损失)
- {"type": "result", "algorithm", "result" | "error"}
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from algorithms import get_algorithm
from datasets import detect_task
from pipeline import fit_and_evaluate, preprocess, validate_columns
from schemas import CompareRequest

ProgressCB = Optional[Callable[[Dict[str, Any]], None]]


def run_comparison(df: pd.DataFrame, req: CompareRequest, progress_cb: ProgressCB = None) -> Dict[str, Any]:
    """批量训练 req.algorithms 中的算法, 返回对比结果与排名。"""
    task = req.task
    if task == "auto":
        task = detect_task(df[req.target])
    if task not in ("classification", "regression"):
        raise ValueError("多算法对比仅支持分类/回归任务")

    validate_columns(df, req.features, req.target)
    X, y, classes, _, _ = preprocess(df, req.features, req.target, task)

    if not req.algorithms:
        raise ValueError("请至少选择一个算法")

    metric_key = "accuracy" if task == "classification" else "r2"
    total = len(req.algorithms)
    results: List[Dict[str, Any]] = []

    for i, sel in enumerate(req.algorithms):
        def emit(event: Dict[str, Any]) -> None:
            if progress_cb is not None:
                progress_cb(event)

        try:
            alg = get_algorithm(task, sel.key)
            emit({
                "type": "progress", "index": i, "total": total,
                "algorithm": sel.key, "algorithm_name": alg.name, "source": alg.source,
            })
            loss_cb = None
            if progress_cb is not None:
                def loss_cb(it: int, loss: float, _key: str = sel.key) -> None:
                    progress_cb({
                        "type": "loss", "algorithm": _key,
                        "iter": int(it), "loss": round(float(loss), 6),
                    })
            res = fit_and_evaluate(
                X, y, classes, task, sel.key, sel.params,
                req.test_size, req.random_state, req.standardize, req.cv_folds,
                progress_cb=loss_cb,
            )
            results.append(res)
            emit({"type": "result", "algorithm": sel.key, "result": res})
        except Exception as exc:  # noqa: BLE001 - 单个算法失败不中断整体对比
            err = {"algorithm": sel.key, "error": str(exc)}
            results.append(err)
            emit({"type": "result", "algorithm": sel.key, "error": str(exc)})

    succeeded = [r for r in results if "metrics" in r]
    ranking = sorted(succeeded, key=lambda r: r["metrics"][metric_key], reverse=True)

    return {
        "task": task,
        "metric_key": metric_key,
        "classes": classes,
        "features": list(X.columns),
        "results": results,
        "ranking": [
            {
                "rank": i + 1,
                "algorithm": r["algorithm"],
                "algorithm_name": r["algorithm_name"],
                "source": r["source"],
                "score": r["metrics"][metric_key],
                "metrics": r["metrics"],
            }
            for i, r in enumerate(ranking)
        ],
    }
