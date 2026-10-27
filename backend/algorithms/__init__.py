"""算法注册表聚合层。

对外暴露与旧 `algorithms.py` 完全一致的符号 (CLASSIFICATION_ALGORITHMS /
REGRESSION_ALGORITHMS / coerce_params / get_algorithm 等), 保证零回归;
并新增自研算法 (source="custom") 与无监督算法注册表。
"""
from __future__ import annotations

from typing import Dict, List

from .base import (
    AlgorithmDef,
    ParamDef,
    algorithm_options,
    coerce_params,
)
from .custom import CUSTOM_CLASSIFICATION, CUSTOM_REGRESSION
from .sklearn_models import CLASSIFICATION_ALGORITHMS as _SKLEARN_CLF
from .sklearn_models import REGRESSION_ALGORITHMS as _SKLEARN_REG
from .unsupervised_models import UNSUPERVISED_ALGORITHMS

# 监督算法注册表 = sklearn 实现 + 自研实现
CLASSIFICATION_ALGORITHMS: Dict[str, AlgorithmDef] = {**_SKLEARN_CLF, **CUSTOM_CLASSIFICATION}
REGRESSION_ALGORITHMS: Dict[str, AlgorithmDef] = {**_SKLEARN_REG, **CUSTOM_REGRESSION}

__all__ = [
    "AlgorithmDef",
    "ParamDef",
    "coerce_params",
    "get_algorithm",
    "get_unsupervised_algorithm",
    "CLASSIFICATION_ALGORITHMS",
    "REGRESSION_ALGORITHMS",
    "UNSUPERVISED_ALGORITHMS",
    "algorithms_meta",
]


def _registry(task: str) -> Dict[str, AlgorithmDef]:
    if task == "classification":
        return CLASSIFICATION_ALGORITHMS
    if task == "regression":
        return REGRESSION_ALGORITHMS
    if task == "unsupervised":
        return UNSUPERVISED_ALGORITHMS
    raise ValueError(f"未知任务类型: {task}")


def get_algorithm(task: str, key: str) -> AlgorithmDef:
    """根据任务类型与算法键名获取算法定义。"""
    from fastapi import HTTPException

    registry = _registry(task)
    if key not in registry:
        raise HTTPException(404, f"算法 '{key}' 不存在或与任务类型不匹配")
    return registry[key]


def get_unsupervised_algorithm(key: str) -> AlgorithmDef:
    return get_algorithm("unsupervised", key)


def algorithms_meta() -> Dict[str, List[dict]]:
    """返回全部算法的元数据, 供前端构建下拉与来源切换。"""
    return {
        "classification": algorithm_options(CLASSIFICATION_ALGORITHMS),
        "regression": algorithm_options(REGRESSION_ALGORITHMS),
        "unsupervised": algorithm_options(UNSUPERVISED_ALGORITHMS),
    }
