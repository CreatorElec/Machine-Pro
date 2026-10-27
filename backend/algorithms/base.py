"""算法注册表基类与元数据定义。

每个算法包含:
- estimator: 估计器类 (sklearn 或自研 numpy 实现, 统一 sklearn 风格接口)
- params: 超参数元数据列表, 前端据此动态生成参数表单
- source: 实现来源 (sklearn / custom), 前端据此做来源切换与对比
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class ParamDef:
    """单个超参数定义。type 取值: int / float / select。"""

    name: str
    label: str
    type: str
    default: Any
    options: List[Any] = field(default_factory=list)
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "type": self.type,
            "default": self.default,
            "options": self.options,
            "min": self.min_value,
            "max": self.max_value,
            "step": self.step,
        }


@dataclass
class AlgorithmDef:
    """算法定义。key 为注册表内唯一键。"""

    key: str
    name: str
    estimator: Callable
    params: List[ParamDef]
    description: str = ""
    source: str = "sklearn"          # sklearn / custom
    task: str = "classification"     # classification / regression / unsupervised
    base_key: str = ""               # 同一算法的不同实现共享, 用于分组展示

    def __post_init__(self) -> None:
        if not self.base_key:
            self.base_key = self.key

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "task": self.task,
            "base_key": self.base_key,
            "params": [p.to_dict() for p in self.params],
        }


def coerce_params(alg: AlgorithmDef, raw_params: Dict[str, Any]) -> Dict[str, Any]:
    """将前端传来的字符串参数转换为正确类型, 未提供/为 null 的参数使用默认值。

    默认值同样经过类型转换 (如 select 类型的 "无限制" -> None)。
    """
    result: Dict[str, Any] = {}
    for p in alg.params:
        v = raw_params.get(p.name)
        if v is None:
            v = p.default
        result[p.name] = _coerce_value(p, v)
    return result


def _coerce_value(p: ParamDef, v: Any) -> Any:
    if v is None:
        return None
    if p.type == "int":
        return int(v)
    if p.type == "float":
        return float(v)
    # select 类型: 特殊值转 None, 数字字符串转数字
    if str(v) in ("无限制", "None", "none", ""):
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        pass
    try:
        return float(v)
    except (ValueError, TypeError):
        pass
    return v


def algorithm_options(registry: Dict[str, AlgorithmDef]) -> List[dict]:
    """把注册表转为前端可用的列表 (按 base_key + source 排序)。"""
    defs = sorted(registry.values(), key=lambda a: (a.base_key, a.source))
    return [a.to_dict() for a in defs]
