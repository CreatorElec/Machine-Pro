"""Pydantic 请求/响应模型。"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    dataset: str
    target: str
    features: List[str]
    algorithm: str
    task: str = "auto"  # auto / classification / regression
    params: Dict[str, Any] = Field(default_factory=dict)
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    random_state: int = 42
    standardize: bool = False           # 是否对特征做标准化 (tricks)
    cv_folds: int = Field(0, ge=0, le=10)  # >1 时额外输出 K 折交叉验证


class AlgorithmSelection(BaseModel):
    """多算法对比中的单个算法选择。"""

    key: str
    params: Dict[str, Any] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    dataset: str
    target: str
    features: List[str]
    algorithms: List[AlgorithmSelection] = Field(default_factory=list)
    task: str = "auto"
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    random_state: int = 42
    standardize: bool = False
    cv_folds: int = Field(0, ge=0, le=10)


class TuneRequest(BaseModel):
    """网格搜索调参请求。"""

    dataset: str
    target: str
    features: List[str]
    algorithm: str
    param_grid: Dict[str, List[Any]] = Field(default_factory=dict)
    task: str = "auto"
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    random_state: int = 42
    standardize: bool = False
    cv_folds: int = Field(3, ge=2, le=10)


class UnsupervisedRequest(BaseModel):
    dataset: str
    features: List[str]
    algorithm: str
    params: Dict[str, Any] = Field(default_factory=dict)
    standardize: bool = True
    random_state: int = 42


class VizRequest(BaseModel):
    """可视化数据请求 (ROC / 学习曲线 / 决策边界 / PCA 散点)。"""

    dataset: str
    features: List[str]
    algorithm: Optional[str] = None   # pca_scatter 不需要算法
    target: Optional[str] = None
    task: str = "auto"
    params: Dict[str, Any] = Field(default_factory=dict)
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    random_state: int = 42
    standardize: bool = False
