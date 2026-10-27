"""Pydantic 请求/响应模型。"""
from typing import Any, Dict, List

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
