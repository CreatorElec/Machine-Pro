"""数据集管理: 内置经典数据集加载 + 用户上传数据集。"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine

DATA_DIR = Path(__file__).resolve().parent / "data"

# 上传的数据集保存在内存中 (演示规模足够)
UPLOADED_DATASETS: Dict[str, pd.DataFrame] = {}


def _load_csv(filename: str) -> Callable[[], pd.DataFrame]:
    def _inner() -> pd.DataFrame:
        return pd.read_csv(DATA_DIR / filename)
    return _inner


def _load_sklearn(loader, target_name: str) -> Callable[[], pd.DataFrame]:
    def _inner() -> pd.DataFrame:
        bunch = loader(as_frame=True)
        df = bunch.frame.copy()
        df.rename(columns={"target": target_name}, inplace=True)
        # 将数字标签映射为真实类别名 (如 iris 的 0/1/2 -> setosa/versicolor/virginica)
        names = getattr(bunch, "target_names", None)
        if names is not None:
            mapping = {i: name for i, name in enumerate(names)}
            df[target_name] = df[target_name].map(mapping)
        return df
    return _inner


# 内置数据集注册表
DATASETS: Dict[str, Dict[str, Any]] = {
    "penguins": {
        "load": _load_csv("penguins.csv"),
        "name": "帕尔默企鹅 (Penguins)",
        "description": "南极帕尔默群岛 3 种企鹅的形态测量数据 (344 条), 经典分类数据集",
        "suggested_target": "species",
    },
    "penguins_lter": {
        "load": _load_csv("penguins_lter.csv"),
        "name": "企鹅 LTER 原始数据 (Penguins LTER)",
        "description": "帕尔默站长期生态研究(LTER)原始记录, 含更多测量与元数据列",
        "suggested_target": "Species",
    },
    "iris": {
        "load": _load_sklearn(load_iris, "species"),
        "name": "鸢尾花 (Iris)",
        "description": "3 类鸢尾花的花萼/花瓣尺寸 (150 条), 最经典的分类数据集",
        "suggested_target": "species",
    },
    "wine": {
        "load": _load_sklearn(load_wine, "class"),
        "name": "葡萄酒 (Wine)",
        "description": "3 种葡萄酒的 13 项化学成分指标 (178 条)",
        "suggested_target": "class",
    },
    "breast_cancer": {
        "load": _load_sklearn(load_breast_cancer, "diagnosis"),
        "name": "乳腺癌 (Breast Cancer)",
        "description": "乳腺肿瘤的 30 项细胞特征, 二分类 (569 条)",
        "suggested_target": "diagnosis",
    },
    "diabetes": {
        "load": _load_sklearn(load_diabetes, "target"),
        "name": "糖尿病 (Diabetes)",
        "description": "糖尿病患者的 10 项指标与病情进展指标 (442 条), 经典回归数据集",
        "suggested_target": "target",
    },
}


def detect_task(y: pd.Series) -> str:
    """自动判断目标列属于分类还是回归任务。"""
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    if y.nunique() <= 20:
        return "classification"
    return "regression"


def summarize_dataframe(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """生成数据集摘要信息, 供前端展示与配置。"""
    columns = df.columns.tolist()
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in columns if c not in numeric_cols]

    # 每列的任务类型 (目标切换时前端据此切换算法列表)
    column_tasks: Dict[str, str] = {}
    for c in columns:
        if c in numeric_cols and df[c].nunique() > 20:
            column_tasks[c] = "regression"
        else:
            column_tasks[c] = "classification"

    missing = {c: int(df[c].isna().sum()) for c in columns}
    dtypes = {c: str(df[c].dtype) for c in columns}

    preview: list = []
    for _, row in df.head(5).iterrows():
        rec: Dict[str, Any] = {}
        for c in columns:
            v = row[c]
            if pd.isna(v):
                rec[c] = None
            elif isinstance(v, (np.integer,)):
                rec[c] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[c] = float(v)
            else:
                rec[c] = str(v)
        preview.append(rec)

    suggested = cfg.get("suggested_target")
    return {
        "name": cfg.get("name", "未命名数据集"),
        "description": cfg.get("description", ""),
        "columns": columns,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "column_tasks": column_tasks,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "n_rows": int(len(df)),
        "preview": preview,
        "missing": missing,
        "dtypes": dtypes,
        "suggested_target": suggested,
        "suggested_task": column_tasks.get(suggested, "classification") if suggested else "classification",
    }


def load_dataframe(ref: str) -> pd.DataFrame:
    """根据引用加载 DataFrame: 内置数据集名 或 'uploaded:<id>'。"""
    from fastapi import HTTPException

    if ref.startswith("uploaded:"):
        dataset_id = ref.split(":", 1)[1]
        if dataset_id not in UPLOADED_DATASETS:
            raise HTTPException(404, "上传的数据集不存在或已过期")
        return UPLOADED_DATASETS[dataset_id]

    if ref not in DATASETS:
        raise HTTPException(404, f"数据集 '{ref}' 不存在")
    return DATASETS[ref]["load"]()


def parse_uploaded_csv(content: bytes) -> pd.DataFrame:
    """解析用户上传的 CSV 内容。"""
    from fastapi import HTTPException

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"CSV 解析失败: {exc}") from exc
    if df.empty or df.shape[1] < 2:
        raise HTTPException(400, "文件为空或列数不足 (至少需要 2 列)")
    return df
