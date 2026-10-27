"""FastAPI 主应用: 机器学习实验平台后端。"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from algorithms import CLASSIFICATION_ALGORITHMS, REGRESSION_ALGORITHMS
from datasets import (
    DATASETS,
    UPLOADED_DATASETS,
    load_dataframe,
    parse_uploaded_csv,
    summarize_dataframe,
)
from pipeline import train_and_evaluate
from schemas import TrainRequest

app = FastAPI(title="机器学习实验平台", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/datasets")
def list_datasets():
    """列出所有内置数据集。"""
    result = []
    for key, cfg in DATASETS.items():
        df = cfg["load"]()
        result.append({
            "key": key,
            "name": cfg["name"],
            "description": cfg["description"],
            "suggested_target": cfg["suggested_target"],
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
        })
    return result


@app.get("/api/datasets/{key}/info")
def dataset_info(key: str):
    """返回数据集的详细信息 (预览、缺失值、每列任务类型等)。"""
    if key not in DATASETS:
        raise HTTPException(404, f"数据集 '{key}' 不存在")
    cfg = DATASETS[key]
    df = cfg["load"]()
    info = summarize_dataframe(df, cfg)
    info["key"] = key
    return info


@app.get("/api/algorithms")
def list_algorithms():
    """返回分类与回归算法的定义及超参数元数据。"""
    return {
        "classification": [a.to_dict() for a in CLASSIFICATION_ALGORITHMS.values()],
        "regression": [a.to_dict() for a in REGRESSION_ALGORITHMS.values()],
    }


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """上传 CSV 数据集, 返回数据集 id 与摘要。"""
    content = await file.read()
    df = parse_uploaded_csv(content)
    dataset_id = uuid.uuid4().hex[:12]
    UPLOADED_DATASETS[dataset_id] = df
    info = summarize_dataframe(df, {"name": file.filename or "上传数据集", "suggested_target": None})
    info["key"] = f"uploaded:{dataset_id}"
    return {"id": dataset_id, "ref": f"uploaded:{dataset_id}", "info": info}


@app.get("/api/upload/{dataset_id}/info")
def uploaded_dataset_info(dataset_id: str):
    """获取已上传数据集的摘要。"""
    if dataset_id not in UPLOADED_DATASETS:
        raise HTTPException(404, "上传的数据集不存在")
    df = UPLOADED_DATASETS[dataset_id]
    info = summarize_dataframe(df, {"name": "上传数据集", "suggested_target": None})
    info["key"] = f"uploaded:{dataset_id}"
    return info


@app.post("/api/train")
def train(req: TrainRequest):
    """训练模型并返回评估结果。"""
    try:
        df = load_dataframe(req.dataset)
        return train_and_evaluate(df, req)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"训练失败: {exc}") from exc


# 前端静态文件 (必须在所有 API 路由之后挂载)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
