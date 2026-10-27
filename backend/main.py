"""FastAPI 主应用: 机器学习实验平台后端。

路由顺序: API 路由 -> WebSocket 路由 -> 前端静态文件挂载 (mount 必须在最后)。
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from algorithms import algorithms_meta
from compare import run_comparison
from datasets import (
    DATASETS,
    UPLOADED_DATASETS,
    load_dataframe,
    parse_uploaded_csv,
    summarize_dataframe,
)
from pipeline import train_and_evaluate
from schemas import CompareRequest, TrainRequest, TuneRequest, UnsupervisedRequest, VizRequest
from tuning import grid_search
from unsupervised import run_unsupervised
from viz import (
    decision_boundary_data,
    learning_curve_data,
    pca_scatter_data,
    roc_curve_data,
)

app = FastAPI(title="机器学习实验平台", version="2.0.0")

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
    """返回分类/回归/无监督算法的定义及超参数元数据 (含 source: sklearn/custom)。"""
    return algorithms_meta()


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


def _guard(fn, *args):
    """统一异常 -> HTTP 错误码。"""
    try:
        return fn(*args)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"执行失败: {exc}") from exc


@app.post("/api/train")
def train(req: TrainRequest):
    """训练单个模型并返回评估结果。"""
    return _guard(lambda: train_and_evaluate(load_dataframe(req.dataset), req))


@app.post("/api/compare")
def compare(req: CompareRequest):
    """多算法对比 (同步版本; 实时进度请用 WebSocket /ws/train)。"""
    return _guard(lambda: run_comparison(load_dataframe(req.dataset), req))


@app.post("/api/tune")
def tune(req: TuneRequest):
    """网格搜索调参。"""
    return _guard(lambda: grid_search(load_dataframe(req.dataset), req))


@app.post("/api/unsupervised")
def unsupervised(req: UnsupervisedRequest):
    """无监督任务: 聚类评估 / PCA 降维。"""
    return _guard(lambda: run_unsupervised(load_dataframe(req.dataset), req))


@app.post("/api/viz/roc")
def viz_roc(req: VizRequest):
    """ROC 曲线数据 (分类)。"""
    return _guard(lambda: roc_curve_data(load_dataframe(req.dataset), req))


@app.post("/api/viz/learning-curve")
def viz_learning_curve(req: VizRequest):
    """学习曲线数据。"""
    return _guard(lambda: learning_curve_data(load_dataframe(req.dataset), req))


@app.post("/api/viz/decision-boundary")
def viz_decision_boundary(req: VizRequest):
    """决策边界数据 (分类, 2 特征或 PCA 降维)。"""
    return _guard(lambda: decision_boundary_data(load_dataframe(req.dataset), req))


@app.post("/api/viz/pca-scatter")
def viz_pca_scatter(req: VizRequest):
    """PCA 2D 散点数据 (探索性分析, target 可选)。"""
    return _guard(lambda: pca_scatter_data(load_dataframe(req.dataset), req))


# --------------------------------------------------------------------------- #
# WebSocket: 训练进度实时推送 (多算法对比)
# --------------------------------------------------------------------------- #
@app.websocket("/ws/train")
async def ws_train(ws: WebSocket):
    """训练进度推送。客户端发送 CompareRequest JSON, 服务端按事件流回推:

    {"type": "start", "total"} ->
    {"type": "progress", "index", "algorithm", ...} ->
    {"type": "loss", "algorithm", "iter", "loss"} (自研 GD 模型) ->
    {"type": "result", "algorithm", "result"} ->
    {"type": "done", "comparison"} / {"type": "error", "message"}
    """
    await ws.accept()
    try:
        payload = await ws.receive_json()
    except WebSocketDisconnect:
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(event: Dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def work() -> None:
        try:
            req = CompareRequest(**payload)
            df = load_dataframe(req.dataset)
            emit({"type": "start", "total": len(req.algorithms)})
            comparison = run_comparison(df, req, progress_cb=emit)
            emit({"type": "done", "comparison": comparison})
        except Exception as exc:  # noqa: BLE001
            emit({"type": "error", "message": str(exc)})

    worker = asyncio.create_task(run_in_threadpool(work))
    try:
        while True:
            event = await queue.get()
            await ws.send_json(event)
            if event.get("type") in ("done", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        await worker


# 前端静态文件 (必须在所有 API/WS 路由之后挂载)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
