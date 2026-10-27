"""无监督算法注册表: 聚类 (K-Means / 层次 / DBSCAN) 与降维 (PCA)。

无监督算法没有目标列, 因此与监督算法分开注册。
"""
from __future__ import annotations

from typing import Dict

from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA

from .base import AlgorithmDef, ParamDef
from .custom import CUSTOM_UNSUPERVISED

SKLEARN_UNSUPERVISED: Dict[str, AlgorithmDef] = {
    "kmeans": AlgorithmDef(
        key="kmeans",
        name="K-Means 聚类",
        estimator=KMeans,
        description="迭代最小化簇内平方和, 最常用的划分聚类",
        source="sklearn",
        task="unsupervised",
        base_key="kmeans",
        params=[
            ParamDef("n_clusters", "簇数量 K", "int", 3, min_value=2, max_value=10),
            ParamDef("init", "初始化方式", "select", "k-means++", options=["k-means++", "random"]),
            ParamDef("n_init", "随机初始化次数", "int", 10, min_value=1, max_value=20),
            ParamDef("random_state", "随机种子", "int", 42, min_value=0, max_value=99999),
        ],
    ),
    "agglomerative": AlgorithmDef(
        key="agglomerative",
        name="层次聚类 (Agglomerative)",
        estimator=AgglomerativeClustering,
        description="自底向上合并, 基于距离的层次聚类",
        source="sklearn",
        task="unsupervised",
        base_key="agglomerative",
        params=[
            ParamDef("n_clusters", "簇数量 K", "int", 3, min_value=2, max_value=10),
            ParamDef("linkage", "连接准则", "select", "ward", options=["ward", "complete", "average"]),
        ],
    ),
    "dbscan": AlgorithmDef(
        key="dbscan",
        name="DBSCAN 密度聚类",
        estimator=DBSCAN,
        description="基于密度, 可发现任意形状簇并识别噪声点",
        source="sklearn",
        task="unsupervised",
        base_key="dbscan",
        params=[
            ParamDef("eps", "邻域半径 ε", "float", 0.5, min_value=0.01, max_value=10.0, step=0.01),
            ParamDef("min_samples", "核心点最小样本数", "int", 5, min_value=2, max_value=50),
        ],
    ),
    "pca": AlgorithmDef(
        key="pca",
        name="PCA 降维",
        estimator=PCA,
        description="主成分分析, 线性降维与方差解释",
        source="sklearn",
        task="unsupervised",
        base_key="pca",
        params=[
            ParamDef("n_components", "主成分数", "int", 2, min_value=1, max_value=10),
        ],
    ),
}

UNSUPERVISED_ALGORITHMS: Dict[str, AlgorithmDef] = {
    **SKLEARN_UNSUPERVISED,
    **CUSTOM_UNSUPERVISED,
}
