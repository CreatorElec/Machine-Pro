"""自研算法注册表。

与 sklearn 实现按 base_key 配对 (如 knn / knn_custom), 前端可切换来源并做性能对比。
"""
from __future__ import annotations

from typing import Dict

from ..base import AlgorithmDef, ParamDef
from .linear import (
    MyKNNClassifier,
    MyKNNRegressor,
    MyLinearRegression,
    MyLinearSVM,
    MyLogisticRegression,
    MyRidge,
)
from .probabilistic import MyGaussianNB
from .tree import MyDecisionTreeClassifier, MyDecisionTreeRegressor
from .unsupervised import MyKMeans, MyPCA

CUSTOM_CLASSIFICATION: Dict[str, AlgorithmDef] = {
    "knn_custom": AlgorithmDef(
        key="knn_custom",
        name="K 近邻 (KNN·自研)",
        estimator=MyKNNClassifier,
        description="纯 numpy 实现: 最近 K 个邻居多数投票 / 距离加权",
        source="custom",
        task="classification",
        base_key="knn",
        params=[
            ParamDef("n_neighbors", "邻居数 K", "int", 5, min_value=1, max_value=50),
            ParamDef("weights", "权重", "select", "uniform", options=["uniform", "distance"]),
            ParamDef("metric", "距离度量", "select", "minkowski",
                     options=["minkowski", "euclidean", "manhattan"]),
        ],
    ),
    "logistic_regression_custom": AlgorithmDef(
        key="logistic_regression_custom",
        name="逻辑回归 (Logistic·自研)",
        estimator=MyLogisticRegression,
        description="纯 numpy 实现: 多项 softmax + 全批量梯度下降 + L2 正则",
        source="custom",
        task="classification",
        base_key="logistic_regression",
        params=[
            ParamDef("C", "正则化强度 C (倒数)", "float", 1.0, min_value=0.01, max_value=100.0, step=0.01),
            ParamDef("max_iter", "最大迭代次数", "int", 500, min_value=50, max_value=3000),
            ParamDef("learning_rate", "学习率", "float", 1.0, min_value=0.01, max_value=2.0, step=0.05),
        ],
    ),
    "decision_tree_custom": AlgorithmDef(
        key="decision_tree_custom",
        name="决策树 (Tree·自研)",
        estimator=MyDecisionTreeClassifier,
        description="纯 numpy 实现: CART, 支持 gini / entropy 分裂准则",
        source="custom",
        task="classification",
        base_key="decision_tree",
        params=[
            ParamDef("max_depth", "最大深度", "select", "无限制",
                     options=["无限制", "3", "5", "10", "20", "50"]),
            ParamDef("criterion", "分裂准则", "select", "gini", options=["gini", "entropy"]),
            ParamDef("min_samples_split", "节点最小分裂样本数", "int", 2, min_value=2, max_value=20),
            ParamDef("min_samples_leaf", "叶节点最小样本数", "int", 1, min_value=1, max_value=10),
        ],
    ),
    "naive_bayes_custom": AlgorithmDef(
        key="naive_bayes_custom",
        name="朴素贝叶斯 (NB·自研)",
        estimator=MyGaussianNB,
        description="纯 numpy 实现: 高斯朴素贝叶斯, 独立正态假设",
        source="custom",
        task="classification",
        base_key="naive_bayes",
        params=[
            ParamDef("var_smoothing", "方差平滑系数", "float", 1e-9,
                     min_value=1e-12, max_value=1e-6, step=1e-9),
        ],
    ),
    "svm_custom": AlgorithmDef(
        key="svm_custom",
        name="线性 SVM (自研)",
        estimator=MyLinearSVM,
        description="纯 numpy 实现: 合页损失 + 次梯度下降, 一对多策略",
        source="custom",
        task="classification",
        base_key="svm",
        params=[
            ParamDef("C", "惩罚系数 C", "float", 1.0, min_value=0.01, max_value=100.0, step=0.01),
            ParamDef("max_iter", "最大迭代次数", "int", 1000, min_value=100, max_value=3000),
            ParamDef("learning_rate", "学习率", "float", 0.001, min_value=0.0001, max_value=0.05, step=0.0005),
        ],
    ),
}

CUSTOM_REGRESSION: Dict[str, AlgorithmDef] = {
    "linear_regression_custom": AlgorithmDef(
        key="linear_regression_custom",
        name="线性回归 (自研)",
        estimator=MyLinearRegression,
        description="纯 numpy 实现: 最小二乘闭式解 (lstsq)",
        source="custom",
        task="regression",
        base_key="linear_regression",
        params=[],
    ),
    "ridge_custom": AlgorithmDef(
        key="ridge_custom",
        name="岭回归 (Ridge·自研)",
        estimator=MyRidge,
        description="纯 numpy 实现: L2 正则闭式解",
        source="custom",
        task="regression",
        base_key="ridge",
        params=[
            ParamDef("alpha", "正则化强度 α", "float", 1.0, min_value=0.001, max_value=100.0, step=0.01),
        ],
    ),
    "knn_regressor_custom": AlgorithmDef(
        key="knn_regressor_custom",
        name="K 近邻回归 (KNN·自研)",
        estimator=MyKNNRegressor,
        description="纯 numpy 实现: 近邻目标值均值 / 距离加权均值",
        source="custom",
        task="regression",
        base_key="knn_regressor",
        params=[
            ParamDef("n_neighbors", "邻居数 K", "int", 5, min_value=1, max_value=50),
            ParamDef("weights", "权重", "select", "uniform", options=["uniform", "distance"]),
            ParamDef("metric", "距离度量", "select", "minkowski",
                     options=["minkowski", "euclidean", "manhattan"]),
        ],
    ),
    "decision_tree_regressor_custom": AlgorithmDef(
        key="decision_tree_regressor_custom",
        name="决策树回归 (Tree·自研)",
        estimator=MyDecisionTreeRegressor,
        description="纯 numpy 实现: CART 回归树, 方差下降分裂",
        source="custom",
        task="regression",
        base_key="decision_tree_regressor",
        params=[
            ParamDef("max_depth", "最大深度", "select", "无限制",
                     options=["无限制", "3", "5", "10", "20", "50"]),
            ParamDef("min_samples_split", "节点最小分裂样本数", "int", 2, min_value=2, max_value=20),
            ParamDef("min_samples_leaf", "叶节点最小样本数", "int", 1, min_value=1, max_value=10),
        ],
    ),
}

CUSTOM_UNSUPERVISED: Dict[str, AlgorithmDef] = {
    "kmeans_custom": AlgorithmDef(
        key="kmeans_custom",
        name="K-Means 聚类 (自研)",
        estimator=MyKMeans,
        description="纯 numpy 实现: k-means++ 初始化 + Lloyd 迭代",
        source="custom",
        task="unsupervised",
        base_key="kmeans",
        params=[
            ParamDef("n_clusters", "簇数量 K", "int", 3, min_value=2, max_value=10),
            ParamDef("max_iter", "最大迭代次数", "int", 300, min_value=10, max_value=1000),
            ParamDef("n_init", "随机初始化次数", "int", 10, min_value=1, max_value=20),
            ParamDef("random_state", "随机种子", "int", 42, min_value=0, max_value=99999),
        ],
    ),
    "pca_custom": AlgorithmDef(
        key="pca_custom",
        name="PCA 降维 (自研)",
        estimator=MyPCA,
        description="纯 numpy 实现: 中心化 + SVD 分解求主成分",
        source="custom",
        task="unsupervised",
        base_key="pca",
        params=[
            ParamDef("n_components", "主成分数", "int", 2, min_value=1, max_value=10),
        ],
    ),
}
