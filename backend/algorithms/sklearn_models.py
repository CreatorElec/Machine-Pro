"""sklearn 实现算法注册表 (分类 / 回归)。"""
from __future__ import annotations

from typing import Dict

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

from .base import AlgorithmDef, ParamDef


CLASSIFICATION_ALGORITHMS: Dict[str, AlgorithmDef] = {
    "knn": AlgorithmDef(
        key="knn",
        name="K 近邻 (KNN)",
        estimator=KNeighborsClassifier,
        description="根据最近 K 个邻居的多数投票进行分类",
        source="sklearn",
        task="classification",
        params=[
            ParamDef("n_neighbors", "邻居数 K", "int", 5, min_value=1, max_value=50),
            ParamDef("weights", "权重", "select", "uniform", options=["uniform", "distance"]),
            ParamDef("metric", "距离度量", "select", "minkowski",
                     options=["minkowski", "euclidean", "manhattan"]),
        ],
    ),
    "logistic_regression": AlgorithmDef(
        key="logistic_regression",
        name="逻辑回归 (Logistic Regression)",
        estimator=LogisticRegression,
        description="线性分类模型, 输出类别概率",
        source="sklearn",
        task="classification",
        params=[
            ParamDef("C", "正则化强度 C (倒数)", "float", 1.0, min_value=0.01, max_value=100.0, step=0.01),
            ParamDef("max_iter", "最大迭代次数", "int", 200, min_value=50, max_value=5000),
            ParamDef("solver", "优化算法", "select", "lbfgs",
                     options=["lbfgs", "liblinear", "newton-cg", "saga"]),
        ],
    ),
    "decision_tree": AlgorithmDef(
        key="decision_tree",
        name="决策树 (Decision Tree)",
        estimator=DecisionTreeClassifier,
        description="基于特征分裂的树形模型, 可解释性强",
        source="sklearn",
        task="classification",
        params=[
            ParamDef("max_depth", "最大深度", "select", "无限制",
                     options=["无限制", "3", "5", "10", "20", "50"]),
            ParamDef("criterion", "分裂准则", "select", "gini", options=["gini", "entropy"]),
            ParamDef("min_samples_split", "节点最小分裂样本数", "int", 2, min_value=2, max_value=20),
        ],
    ),
    "random_forest": AlgorithmDef(
        key="random_forest",
        name="随机森林 (Random Forest)",
        estimator=RandomForestClassifier,
        description="集成多棵决策树, 抗过拟合能力强",
        source="sklearn",
        task="classification",
        params=[
            ParamDef("n_estimators", "决策树数量", "int", 100, min_value=10, max_value=500),
            ParamDef("max_depth", "单棵树最大深度", "select", "无限制",
                     options=["无限制", "3", "5", "10", "20", "50"]),
            ParamDef("min_samples_split", "节点最小分裂样本数", "int", 2, min_value=2, max_value=20),
        ],
    ),
    "svm": AlgorithmDef(
        key="svm",
        name="支持向量机 (SVM)",
        estimator=SVC,
        description="寻找最大间隔超平面, 适合小样本高维数据",
        source="sklearn",
        task="classification",
        params=[
            ParamDef("C", "惩罚系数 C", "float", 1.0, min_value=0.01, max_value=100.0, step=0.01),
            ParamDef("kernel", "核函数", "select", "rbf",
                     options=["rbf", "linear", "poly", "sigmoid"]),
            ParamDef("gamma", "核系数 gamma", "select", "scale", options=["scale", "auto"]),
        ],
    ),
    "naive_bayes": AlgorithmDef(
        key="naive_bayes",
        name="朴素贝叶斯 (Naive Bayes)",
        estimator=GaussianNB,
        description="基于贝叶斯定理的生成式分类器, 训练速度快",
        source="sklearn",
        task="classification",
        params=[
            ParamDef("var_smoothing", "方差平滑系数", "float", 1e-9,
                     min_value=1e-12, max_value=1e-6, step=1e-9),
        ],
    ),
    "gradient_boosting": AlgorithmDef(
        key="gradient_boosting",
        name="梯度提升 (Gradient Boosting)",
        estimator=GradientBoostingClassifier,
        description="串行集成弱学习器, 通常精度较高",
        source="sklearn",
        task="classification",
        params=[
            ParamDef("n_estimators", "弱学习器数量", "int", 100, min_value=10, max_value=500),
            ParamDef("learning_rate", "学习率", "float", 0.1, min_value=0.001, max_value=1.0, step=0.01),
            ParamDef("max_depth", "单棵树最大深度", "int", 3, min_value=1, max_value=20),
        ],
    ),
}

REGRESSION_ALGORITHMS: Dict[str, AlgorithmDef] = {
    "linear_regression": AlgorithmDef(
        key="linear_regression",
        name="线性回归 (Linear Regression)",
        estimator=LinearRegression,
        description="最小二乘线性拟合, 最基础的回归模型",
        source="sklearn",
        task="regression",
        params=[],
    ),
    "ridge": AlgorithmDef(
        key="ridge",
        name="岭回归 (Ridge)",
        estimator=Ridge,
        description="带 L2 正则的线性回归, 防止过拟合",
        source="sklearn",
        task="regression",
        params=[
            ParamDef("alpha", "正则化强度 α", "float", 1.0, min_value=0.001, max_value=100.0, step=0.01),
        ],
    ),
    "lasso": AlgorithmDef(
        key="lasso",
        name="Lasso 回归",
        estimator=Lasso,
        description="带 L1 正则的线性回归, 可做特征选择",
        source="sklearn",
        task="regression",
        params=[
            ParamDef("alpha", "正则化强度 α", "float", 1.0, min_value=0.001, max_value=100.0, step=0.01),
        ],
    ),
    "random_forest_regressor": AlgorithmDef(
        key="random_forest_regressor",
        name="随机森林回归 (Random Forest)",
        estimator=RandomForestRegressor,
        description="集成多棵回归树的非线性模型",
        source="sklearn",
        task="regression",
        params=[
            ParamDef("n_estimators", "决策树数量", "int", 100, min_value=10, max_value=500),
            ParamDef("max_depth", "单棵树最大深度", "select", "无限制",
                     options=["无限制", "3", "5", "10", "20", "50"]),
        ],
    ),
    "svr": AlgorithmDef(
        key="svr",
        name="支持向量回归 (SVR)",
        estimator=SVR,
        description="基于 SVM 的回归模型",
        source="sklearn",
        task="regression",
        params=[
            ParamDef("C", "惩罚系数 C", "float", 1.0, min_value=0.01, max_value=100.0, step=0.01),
            ParamDef("kernel", "核函数", "select", "rbf", options=["rbf", "linear", "poly"]),
            ParamDef("gamma", "核系数 gamma", "select", "scale", options=["scale", "auto"]),
            ParamDef("epsilon", "不敏感区间 ε", "float", 0.1, min_value=0.001, max_value=1.0, step=0.01),
        ],
    ),
}
