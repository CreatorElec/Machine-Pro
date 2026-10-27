# 机器学习实验平台

一个基于 **FastAPI + scikit-learn + 自研 numpy 算法** 的机器学习 Web 实验平台。用户可以在网页上选择经典数据集与算法、配置超参数，一键训练模型、对比多种算法，并直观查看评估结果与可视化图表。

## 实验模式

平台提供四种实验模式：

| 模式 | 说明 |
|------|------|
| 🎯 单模型训练 | 选择单个算法（scikit-learn 或自研实现）训练并评估 |
| ⚔️ 多算法对比 | 在同一数据划分上批量训练多个算法并排名，WebSocket 实时推送进度 |
| 🧩 聚类/降维 | K-Means / 层次聚类 / DBSCAN / PCA |
| 📊 可视化 | ROC 曲线、学习曲线、决策边界、PCA 散点 |

> 说明：多算法对比仅提供 scikit-learn 实现；自研（纯 numpy）算法可在「单模型训练」与「聚类/降维」中使用。

## 功能特性

- **内置经典数据集**：帕尔默企鹅（含 LTER 原始版）、鸢尾花、葡萄酒、乳腺癌、糖尿病
- **支持上传 CSV**：用户上传自己的数据集进行训练
- **sklearn + 自研算法**：同一种算法可切换 scikit-learn / 自研 numpy 实现，便于对比
- **自动任务识别**：根据目标列自动判断分类 / 回归
- **数据预处理**：自动处理缺失值（数值列中位数填充、类别列众数填充 + one-hot）
- **评估指标与图表**：指标卡片、混淆矩阵热力图、分类报告、特征重要性、预测样本对比
- **交叉验证**：可选 K 折交叉验证，输出各指标 mean / std
- **动态参数表单**：每个算法的超参数由后端元数据驱动，前端自动生成
- **WebSocket 实时进度**：多算法对比时实时推送训练进度

## 算法列表

### 分类

| 实现 | 算法 |
|------|------|
| scikit-learn | K 近邻 (KNN)、逻辑回归、决策树、随机森林、SVM、朴素贝叶斯、梯度提升 |
| 自研 (纯 numpy) | K 近邻 (KNN)、逻辑回归、决策树、朴素贝叶斯、线性 SVM |

### 回归

| 实现 | 算法 |
|------|------|
| scikit-learn | 线性回归、岭回归、Lasso、随机森林回归、SVR |
| 自研 (纯 numpy) | 线性回归、岭回归、K 近邻回归、决策树回归 |

### 无监督

| 实现 | 算法 |
|------|------|
| scikit-learn | K-Means、层次聚类、DBSCAN、PCA |
| 自研 (纯 numpy) | K-Means、PCA |

> 自研算法均以纯 numpy 实现，遵循 sklearn 风格接口（`fit` / `predict` / `predict_proba`），并暴露 `classes_`、`coef_` 等属性以便复用平台的评估与特征重要性逻辑。

## 项目结构

```
backend/
├── main.py                  # FastAPI 应用、REST 路由、WebSocket、静态文件挂载
├── algorithms/              # 算法注册表（包）
│   ├── __init__.py          # 聚合导出：分类/回归/无监督算法注册表
│   ├── base.py              # AlgorithmDef / ParamDef 元数据与参数类型转换
│   ├── sklearn_models.py    # scikit-learn 实现（分类/回归）
│   ├── unsupervised_models.py # scikit-learn 无监督实现
│   └── custom/              # 自研纯 numpy 实现
│       ├── linear.py        #   KNN / 逻辑回归 / 线性 SVM / 线性回归 / 岭回归
│       ├── probabilistic.py #   高斯朴素贝叶斯
│       ├── tree.py          #   CART 决策树（分类/回归）
│       ├── unsupervised.py  #   K-Means / PCA
│       └── _common.py       #   标准化器 / softmax 等公共工具
├── datasets.py              # 内置数据集 + 上传数据集管理
├── pipeline.py              # 预处理 + 训练 + 评估 + 交叉验证流水线
├── compare.py               # 多算法对比
├── unsupervised.py          # 无监督任务（聚类评估 / PCA 降维）
├── viz.py                   # 可视化数据生成（ROC / 学习曲线 / 决策边界 / PCA 散点）
├── schemas.py               # Pydantic 请求模型
└── data/                    # 内置 CSV 数据（企鹅数据集）

frontend/
├── index.html
├── style.css
├── js/
│   ├── api.js               # REST / WebSocket 封装
│   ├── app.js               # Vue3 应用逻辑
│   └── charts.js            # ECharts 图表构建
└── vendor/                  # Vue3 + ECharts 本地库文件（无 CDN 依赖）

requirements.txt
README.md
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务
cd backend
uvicorn main:app --reload

# 3. 浏览器访问
#    前端页面:  http://127.0.0.1:8000/
#    API 文档:  http://127.0.0.1:8000/docs
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/datasets` | 列出内置数据集 |
| GET | `/api/datasets/{key}/info` | 数据集详情（预览、缺失值、列类型） |
| GET | `/api/algorithms` | 算法与超参数元数据（含 source: sklearn/custom） |
| POST | `/api/upload` | 上传 CSV 数据集 |
| GET | `/api/upload/{dataset_id}/info` | 上传数据集的摘要 |
| POST | `/api/train` | 训练单模型并返回评估结果 |
| POST | `/api/compare` | 多算法对比（同步版本） |
| POST | `/api/unsupervised` | 无监督任务（聚类评估 / PCA 降维） |
| POST | `/api/viz/roc` | ROC 曲线数据 |
| POST | `/api/viz/learning-curve` | 学习曲线数据 |
| POST | `/api/viz/decision-boundary` | 决策边界数据 |
| POST | `/api/viz/pca-scatter` | PCA 2D 散点数据 |
| WS | `/ws/train` | 多算法对比实时进度推送 |

## 使用示例

以企鹅数据集为例：

1. 选择数据集「帕尔默企鹅」，目标列默认 `species`
2. 选择特征列（喙长、喙深、鳍长、体重等）
3. 选择实验模式：
   - **单模型训练**：选算法（如「随机森林」或「决策树 (自研)」），调参数，点「开始训练」
   - **多算法对比**：勾选多个算法，点「开始对比」，查看排名与实时进度
   - **可视化**：选 ROC 曲线 / 学习曲线 / 决策边界等，生成图表
4. 查看右侧的准确率、混淆矩阵、分类报告、特征重要性等结果

## 技术要点

- **后端**：FastAPI + Pydantic v2，scikit-learn 1.x 训练模型
- **自研算法**：纯 numpy 实现，统一遵循 sklearn 风格接口
- **预处理**：数值列中位数填充、类别列众数填充 + one-hot、目标列 LabelEncoder
- **评估**：分类输出准确率/精确率/召回率/F1/混淆矩阵/分类报告；回归输出 R²/MSE/RMSE/MAE；`train_test_split` 分层抽样 + 可选 K 折交叉验证
- **实时通信**：WebSocket 推送多算法对比的训练进度
- **前端**：Vue3 + ECharts，本地 vendor 引入，无需构建工具
