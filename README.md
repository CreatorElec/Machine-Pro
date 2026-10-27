# 机器学习实验平台

一个基于 **FastAPI + scikit-learn** 的机器学习 Web 实验平台。用户可以在网页上选择经典数据集与经典算法、配置超参数，一键训练模型，并直观查看评估结果。

## 功能特性

- **内置经典数据集**：帕尔默企鹅（含 LTER 原始版）、鸢尾花、葡萄酒、乳腺癌、糖尿病
- **支持上传 CSV**：用户上传自己的数据集进行训练
- **经典算法全覆盖**：
  - 分类：KNN、逻辑回归、决策树、随机森林、SVM、朴素贝叶斯、梯度提升
  - 回归：线性回归、岭回归、Lasso、随机森林回归、SVR
- **自动任务识别**：根据目标列自动判断分类 / 回归
- **数据预处理**：自动处理缺失值（中位数/众数填充）、类别特征 one-hot 编码
- **可视化评估结果**：评估指标卡片、混淆矩阵热力图、分类报告、特征重要性、预测样本对比
- **动态参数表单**：每个算法的超参数由后端元数据驱动，前端自动生成

## 项目结构

```
ml-platform/
├── backend/
│   ├── main.py           # FastAPI 应用与路由
│   ├── algorithms.py     # 算法注册表 (含超参数元数据)
│   ├── datasets.py       # 内置数据集 + 上传数据集管理
│   ├── pipeline.py       # 预处理 + 训练 + 评估流水线
│   ├── schemas.py        # Pydantic 模型
│   └── data/             # 内置 CSV 数据 (企鹅数据集)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── requirements.txt
└── README.md
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务 (在 ml-platform 目录下)
cd backend
uvicorn main:app --reload

# 3. 浏览器访问
#    前端页面:  http://127.0.0.1:8000/
#    API 文档:  http://127.0.0.1:8000/docs
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/datasets` | 列出内置数据集 |
| GET | `/api/datasets/{key}/info` | 数据集详情（预览、缺失值、列类型） |
| GET | `/api/algorithms` | 算法与超参数元数据 |
| POST | `/api/upload` | 上传 CSV 数据集 |
| POST | `/api/train` | 训练并返回评估结果 |

## 使用示例

以企鹅数据集为例：

1. 选择数据集「帕尔默企鹅」，目标列默认 `species`
2. 选择特征列（喙长、喙深、鳍长、体重等）
3. 选择算法「随机森林」，调整 `n_estimators=100`
4. 点击「开始训练」
5. 查看右侧的准确率、混淆矩阵、分类报告、特征重要性
6. 查看返回的模型评估图以及各种反馈结果

## 技术要点

- **Pydantic v2** 做请求校验
- **sklearn 1.x** 训练模型，`train_test_split` 分层抽样
- **预处理**：数值列中位数填充、类别列众数填充 + one-hot、目标列 LabelEncoder
- **评估**：分类输出准确率/精确率/召回率/F1/混淆矩阵/分类报告；回归输出 R²/MSE/RMSE/MAE
- **前端零依赖**：纯原生 HTML/CSS/JS，无需构建工具
