/* 机器学习实验平台 - Vue3 应用 */
"use strict";

const { createApp } = Vue;

const METRIC_LABELS = {
  accuracy: "准确率", precision_macro: "精确率(宏)", recall_macro: "召回率(宏)",
  f1_macro: "F1(宏)", f1_weighted: "F1(加权)",
  mse: "MSE", rmse: "RMSE", mae: "MAE", r2: "R²",
  neg_mse: "MSE", neg_mae: "MAE",
};

createApp({
  data() {
    return {
      datasets: [],
      algorithms: { classification: [], regression: [], unsupervised: [] },
      datasetRef: null,
      info: null,
      target: "",
      features: [],
      task: "classification",
      mode: "train",
      modes: [
        { key: "train", name: "单模型训练", icon: "🎯" },
        { key: "compare", name: "多算法对比", icon: "⚔️" },
        { key: "unsupervised", name: "聚类/降维", icon: "🧩" },
        { key: "viz", name: "可视化", icon: "📊" },
      ],
      algorithm: "",
      paramValues: {},
      compareSelection: [],
      unsupAlgorithm: "",
      unsupParamValues: {},
      vizKind: "roc",
      testSizePct: 20,
      randomState: 42,
      standardize: false,
      cvFolds: 0,
      running: false,
      status: "",
      statusError: false,
      trainResult: null,
      compareResult: null,
      unsupResult: null,
      vizResult: null,
      liveProgress: [],
      lossMap: {},
      wsConn: null,
    };
  },

  computed: {
    totalMissing() {
      if (!this.info) return 0;
      return Object.values(this.info.missing).reduce((a, b) => a + b, 0);
    },
    needsTarget() {
      if (this.mode === "unsupervised") return false;
      if (this.mode === "viz" && this.vizKind === "pca-scatter") return false;
      return true;
    },
    featureColumns() {
      if (!this.info) return [];
      return this.info.columns.filter((c) => c !== this.target);
    },
    taskClass() {
      return this.task === "regression" ? "task-badge task-regression" : "task-badge task-classification";
    },
    algorithmList() {
      return this.algorithms[this.task] || [];
    },
    currentAlg() {
      return this.algorithmList.find((a) => a.key === this.algorithm) || null;
    },
    currentUnsupAlg() {
      return this.algorithms.unsupervised.find((a) => a.key === this.unsupAlgorithm) || null;
    },
    compareCandidates() {
      // 多算法对比仅保留 sklearn 实现 (自研算法已从对比入口移除)
      return this.algorithmList.filter((a) => a.source === "sklearn");
    },
    testSize() {
      return this.testSizePct / 100;
    },
    execLabel() {
      return { train: "🚀 开始训练", compare: "⚔️ 开始对比",
               unsupervised: "🧩 开始分析", viz: "📊 生成图表" }[this.mode];
    },
    hasResult() {
      return !!(this.trainResult || this.compareResult ||
                this.unsupResult || this.vizResult);
    },
    hasLoss() {
      return Object.keys(this.lossMap).length > 0;
    },
    progressPct() {
      if (!this.liveProgress.length) return 0;
      const done = this.liveProgress.filter((p) => p.status === "done" || p.status === "error").length;
      return Math.round((done / this.liveProgress.length) * 100);
    },
    cvRows() {
      if (!this.trainResult || !this.trainResult.cv) return {};
      const cv = this.trainResult.cv;
      const rows = {};
      Object.keys(cv).forEach((k) => { if (k !== "folds") rows[k] = cv[k]; });
      return rows;
    },
    extraMetricKeys() {
      if (!this.compareResult || !this.compareResult.ranking.length) return [];
      return Object.keys(this.compareResult.ranking[0].metrics)
        .filter((k) => k !== this.compareResult.metric_key).slice(0, 3);
    },
  },

  watch: {
    currentAlg(alg) {
      this.paramValues = {};
      if (alg) {
        alg.params.forEach((p) => { this.paramValues[p.name] = p.default; });
      }
    },
    currentUnsupAlg(alg) {
      this.unsupParamValues = {};
      if (alg) alg.params.forEach((p) => { this.unsupParamValues[p.name] = p.default; });
    },
    task() {
      this.ensureAlgorithm();
    },
    algorithms: {
      deep: true,
      handler() {
        this.ensureAlgorithm();
      },
    },
    vizKind() {
      this.vizResult = null;
    },
  },

  methods: {
    ensureAlgorithm() {
      const list = this.algorithmList;
      if (list.length && !list.find((a) => a.key === this.algorithm)) {
        this.algorithm = list[0].key;
      }
      if (!this.compareSelection.length) {
        this.compareSelection = list.filter((a) => a.source === "sklearn").map((a) => a.key);
      }
    },

    metricLabel(k) { return METRIC_LABELS[k] || k; },
    fmtDefault(v) { return v === null ? "无限制" : v; },
    setStatus(msg, isError) {
      this.status = msg;
      this.statusError = !!isError;
    },

    /* ---------- 初始化与数据集 ---------- */
    async init() {
      try {
        [this.datasets, this.algorithms] = await Promise.all([
          Api.get("/api/datasets"), Api.get("/api/algorithms"),
        ]);
        if (this.algorithms.unsupervised.length) {
          this.unsupAlgorithm = this.algorithms.unsupervised[0].key;
        }
        if (this.datasets.length) {
          this.datasetRef = this.datasets[0].key;
          await this.selectDataset();
        }
        this.ensureAlgorithm();
      } catch (err) {
        this.setStatus("初始化失败: " + err.message, true);
      }
    },

    async selectDataset() {
      try {
        this.info = await Api.get(`/api/datasets/${this.datasetRef}/info`);
        this.applyDatasetInfo();
      } catch (err) {
        this.setStatus(err.message, true);
      }
    },

    applyDatasetInfo() {
      const info = this.info;
      this.target = info.suggested_target && info.columns.includes(info.suggested_target)
        ? info.suggested_target : info.columns[info.columns.length - 1];
      this.onTargetChange();
    },

    onTargetChange() {
      this.task = (this.info.column_tasks && this.info.column_tasks[this.target]) || "classification";
      const nums = this.info.numeric_columns.filter((c) => c !== this.target);
      this.features = nums.length ? nums.slice() : this.info.columns.filter((c) => c !== this.target);
      this.clearResults();
    },

    selectNumericFeatures() {
      this.features = this.info.numeric_columns.filter((c) => c !== this.target);
    },

    async onUpload(e) {
      const file = e.target.files[0];
      if (!file) return;
      this.setStatus("正在上传解析...");
      try {
        const res = await Api.upload(file);
        this.datasets.push({ key: res.ref, name: "📁 " + res.info.name });
        this.datasetRef = res.ref;
        this.info = res.info;
        this.applyDatasetInfo();
        this.setStatus("上传成功, 请选择目标列与特征");
      } catch (err) {
        this.setStatus(err.message, true);
      } finally {
        e.target.value = "";
      }
    },

    switchMode(m) {
      this.mode = m;
      this.clearResults();
    },

    clearResults() {
      this.trainResult = this.compareResult = null;
      this.unsupResult = this.vizResult = null;
      this.liveProgress = [];
      this.lossMap = {};
      this.setStatus("");
      if (this.wsConn) { this.wsConn.close(); this.wsConn = null; }
      this.running = false;
    },

    basePayload() {
      return {
        dataset: this.datasetRef,
        target: this.target,
        features: this.features,
        task: this.task,
        test_size: this.testSize,
        random_state: this.randomState,
        standardize: this.standardize,
      };
    },

    validateCommon() {
      if (!this.info) { this.setStatus("请先选择数据集", true); return false; }
      if (!this.features.length) { this.setStatus("请至少选择一个特征列", true); return false; }
      if (this.needsTarget && !this.target) { this.setStatus("请选择目标列", true); return false; }
      return true;
    },

    execute() {
      if (!this.validateCommon() || this.running) return;
      this.clearResults();
      this.running = true;
      const fn = {
        train: this.doTrain, compare: this.doCompare,
        unsupervised: this.doUnsupervised, viz: this.doViz,
      }[this.mode];
      fn.call(this);
    },

    /* ---------- 单模型训练 ---------- */
    async doTrain() {
      this.setStatus("训练中...");
      try {
        const payload = { ...this.basePayload(), algorithm: this.algorithm,
                          params: this.paramValues, cv_folds: this.cvFolds };
        this.trainResult = await Api.post("/api/train", payload);
        this.setStatus("训练完成");
        this.$nextTick(() => {
          const r = this.trainResult;
          if (r.confusion_matrix && this.$refs.cmChart) {
            Charts.confusionMatrix(this.$refs.cmChart, r.confusion_matrix, r.classes);
          }
          if (r.feature_importance.length && this.$refs.impChart) {
            Charts.importance(this.$refs.impChart, r.feature_importance);
          }
        });
      } catch (err) {
        this.setStatus(err.message, true);
      } finally {
        this.running = false;
      }
    },

    /* ---------- 多算法对比 (WebSocket) ---------- */
    doCompare() {
      if (!this.compareSelection.length) {
        this.setStatus("请至少选择一个对比算法", true);
        this.running = false;
        return;
      }
      this.setStatus("WebSocket 连接中...");
      const payload = {
        ...this.basePayload(),
        cv_folds: this.cvFolds,
        algorithms: this.compareSelection.map((key) => ({ key, params: {} })),
      };
      const metricKey = this.task === "classification" ? "accuracy" : "r2";
      const findItem = (key) => this.liveProgress.find((p) => p.algorithm === key);

      this.wsConn = Api.compareWS(payload, {
        onStart: (ev) => {
          this.setStatus(`开始训练 ${ev.total} 个算法...`);
        },
        onProgress: (ev) => {
          let item = findItem(ev.algorithm);
          if (!item) {
            this.liveProgress.push({ algorithm: ev.algorithm, algorithm_name: ev.algorithm_name,
                                     source: ev.source, status: "running" });
          } else {
            item.status = "running";
          }
        },
        onLoss: (ev) => {
          if (!this.lossMap[ev.algorithm]) this.lossMap[ev.algorithm] = [];
          this.lossMap[ev.algorithm].push({ iter: ev.iter, loss: ev.loss });
          this.$nextTick(() => {
            if (this.$refs.lossChart) Charts.lossCurve(this.$refs.lossChart, this.lossMap);
          });
        },
        onResult: (ev) => {
          const item = findItem(ev.algorithm);
          if (!item) return;
          if (ev.error) {
            item.status = "error";
            item.error = ev.error;
          } else {
            item.status = "done";
            item.score = ev.result.metrics[metricKey];
          }
        },
        onDone: (ev) => {
          this.compareResult = ev.comparison;
          this.setStatus("对比完成");
          this.running = false;
          this.$nextTick(() => {
            if (this.$refs.rankChart) {
              Charts.rankingBar(this.$refs.rankChart, ev.comparison.ranking, ev.comparison.metric_key);
            }
          });
        },
        onError: (ev) => {
          this.setStatus("对比失败: " + ev.message, true);
          this.running = false;
        },
      });
    },

    /* ---------- 无监督 ---------- */
    async doUnsupervised() {
      this.setStatus("分析中...");
      try {
        const payload = {
          dataset: this.datasetRef, features: this.features,
          algorithm: this.unsupAlgorithm, params: this.unsupParamValues,
          standardize: this.standardize, random_state: this.randomState,
        };
        this.unsupResult = await Api.post("/api/unsupervised", payload);
        this.setStatus("分析完成");
        this.$nextTick(() => {
          const r = this.unsupResult;
          if (r.scatter && this.$refs.unsupScatter) {
            Charts.scatter2d(this.$refs.unsupScatter, r,
              { title: r.kind === "cluster" ? "聚类结果 (2D)" : "PCA 投影", cluster: r.kind === "cluster" });
          }
          if (r.elbow && this.$refs.elbowChart) Charts.elbow(this.$refs.elbowChart, r.elbow);
          if (r.kind === "pca" && this.$refs.varChart) {
            Charts.variance(this.$refs.varChart, r.explained_variance_ratio, r.cumulative_variance);
          }
        });
      } catch (err) {
        this.setStatus(err.message, true);
      } finally {
        this.running = false;
      }
    },

    /* ---------- 可视化 ---------- */
    async doViz() {
      this.setStatus("生成图表中...");
      try {
        const payload = {
          dataset: this.datasetRef, features: this.features,
          target: this.target || null, task: this.task,
          params: this.paramValues, test_size: this.testSize,
          random_state: this.randomState, standardize: this.standardize,
        };
        if (this.vizKind !== "pca-scatter") payload.algorithm = this.algorithm;
        this.vizResult = await Api.post(`/api/viz/${this.vizKind}`, payload);
        this.setStatus("已生成");
        this.$nextTick(() => {
          const el = this.$refs.vizChart;
          if (!el) return;
          const d = this.vizResult;
          if (d.kind === "roc") Charts.roc(el, d.curves);
          else if (d.kind === "learning_curve") Charts.learningCurve(el, d);
          else if (d.kind === "decision_boundary") Charts.decisionBoundary(el, d);
          else if (d.kind === "pca_scatter") Charts.scatter2d(el, d, { title: "PCA 2D 散点" });
        });
      } catch (err) {
        this.setStatus(err.message, true);
      } finally {
        this.running = false;
      }
    },
  },

  mounted() {
    this.init();
    window.addEventListener("resize", () => {
      document.querySelectorAll(".chart").forEach((el) => {
        const inst = echarts.getInstanceByDom(el);
        if (inst) inst.resize();
      });
    });
  },
}).mount("#app");
