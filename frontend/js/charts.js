/* ECharts 图表构建: 所有函数接收 DOM 元素与后端数据, 返回 echarts 实例 */
"use strict";

const Charts = (() => {
  const PALETTE = ["#4f7cff", "#34c38f", "#f7b84b", "#f46a6a", "#9b7ede",
                   "#50a5f1", "#e83e8c", "#6f42c1", "#20c997", "#fd7e14"];

  function make(el) {
    const inst = echarts.getInstanceByDom(el) || echarts.init(el);
    return inst;
  }

  const baseTooltip = { trigger: "item", confine: true };

  /* 指标对比柱状图: metrics = {acc:.., ..} 或多算法 [{name, value}] */
  function metricBar(el, items, title) {
    make(el).setOption({
      title: { text: title || "指标", left: "center", textStyle: { fontSize: 13 } },
      tooltip: baseTooltip,
      grid: { left: 60, right: 20, top: 40, bottom: 30 },
      xAxis: { type: "category", data: items.map((i) => i.name), axisLabel: { rotate: 20, fontSize: 10 } },
      yAxis: { type: "value" },
      series: [{ type: "bar", data: items.map((i) => i.value), itemStyle: { color: "#4f7cff" }, barMaxWidth: 40 }],
    }, true);
  }

  /* 多算法排名条形图 */
  function rankingBar(el, ranking, metricKey) {
    const items = ranking.slice().reverse();
    make(el).setOption({
      title: { text: `算法排名 (${metricKey})`, left: "center", textStyle: { fontSize: 13 } },
      tooltip: { ...baseTooltip, formatter: (p) => `${p.name}<br/>${metricKey}: ${p.value}` },
      grid: { left: 150, right: 40, top: 40, bottom: 20 },
      xAxis: { type: "value" },
      yAxis: { type: "category", data: items.map((r) => r.algorithm_name), axisLabel: { fontSize: 11 } },
      series: [{
        type: "bar",
        data: items.map((r, i) => ({
          value: r.score,
          itemStyle: { color: r.source === "custom" ? "#9b7ede" : "#4f7cff" },
        })),
        barMaxWidth: 18,
        label: { show: true, position: "right", fontSize: 10 },
      }],
    }, true);
  }

  /* 混淆矩阵热力图 */
  function confusionMatrix(el, matrix, classes) {
    const data = [];
    matrix.forEach((row, i) => row.forEach((v, j) => data.push([j, i, v])));
    make(el).setOption({
      title: { text: "混淆矩阵", left: "center", textStyle: { fontSize: 13 } },
      tooltip: { ...baseTooltip, formatter: (p) => `真实 ${classes[p.value[1]]} → 预测 ${classes[p.value[0]]}: ${p.value[2]}` },
      grid: { left: 80, right: 60, top: 40, bottom: 50 },
      xAxis: { type: "category", name: "预测", data: classes, axisLabel: { fontSize: 10, rotate: 20 } },
      yAxis: { type: "category", name: "真实", data: classes, axisLabel: { fontSize: 10 } },
      visualMap: { min: 0, max: Math.max(...data.map((d) => d[2]), 1), orient: "vertical", right: 0, top: "center", inRange: { color: ["#eef3ff", "#4f7cff"] }, calculable: false, textStyle: { fontSize: 10 } },
      series: [{ type: "heatmap", data, label: { show: true, fontSize: 10 } }],
    }, true);
  }

  /* 特征重要性横向条形图 */
  function importance(el, items) {
    const list = items.slice(0, 12).reverse();
    make(el).setOption({
      title: { text: "特征重要性 Top", left: "center", textStyle: { fontSize: 13 } },
      tooltip: baseTooltip,
      grid: { left: 130, right: 40, top: 40, bottom: 20 },
      xAxis: { type: "value" },
      yAxis: { type: "category", data: list.map((i) => i.feature), axisLabel: { fontSize: 10 } },
      series: [{ type: "bar", data: list.map((i) => i.importance), itemStyle: { color: "#34c38f" }, barMaxWidth: 14 }],
    }, true);
  }

  /* ROC 曲线 (支持多条) */
  function roc(el, curves) {
    make(el).setOption({
      title: { text: "ROC 曲线", left: "center", textStyle: { fontSize: 13 } },
      tooltip: { trigger: "axis", confine: true },
      legend: { bottom: 0, textStyle: { fontSize: 10 }, type: "scroll" },
      grid: { left: 50, right: 20, top: 40, bottom: 50 },
      xAxis: { type: "value", name: "FPR", max: 1 },
      yAxis: { type: "value", name: "TPR", max: 1 },
      series: [
        ...curves.map((c, i) => ({
          name: `${c.label} AUC=${c.auc}`,
          type: "line",
          showSymbol: false,
          data: c.fpr.map((f, j) => [f, c.tpr[j]]),
          lineStyle: { width: 2, color: PALETTE[i % PALETTE.length] },
          itemStyle: { color: PALETTE[i % PALETTE.length] },
        })),
        { name: "随机猜测", type: "line", showSymbol: false, data: [[0, 0], [1, 1]],
          lineStyle: { type: "dashed", color: "#aaa", width: 1 }, itemStyle: { color: "#aaa" } },
      ],
    }, true);
  }

  /* 学习曲线: 训练/验证均值 ± std 带 */
  function learningCurve(el, d) {
    const band = (mean, std, color) => [
      { type: "line", data: mean.map((m, i) => [d.train_sizes[i], +(m - std[i]).toFixed(4)]), lineStyle: { opacity: 0 }, stack: `band-${color}`, symbol: "none", silent: true },
      { type: "line", data: mean.map((m, i) => [d.train_sizes[i], +(2 * std[i]).toFixed(4)]), lineStyle: { opacity: 0 }, areaStyle: { color, opacity: 0.15 }, stack: `band-${color}`, symbol: "none", silent: true },
    ];
    make(el).setOption({
      title: { text: `学习曲线 (${d.scoring})`, left: "center", textStyle: { fontSize: 13 } },
      tooltip: { trigger: "axis", confine: true },
      legend: { bottom: 0, textStyle: { fontSize: 10 } },
      grid: { left: 50, right: 20, top: 40, bottom: 50 },
      xAxis: { type: "value", name: "训练样本数" },
      yAxis: { type: "value", name: d.scoring },
      series: [
        ...band(d.train_mean, d.train_std, "#4f7cff"),
        ...band(d.val_mean, d.val_std, "#f46a6a"),
        { name: "训练分数", type: "line", data: d.train_sizes.map((s, i) => [s, d.train_mean[i]]), itemStyle: { color: "#4f7cff" } },
        { name: "验证分数", type: "line", data: d.train_sizes.map((s, i) => [s, d.val_mean[i]]), itemStyle: { color: "#f46a6a" } },
      ],
    }, true);
  }

  /* 决策边界: 网格热力 + 样本散点 */
  function decisionBoundary(el, d) {
    const gx = d.grid.x, gy = d.grid.y, gz = d.grid.z;
    const heatData = gz.map((z, i) => [i % gx.length, Math.floor(i / gx.length), z]);
    const labelOf = (v) => (d.classes ? d.classes[v] : String(v));
    make(el).setOption({
      title: { text: `决策边界${d.used_pca ? " (PCA 2D)" : ""} · 精度 ${d.accuracy_2d}`, left: "center", textStyle: { fontSize: 13 } },
      tooltip: { ...baseTooltip, formatter: (p) => (p.seriesType === "scatter" ? `${labelOf(p.value[2])}<br/>(${p.value[0]}, ${p.value[1]})` : labelOf(p.value[2])) },
      grid: { left: 60, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "category", name: d.x_name, data: gx.map((v) => v.toFixed(1)), axisLabel: { fontSize: 9, interval: Math.ceil(gx.length / 10) } },
      yAxis: { type: "category", name: d.y_name, data: gy.map((v) => v.toFixed(1)), axisLabel: { fontSize: 9, interval: Math.ceil(gy.length / 10) } },
      visualMap: { show: false, type: "piecewise", dimension: 2, categories: [...new Set(gz)], inRange: { color: PALETTE.map((c) => c + "44") } },
      series: [
        { type: "heatmap", data: heatData, silent: true, label: { show: false } },
        {
          type: "scatter", symbolSize: 7,
          data: d.points.map((p) => {
            const xi = gx.findIndex((v) => v >= p.x) === -1 ? gx.length - 1 : Math.max(0, gx.findIndex((v) => v >= p.x) - 0.5);
            const yi = gy.findIndex((v) => v >= p.y) === -1 ? gy.length - 1 : Math.max(0, gy.findIndex((v) => v >= p.y) - 0.5);
            return { value: [p.x.toFixed(2), p.y.toFixed(2), p.label], x: xi, y: yi };
          }),
          encode: { x: 0, y: 1 },
          itemStyle: { color: (p) => PALETTE[p.value[2] % PALETTE.length], borderColor: "#fff", borderWidth: 1 },
        },
      ],
    }, true);
    // 散点需要数值轴: 用第二个坐标系覆盖
    const inst = make(el);
    inst.setOption({
      xAxis: [{ show: true }, { show: false, type: "value", min: gx[0], max: gx[gx.length - 1], gridIndex: 0 }],
      yAxis: [{ show: true }, { show: false, type: "value", min: gy[0], max: gy[gy.length - 1], gridIndex: 0 }],
      series: [
        { type: "heatmap", data: heatData, silent: true, label: { show: false } },
        {
          type: "scatter", symbolSize: 7, xAxisIndex: 1, yAxisIndex: 1,
          data: d.points.map((p) => [p.x, p.y, p.label]),
          itemStyle: { color: (p) => PALETTE[p.value[2] % PALETTE.length], borderColor: "#fff", borderWidth: 1 },
        },
      ],
    }, true);
  }

  /* 通用 2D 散点: d.points 或 d.scatter = [{x, y, label?|cluster?}] */
  function scatter2d(el, d, opts) {
    opts = opts || {};
    const key = opts.cluster ? "cluster" : "label";
    const points = d.points || d.scatter || [];
    const groups = {};
    points.forEach((p) => {
      const g = p[key] === undefined ? "样本" : String(p[key]);
      (groups[g] = groups[g] || []).push([p.x, p.y]);
    });
    const names = Object.keys(groups).sort();
    make(el).setOption({
      title: { text: opts.title || "散点图", left: "center", textStyle: { fontSize: 13 } },
      tooltip: { ...baseTooltip, formatter: (p) => `${p.seriesName}<br/>(${p.value[0]}, ${p.value[1]})` },
      legend: { bottom: 0, type: "scroll", textStyle: { fontSize: 10 } },
      grid: { left: 60, right: 20, top: 40, bottom: 50 },
      xAxis: { type: "value", name: d.x_name, scale: true },
      yAxis: { type: "value", name: d.y_name, scale: true },
      series: names.map((n, i) => ({
        name: opts.cluster ? (n === "-1" ? "噪声" : `簇 ${n}`) : n,
        type: "scatter", symbolSize: 8, data: groups[n],
        itemStyle: { color: n === "-1" ? "#bbb" : PALETTE[i % PALETTE.length], opacity: 0.85 },
      })),
    }, true);
  }

  /* 肘部法则折线 */
  function elbow(el, data) {
    make(el).setOption({
      title: { text: "K-Means 肘部法则", left: "center", textStyle: { fontSize: 13 } },
      tooltip: { trigger: "axis", confine: true },
      grid: { left: 70, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "category", name: "K", data: data.map((d) => d.k) },
      yAxis: { type: "value", name: "Inertia", scale: true },
      series: [{ type: "line", data: data.map((d) => d.inertia), itemStyle: { color: "#f7b84b" }, symbolSize: 8 }],
    }, true);
  }

  /* 方差解释 (PCA) */
  function variance(el, evr, cum) {
    make(el).setOption({
      title: { text: "PCA 方差解释", left: "center", textStyle: { fontSize: 13 } },
      tooltip: { trigger: "axis", confine: true },
      legend: { bottom: 0, textStyle: { fontSize: 10 } },
      grid: { left: 50, right: 50, top: 40, bottom: 50 },
      xAxis: { type: "category", data: evr.map((_, i) => `PC${i + 1}`) },
      yAxis: [{ type: "value", name: "方差比", max: 1 }, { type: "value", name: "累计", max: 1 }],
      series: [
        { name: "方差解释比", type: "bar", data: evr, itemStyle: { color: "#4f7cff" }, barMaxWidth: 30 },
        { name: "累计方差", type: "line", yAxisIndex: 1, data: cum, itemStyle: { color: "#f46a6a" } },
      ],
    }, true);
  }

  return { metricBar, rankingBar, confusionMatrix, importance, roc, learningCurve,
           decisionBoundary, scatter2d, elbow, variance };
})();
