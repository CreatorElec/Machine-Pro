/* 机器学习实验平台 - 前端逻辑 */
"use strict";

const state = {
  datasets: [],
  algorithms: { classification: [], regression: [] },
  current: { ref: null, info: null, task: "classification" },
};

const $ = (id) => document.getElementById(id);

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const data = await res.json();
      msg = data.detail || msg;
    } catch (_) {
      /* ignore */
    }
    throw new Error(msg);
  }
  return res.json();
}

/* ================= 初始化 ================= */
async function init() {
  bindEvents();
  await Promise.all([loadDatasets(), loadAlgorithms()]);
}

function bindEvents() {
  $("dataset-select").addEventListener("change", (e) => selectDataset(e.target.value));
  $("target-select").addEventListener("change", onTargetChange);
  $("algorithm-select").addEventListener("change", onAlgorithmChange);
  $("csv-file").addEventListener("change", onUpload);
  $("test-size").addEventListener("input", (e) => {
    $("test-size-val").textContent = e.target.value + "%";
  });
  $("random-state").addEventListener("input", (e) => {
    if (e.target.value < 0) e.target.value = 0;
  });
  $("check-all").addEventListener("click", () => setFeaturesByNumeric(true));
  $("uncheck-all").addEventListener("click", () => setAllFeatures(false));
  $("train-btn").addEventListener("click", onTrain);
}

/* ================= 数据集 ================= */
async function loadDatasets() {
  const list = await api("/api/datasets");
  state.datasets = list;
  const sel = $("dataset-select");
  sel.innerHTML = list
    .map((d) => `<option value="${d.key}">${escapeHtml(d.name)}</option>`)
    .join("");
  if (list.length) await selectDataset(list[0].key);
}

async function selectDataset(ref) {
  try {
    state.current.ref = ref;
    const info = await api(`/api/datasets/${ref}/info`);
    state.current.info = info;
    renderDatasetMeta(info);
    renderTargetOptions(info);
    renderFeatureList(info);
    applyTask(info.suggested_task);
  } catch (err) {
    setStatus(err.message, true);
  }
}

function renderDatasetMeta(info) {
  const totalMissing = Object.values(info.missing).reduce((a, b) => a + b, 0);
  const cols = info.columns;
  const head = `<tr>${cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr>`;
  const rows = info.preview
    .map(
      (row) =>
        `<tr>${cols.map((c) => `<td>${row[c] === null ? "—" : escapeHtml(row[c])}</td>`).join("")}</tr>`
    )
    .join("");
  $("dataset-meta").innerHTML = `
    <div class="meta-grid">
      <div><span class="meta-label">样本数</span><span class="meta-value">${info.n_rows}</span></div>
      <div><span class="meta-label">特征数</span><span class="meta-value">${info.columns.length}</span></div>
      <div><span class="meta-label">缺失值</span><span class="meta-value">${totalMissing}</span></div>
    </div>
    <p class="hint">${escapeHtml(info.description)}</p>
    <div class="preview-wrap"><table class="preview"><thead>${head}</thead><tbody>${rows}</tbody></table></div>`;
}

function renderTargetOptions(info) {
  const sel = $("target-select");
  sel.innerHTML = info.columns
    .map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`)
    .join("");
  if (info.suggested_target && info.columns.includes(info.suggested_target)) {
    sel.value = info.suggested_target;
  }
}

function renderFeatureList(info) {
  const container = $("feature-list");
  container.innerHTML = info.columns
    .map((c) => {
      const numeric = info.numeric_columns.includes(c);
      return `<label title="${escapeHtml(c)}">
        <input type="checkbox" class="feature-cb" value="${escapeHtml(c)}" />
        <span>${escapeHtml(c)}${numeric ? " 🔢" : " 🏷️"}</span>
      </label>`;
    })
    .join("");
  // 默认选中: 数值列(排除目标列)
  const target = $("target-select").value;
  const defaultFeatures = info.numeric_columns.filter((c) => c !== target);
  container.querySelectorAll(".feature-cb").forEach((cb) => {
    cb.checked = defaultFeatures.length
      ? defaultFeatures.includes(cb.value)
      : cb.value !== target;
  });
}

function setAllFeatures(checked) {
  const target = $("target-select").value;
  document.querySelectorAll(".feature-cb").forEach((cb) => {
    cb.checked = checked && cb.value !== target;
  });
}

function setFeaturesByNumeric() {
  const info = state.current.info;
  const target = $("target-select").value;
  document.querySelectorAll(".feature-cb").forEach((cb) => {
    cb.checked = info.numeric_columns.includes(cb.value) && cb.value !== target;
  });
}

function onTargetChange() {
  const info = state.current.info;
  const target = $("target-select").value;
  const task = info.column_tasks[target] || "classification";
  applyTask(task);
}

function applyTask(task) {
  state.current.task = task;
  const badge = $("task-badge");
  if (task === "regression") {
    badge.textContent = "回归";
    badge.className = "task-badge task-regression";
  } else {
    badge.textContent = "分类";
    badge.className = "task-badge task-classification";
  }
  renderAlgorithms(task);
}

/* ================= 算法 ================= */
async function loadAlgorithms() {
  state.algorithms = await api("/api/algorithms");
  if (state.current.task) renderAlgorithms(state.current.task);
}

function renderAlgorithms(task) {
  const list = state.algorithms[task] || [];
  const sel = $("algorithm-select");
  sel.innerHTML = list
    .map((a) => `<option value="${a.key}">${escapeHtml(a.name)}</option>`)
    .join("");
  if (list.length) onAlgorithmChange();
}

function onAlgorithmChange() {
  const task = state.current.task;
  const key = $("algorithm-select").value;
  const alg = (state.algorithms[task] || []).find((a) => a.key === key);
  if (!alg) return;
  $("algorithm-desc").textContent = alg.description || "";
  renderParamForm(alg.params);
}

function renderParamForm(params) {
  const form = $("param-form");
  if (!params || !params.length) {
    form.innerHTML = '<p class="hint">该算法无可调参数</p>';
    return;
  }
  form.innerHTML = params
    .map((p) => {
      const id = `param-${p.name}`;
      let input;
      if (p.type === "select") {
        input = `<select id="${id}" name="${p.name}">${p.options
          .map(
            (o) =>
              `<option value="${escapeHtml(o)}" ${String(o) === String(p.default) ? "selected" : ""}>${escapeHtml(o === null ? "无限制" : o)}</option>`
          )
          .join("")}</select>`;
      } else if (p.type === "int") {
        input = `<input type="number" id="${id}" name="${p.name}" value="${p.default}"
          ${p.min != null ? `min="${p.min}"` : ""} ${p.max != null ? `max="${p.max}"` : ""} step="1" />`;
      } else {
        input = `<input type="number" id="${id}" name="${p.name}" value="${p.default}"
          ${p.min != null ? `min="${p.min}"` : ""} ${p.max != null ? `max="${p.max}"` : ""}
          step="${p.step || "any"}" />`;
      }
      return `<div class="param-row"><label for="${id}">${escapeHtml(p.label)}</label>${input}</div>`;
    })
    .join("");
}

function collectParams() {
  const params = {};
  document.querySelectorAll("#param-form [name]").forEach((el) => {
    params[el.name] = el.value;
  });
  return params;
}

/* ================= 上传 ================= */
async function onUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  setStatus("正在上传解析...");
  try {
    const res = await api("/api/upload", { method: "POST", body: fd });
    state.current.ref = res.ref;
    state.current.info = res.info;
    // 将上传数据集加入下拉框
    const sel = $("dataset-select");
    const opt = document.createElement("option");
    opt.value = res.ref;
    opt.textContent = `📁 ${res.info.name}`;
    sel.appendChild(opt);
    sel.value = res.ref;
    renderDatasetMeta(res.info);
    renderTargetOptions(res.info);
    renderFeatureList(res.info);
    applyTask(res.info.suggested_task);
    setStatus("上传成功, 请选择目标列与特征");
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    e.target.value = "";
  }
}

/* ================= 训练 ================= */
async function onTrain() {
  const info = state.current.info;
  if (!info) {
    setStatus("请先选择数据集", true);
    return;
  }
  const features = [...document.querySelectorAll(".feature-cb:checked")].map(
    (cb) => cb.value
  );
  if (!features.length) {
    setStatus("请至少选择一个特征列", true);
    return;
  }
  const target = $("target-select").value;
  const task = info.column_tasks[target] || state.current.task;

  const payload = {
    dataset: state.current.ref,
    target,
    features,
    algorithm: $("algorithm-select").value,
    task,
    params: collectParams(),
    test_size: Number($("test-size").value) / 100,
    random_state: Number($("random-state").value) || 42,
  };

  const btn = $("train-btn");
  btn.disabled = true;
  setStatus("正在训练模型...");
  try {
    const data = await api("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderResults(data);
    setStatus("✅ 训练完成");
    $("empty-state").classList.add("hidden");
    $("results").classList.remove("hidden");
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    setStatus("❌ " + err.message, true);
  } finally {
    btn.disabled = false;
  }
}

function setStatus(msg, isError = false) {
  const el = $("train-status");
  el.textContent = msg;
  el.className = "status" + (isError ? " error" : "");
}

/* ================= 结果渲染 ================= */
function renderResults(data) {
  const summary = `
    <div class="result-summary">
      <span class="algo-name">${escapeHtml(data.algorithm_name)}</span>
      <span class="summary-tag">任务: ${data.task === "classification" ? "分类" : "回归"}</span>
      <span class="summary-tag">训练集 ${data.train_size} 条</span>
      <span class="summary-tag">测试集 ${data.test_size} 条</span>
      <span class="summary-tag">特征 ${data.n_features_used} 维</span>
    </div>`;

  const metrics = renderMetrics(data.task, data.metrics);

  const cmSection =
    data.task === "classification"
      ? `<div class="card"><h3>🧩 混淆矩阵</h3>${renderConfusionMatrix(
          data.confusion_matrix,
          data.classes
        )}</div>`
      : "";

  const reportSection =
    data.task === "classification"
      ? `<div class="card"><h3>📋 分类报告</h3>${renderReport(data.classification_report)}</div>`
      : "";

  const fiSection = data.feature_importance.length
    ? `<div class="card"><h3>⭐ 特征重要性</h3>${renderFeatureImportance(
        data.feature_importance
      )}</div>`
    : "";

  const predSection = `<div class="card"><h3>🔍 预测样本对比 (测试集前 10 条)</h3>${renderPredictions(
    data.predictions
  )}</div>`;

  $("results").innerHTML =
    summary + metrics + cmSection + reportSection + fiSection + predSection;
}

function renderMetrics(task, m) {
  let items;
  if (task === "classification") {
    items = [
      ["准确率 Accuracy", m.accuracy],
      ["精确率 Precision (宏)", m.precision_macro],
      ["召回率 Recall (宏)", m.recall_macro],
      ["F1 分数 (宏)", m.f1_macro],
      ["F1 分数 (加权)", m.f1_weighted],
    ];
  } else {
    items = [
      ["R² 决定系数", m.r2],
      ["均方误差 MSE", m.mse],
      ["均方根误差 RMSE", m.rmse],
      ["平均绝对误差 MAE", m.mae],
    ];
  }
  return `<div class="metric-grid">${items
    .map(
      ([label, value]) =>
        `<div class="metric-card"><div class="metric-value">${fmt(value)}</div><div class="metric-label">${label}</div></div>`
    )
    .join("")}</div>`;
}

function renderConfusionMatrix(cm, classes) {
  const max = Math.max(...cm.flat(), 1);
  let html = '<table class="cm"><tr><th></th>';
  classes.forEach((c) => (html += `<th>预测: ${escapeHtml(c)}</th>`));
  html += "</tr>";
  cm.forEach((row, i) => {
    html += `<tr><th>真实: ${escapeHtml(classes[i])}</th>`;
    row.forEach((v) => {
      const intensity = v / max;
      const bg = `rgba(79, 70, 229, ${0.06 + intensity * 0.85})`;
      const color = intensity > 0.5 ? "#fff" : "#1f2937";
      html += `<td style="background:${bg};color:${color}">${v}</td>`;
    });
    html += "</tr>";
  });
  return html + "</table>";
}

function renderReport(report) {
  const rows = [];
  for (const [label, v] of Object.entries(report)) {
    if (label === "accuracy" || typeof v !== "object" || v === null) continue;
    rows.push(
      `<tr><td>${escapeHtml(label)}</td><td>${fmt(v.precision)}</td><td>${fmt(
        v.recall
      )}</td><td>${fmt(v["f1-score"])}</td><td>${v.support}</td></tr>`
    );
  }
  return `<table class="report"><thead><tr><th>类别</th><th>精确率</th><th>召回率</th><th>F1</th><th>样本数</th></tr></thead><tbody>${rows.join(
    ""
  )}</tbody></table>`;
}

function renderFeatureImportance(items) {
  const max = Math.max(...items.map((i) => i.importance));
  return items
    .map(
      (i) => `<div class="fi-row">
        <div class="fi-label" title="${escapeHtml(i.feature)}">${escapeHtml(i.feature)}</div>
        <div class="fi-bar-track"><div class="fi-bar" style="width:${(
          (i.importance / max) *
          100
        ).toFixed(1)}%"></div></div>
        <div class="fi-val">${i.importance.toFixed(4)}</div>
      </div>`
    )
    .join("");
}

function renderPredictions(preds) {
  const rows = preds
    .map((p) => {
      const ok = String(p.true) === String(p.pred);
      return `<tr><td>${p.index + 1}</td><td>${escapeHtml(p.true)}</td><td>${escapeHtml(
        p.pred
      )}</td><td>${ok ? "✅" : "❌"}</td></tr>`;
    })
    .join("");
  return `<table class="pred"><thead><tr><th>#</th><th>真实值</th><th>预测值</th><th>结果</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function fmt(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? v : v.toFixed(4);
  return v;
}

document.addEventListener("DOMContentLoaded", init);
