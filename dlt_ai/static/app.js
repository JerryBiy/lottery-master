const state = {
  mode: "model",
  historyOffset: 0,
};

const $ = (id) => document.getElementById(id);

document.addEventListener("DOMContentLoaded", () => {
  wireControls();
  refreshAll();
});

function wireControls() {
  document.querySelectorAll(".segment").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".segment").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      state.mode = button.dataset.mode;
      $("modelSelect").disabled = state.mode === "random";
      $("candidateCount").disabled = state.mode === "random";
      $("autoTrain").disabled = state.mode === "random";
    });
  });

  $("refreshBtn").addEventListener("click", refreshAll);
  $("evaluateBtn").addEventListener("click", evaluatePredictions);
  $("predictBtn").addEventListener("click", predict);
  $("trainBtn").addEventListener("click", trainSelectedModel);
  $("holdoutBtn").addEventListener("click", runHoldout);
  $("recommendBtn").addEventListener("click", runRecommendation);
  $("historySearch").addEventListener("input", debounce(loadHistory, 250));
}

async function refreshAll() {
  setStatus("正在尝试抓取最新开奖数据...");
  try {
    const result = await postJSON("/api/refresh", {});
    setStatus(result.ok ? `数据已同步：${result.message}` : result.message, result.ok ? "ok" : "warn");
  } catch (error) {
    setStatus(`自动刷新失败：${error.message}`, "warn");
  }
  await Promise.all([loadSummary(), loadHistory(), loadPredictions(), loadMetrics(), loadDefaultRecommendation()]);
}

async function loadSummary() {
  const summary = await getJSON("/api/summary");
  const cards = $("summaryCards").querySelectorAll(".metric strong");
  cards[0].textContent = summary.draw_count ?? "-";
  cards[1].textContent = summary.latest?.issue ?? "-";
  cards[2].textContent = summary.latest ? `${summary.latest.front_text} + ${summary.latest.back_text}` : "-";
  cards[3].textContent = summary.pending_count ?? "-";

  $("modelCards").innerHTML = summary.models.map(modelCard).join("");
}

async function loadHistory() {
  const q = encodeURIComponent($("historySearch").value.trim());
  const data = await getJSON(`/api/history?limit=80&q=${q}`);
  $("historyBody").innerHTML = data.rows.map(drawRow).join("");
}

async function loadPredictions() {
  const rows = await getJSON("/api/predictions?limit=40");
  $("predictionList").innerHTML = rows.length ? rows.map(predictionCard).join("") : `<div class="muted">还没有保存预测。</div>`;
}

async function loadMetrics() {
  const rows = await getJSON("/api/metrics");
  $("metricsTable").innerHTML = rows.length ? rows.map(metricRow).join("") : `<div class="muted">等保存的预测被真实开奖命中后，这里会显示模型表现。</div>`;
}

async function loadDefaultRecommendation() {
  $("recommendResult").innerHTML = `<div class="muted">正在读取推荐缓存；如最新期号变化，会自动跑一次快速推荐。</div>`;
  try {
    const data = await getJSON("/api/recommend/default");
    $("recommendResult").innerHTML = renderRecommendationBlock(data);
  } catch (error) {
    $("recommendResult").innerHTML = `<div class="muted">默认推荐暂不可用：${escapeHTML(error.message)}</div>`;
  }
}

async function predict() {
  $("predictBtn").disabled = true;
  $("predictBtn").textContent = "生成中...";
  try {
    const payload = {
      mode: state.mode,
      model: $("modelSelect").value,
      trainWindow: Number($("trainWindow").value),
      trainMonths: Number($("trainMonths").value),
      tickets: Number($("ticketCount").value),
      candidates: Number($("candidateCount").value),
      save: $("savePrediction").checked,
      autoTrain: $("autoTrain").checked,
    };
    const data = await postJSON("/api/predict", payload);
    $("ticketList").innerHTML = data.tickets.map(ticketCard).join("");
    setStatus(`已基于 ${data.trained_until_issue} 生成 ${data.tickets.length} 注${data.prediction_id ? "，并保存记录" : ""}`, "ok");
    await Promise.all([loadSummary(), loadPredictions(), loadMetrics()]);
  } catch (error) {
    setStatus(`预测失败：${error.message}`, "warn");
  } finally {
    $("predictBtn").disabled = false;
    $("predictBtn").textContent = "生成并保存";
  }
}

async function trainSelectedModel() {
  $("trainBtn").disabled = true;
  $("trainBtn").textContent = "训练中...";
  try {
    const model = $("modelSelect").value;
    const data = await postJSON("/api/train", {
      model,
      trainWindow: Number($("trainWindow").value),
      trainMonths: Number($("trainMonths").value),
    });
    setStatus(`${model} 已训练到 ${data.summary.last_issue}`, "ok");
    $("trainResult").innerHTML = trainingMetrics(data.summary);
    await loadSummary();
  } catch (error) {
    setStatus(`训练失败：${error.message}`, "warn");
  } finally {
    $("trainBtn").disabled = false;
    $("trainBtn").textContent = "训练所选模型";
  }
}

async function evaluatePredictions() {
  const data = await postJSON("/api/evaluate", {});
  setStatus(`已评估 ${data.evaluated} 条预测记录`, "ok");
  await Promise.all([loadSummary(), loadPredictions(), loadMetrics()]);
}

async function runHoldout() {
  $("holdoutBtn").disabled = true;
  $("holdoutBtn").textContent = "评估中...";
  $("holdoutResult").innerHTML = `<div class="muted">正在训练并评估，这可能需要几十秒。</div>`;
  try {
    const payload = {
      model: $("modelSelect").value,
      trainUntilIssue: $("holdoutIssue").value,
      trainWindow: Number($("trainWindow").value),
      trainMonths: Number($("trainMonths").value),
      tickets: Number($("holdoutTickets").value),
      candidates: Number($("holdoutCandidates").value),
      maxTestDraws: Number($("holdoutDraws").value),
    };
    const data = await postJSON("/api/holdout", payload);
    $("holdoutResult").innerHTML = data.rows.map(holdoutRow).join("") +
      `<div class="score">测试期数 ${data.test_draws} · 明细已保存到 reports/</div>`;
    setStatus("模型能力评估完成", "ok");
  } catch (error) {
    $("holdoutResult").innerHTML = `<div class="muted">评估失败：${escapeHTML(error.message)}</div>`;
    setStatus(`评估失败：${error.message}`, "warn");
  } finally {
    $("holdoutBtn").disabled = false;
    $("holdoutBtn").textContent = "开始模型能力评估";
  }
}

async function runRecommendation() {
  $("recommendBtn").disabled = true;
  $("recommendBtn").textContent = "搜索中...";
  $("recommendResult").innerHTML = `<div class="muted">正在跑自定义网格。模型越多、窗口越多、测试期越长，就会越慢。</div>`;
  try {
    const payload = {
      models: $("recommendModels").value.split(",").map((x) => x.trim()).filter(Boolean),
      windows: $("recommendWindows").value.split(",").map((x) => x.trim()).filter(Boolean),
      trainUntilIssue: $("holdoutIssue").value,
      tickets: Number($("holdoutTickets").value),
      candidates: Number($("holdoutCandidates").value),
      maxTestDraws: Number($("holdoutDraws").value),
    };
    const data = await postJSON("/api/recommend", payload);
    $("recommendResult").innerHTML = renderRecommendationBlock(data) +
      `<div class="score">自定义推荐报告已保存到 reports/</div>`;
    const best = data.rows.find((row) => !row.error);
    if (best) {
      setStatus(`当前推荐：${best.model} / ${best.train_window_draws || "all"} 期窗口`, "ok");
    }
  } catch (error) {
    $("recommendResult").innerHTML = `<div class="muted">推荐失败：${escapeHTML(error.message)}</div>`;
    setStatus(`推荐失败：${error.message}`, "warn");
  } finally {
    $("recommendBtn").disabled = false;
    $("recommendBtn").textContent = "运行自定义搜索";
  }
}

function modelCard(model) {
  const cls = model.fresh ? "ok" : "warn";
  const status = model.exists ? (model.fresh ? "最新" : "需更新") : "未训练";
  return `
    <div class="model-card">
      <div class="prediction-head">
        <strong>${escapeHTML(model.name)}</strong>
        <span class="status ${cls}">${status}</span>
      </div>
      <small>训练到期号：${model.last_issue || "-"} · 文件：${model.size_mb ?? "-"} MB</small>
    </div>
  `;
}

function drawRow(row) {
  return `
    <tr>
      <td>${escapeHTML(row.issue)}</td>
      <td>${balls(row.front, "front-ball")}</td>
      <td>${balls(row.back, "back-ball")}</td>
      <td>${escapeHTML(row.date)}</td>
    </tr>
  `;
}

function ticketCard(ticket, index) {
  const front = splitNums(ticket.front);
  const back = splitNums(ticket.back);
  return `
    <div class="ticket">
      <div class="balls">${balls(front, "front-ball")}${balls(back, "back-ball")}</div>
      <div class="score">#${index + 1}${ticket.score === null || ticket.score === undefined ? "" : ` · score ${Number(ticket.score).toFixed(4)}`}</div>
    </div>
  `;
}

function predictionCard(row) {
  const best = row.status === "evaluated"
    ? `最佳：前区 ${row.best_front_hits}，后区 ${row.best_back_hits}${row.best_prize ? `，${row.best_prize}等奖` : ""}`
    : "等待下一期开奖";
  const sample = row.tickets.slice(0, 3).map((ticket) => `${ticket.front} + ${ticket.back}`).join("<br>");
  return `
    <div class="prediction">
      <div class="prediction-head">
        <strong>#${row.id} ${escapeHTML(row.mode)} / ${escapeHTML(row.model_name)}</strong>
        <span class="status ${row.status === "evaluated" ? "ok" : "warn"}">${escapeHTML(row.status)}</span>
      </div>
      <small>创建：${escapeHTML(row.created_at)} · 训练到：${escapeHTML(row.trained_until_issue)} · 对比期：${row.target_issue || "-"}</small>
      <p>${best}</p>
      <small>${sample}</small>
    </div>
  `;
}

function metricRow(row) {
  return `
    <div class="metric-row">
      <strong>${escapeHTML(row.mode)} / ${escapeHTML(row.model)}</strong>
      <div class="score">票数 ${row.tickets} · 前区均值 ${row.avg_front_hits.toFixed(3)} · 后区均值 ${row.avg_back_hits.toFixed(3)} · 中奖率 ${(row.prize_rate * 100).toFixed(2)}% · 最佳奖级 ${row.best_prize || "-"}</div>
    </div>
  `;
}

function holdoutRow(row) {
  return `
    <div class="metric-row">
      <strong>${escapeHTML(row.strategy)}</strong>
      <div class="score">
        票数 ${row.tickets} · 前区均值 ${Number(row.avg_front_hits).toFixed(3)}
        · 后区均值 ${Number(row.avg_back_hits).toFixed(3)}
        · 中奖率 ${(Number(row.any_prize_rate) * 100).toFixed(2)}%
        · 前区Lift ${Number(row.front_lift_vs_random || 0).toFixed(3)}
        · 后区Lift ${Number(row.back_lift_vs_random || 0).toFixed(3)}
        · 中奖率Lift ${(Number(row.prize_rate_lift_vs_random || 0) * 100).toFixed(2)}%
      </div>
    </div>
  `;
}

function recommendRow(row) {
  if (row.error) {
    return `<div class="metric-row"><strong>${escapeHTML(row.model)} / ${row.train_window_draws || "all"}</strong><div class="score">失败：${escapeHTML(row.error)}</div></div>`;
  }
  return `
    <div class="metric-row">
      <strong>${escapeHTML(row.model)} · ${row.train_window_draws || "all"} 期窗口 · score ${Number(row.score).toFixed(3)}</strong>
      <div class="score">
        前区Lift ${Number(row.front_lift_vs_random || 0).toFixed(3)}
        · 后区Lift ${Number(row.back_lift_vs_random || 0).toFixed(3)}
        · 中奖率Lift ${(Number(row.prize_rate_lift_vs_random || 0) * 100).toFixed(2)}%
        · 中奖率 ${(Number(row.any_prize_rate || 0) * 100).toFixed(2)}%
      </div>
    </div>
  `;
}

function renderRecommendationBlock(data) {
  const best = (data.rows || []).find((row) => !row.error);
  const banner = best ? `
    <div class="metric-row">
      <strong>推荐：${escapeHTML(best.model)} · ${best.train_window_draws || "all"} 期窗口</strong>
      <div class="score">
        score ${Number(best.score || 0).toFixed(3)}
        · 前区Lift ${Number(best.front_lift_vs_random || 0).toFixed(3)}
        · 后区Lift ${Number(best.back_lift_vs_random || 0).toFixed(3)}
        · 中奖率Lift ${(Number(best.prize_rate_lift_vs_random || 0) * 100).toFixed(2)}%
        ${data.stale ? " · 使用过期缓存" : data.cached ? " · 使用缓存" : " · 刚刚计算"}
      </div>
    </div>
  ` : "";
  return banner + (data.rows || []).map(recommendRow).join("");
}

function trainingMetrics(summary) {
  const rows = [
    ["Front ROC AUC", summary.front_metrics.roc_auc],
    ["Front Avg Precision", summary.front_metrics.avg_precision],
    ["Front Brier", 1 - summary.front_metrics.brier],
    ["Back ROC AUC", summary.back_metrics.roc_auc],
    ["Back Avg Precision", summary.back_metrics.avg_precision],
    ["Back Brier", 1 - summary.back_metrics.brier],
  ];
  return rows.map(([label, value]) => {
    const pct = Math.max(0, Math.min(1, Number(value) || 0)) * 100;
    return `<div class="chart-card"><strong>${label}: ${Number(value).toFixed(4)}</strong><div class="bar"><span style="width:${pct}%"></span></div></div>`;
  }).join("");
}

function balls(values, cls) {
  return values.map((value) => `<span class="ball ${cls}">${String(value).padStart(2, "0")}</span>`).join("");
}

function splitNums(text) {
  return String(text).split(/\s+/).filter(Boolean).map(Number);
}

async function getJSON(url) {
  const response = await fetch(url);
  return parseResponse(response);
}

async function postJSON(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

async function parseResponse(response) {
  const text = await response.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(text || response.statusText);
  }
  if (!response.ok) {
    throw new Error(data.error || data.message || response.statusText);
  }
  return data;
}

function setStatus(message, level = "") {
  const el = $("refreshStatus");
  el.textContent = message;
  el.className = level;
}

function debounce(fn, wait) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[char]);
}
