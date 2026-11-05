"""`/evaluate` 单页评测面板（FastAPI 直接托管，位于 8001 端口）。

页面不要求登录：离线/在线评测记录全部可见；在线评测对每个用户取最近 10 条
带 Langfuse trace 的问答，并支持按用户名筛选运行记录与用例。
"""

from __future__ import annotations

PAGE_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent 综合评测 · 招投标问答</title>
<style>
  :root {
    --bg: #f8fafc; --card: #fff; --line: #e2e8f0; --text: #0f172a;
    --muted: #64748b; --brand: #2563eb; --ok: #059669; --warn: #d97706;
    --bad: #dc2626; --soft: #f1f5f9;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; }
  header { background: #0f172a; color: #fff; padding: 20px 32px; }
  header h1 { margin: 0; font-size: 20px; letter-spacing: .02em; }
  header p { margin: 6px 0 0; color: #94a3b8; font-size: 12px; }
  main { max-width: 1220px; margin: 24px auto; padding: 0 20px 60px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
    padding: 18px 20px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(15,23,42,.03); }
  .row { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
  .between { justify-content: space-between; }
  h2 { font-size: 15px; margin: 0 0 12px; }
  h3 { font-size: 13px; margin: 18px 0 8px; color: var(--muted); font-weight: 600; }
  .muted { color: var(--muted); font-size: 12px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px;
    background: var(--soft); color: var(--muted); font-size: 11px; margin-left: 6px; }
  .badge.ok { background: #ecfdf5; color: var(--ok); }
  .badge.warn { background: #fffbeb; color: var(--warn); }
  .badge.bad { background: #fef2f2; color: var(--bad); }
  button { border: 0; border-radius: 8px; padding: 9px 14px; font: inherit;
    cursor: pointer; background: var(--brand); color: #fff; }
  button.secondary { background: var(--soft); color: var(--text); }
  button:disabled { opacity: .5; cursor: not-allowed; }
  input { border: 1px solid var(--line); border-radius: 8px; padding: 9px 11px;
    font: inherit; min-width: 160px; background: #fff; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
  .metric { background: var(--soft); border-radius: 10px; padding: 14px; }
  .metric .label { color: var(--muted); font-size: 12px; }
  .metric .value { font-size: 26px; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; }
  .metric .note { color: var(--muted); font-size: 11px; margin-top: 2px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; vertical-align: top; }
  th { color: var(--muted); font-weight: 600; white-space: nowrap; }
  tbody tr[data-id] { cursor: pointer; }
  tbody tr[data-id]:hover { background: #f8fafc; }
  .num { font-variant-numeric: tabular-nums; white-space: nowrap; }
  .status-running { color: var(--brand); } .status-completed { color: var(--ok); }
  .status-failed { color: var(--bad); }
  .progress { height: 8px; background: var(--soft); border-radius: 999px; overflow: hidden; }
  .progress > div { height: 100%; background: var(--brand); width: 0; transition: width .25s; }
  .warnbox { background: #fffbeb; border: 1px solid #fde68a; color: #92400e;
    border-radius: 8px; padding: 10px 12px; margin-top: 10px; font-size: 12px; }
  .errorbox { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b;
    border-radius: 8px; padding: 10px 12px; margin-top: 10px; font-size: 12px; white-space: pre-wrap; }
  details { margin-top: 4px; } summary { cursor: pointer; color: var(--brand); font-size: 12px; }
  pre { white-space: pre-wrap; word-break: break-word; max-height: 280px; overflow: auto;
    background: var(--soft); padding: 8px; border-radius: 6px; font-size: 12px; }
  .detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px 20px; }
  .hidden { display: none !important; }
  @media (max-width: 640px) { header { padding: 16px; } main { padding: 0 12px 40px; }
    .card { padding: 14px; } }
</style>
</head>
<body>
<header>
  <h1>Agent 综合评测</h1>
  <p>Ragas 判分 · 工具选择准确率 · 忠诚度 / 相关性 / 召回率 · 平均延迟</p>
</header>
<main>
  <section class="card">
    <div class="row between">
      <div>
        <h2>评测配置 <span class="badge" id="enabledBadge"></span></h2>
        <div class="muted" id="configLine">加载中…</div>
      </div>
      <span class="badge" id="modeBadge">离线数据集 + 在线全用户</span>
    </div>
    <div id="warningBox" class="warnbox hidden"></div>
    <div class="row" style="margin-top:14px">
      <button id="offlineBtn">运行离线评测</button>
      <button id="onlineBtn" class="secondary">运行在线评测（每用户最近 10 条）</button>
      <span class="muted" id="runHint"></span>
    </div>
    <div class="errorbox hidden" id="runError"></div>
    <div id="progressBox" class="hidden" style="margin-top:14px">
      <div class="row between">
        <span class="muted" id="progressText"></span>
        <span class="muted num" id="progressPct"></span>
      </div>
      <div class="progress" style="margin-top:6px"><div id="progressBar"></div></div>
    </div>
  </section>

  <section class="card">
    <div class="row between">
      <h2>各项指标 <span class="badge" id="latestBadge"></span></h2>
      <span class="muted" id="latestMeta"></span>
    </div>
    <div class="grid" id="metricGrid"></div>
    <p class="muted" id="metricNote" style="margin:10px 0 0"></p>
  </section>

  <section class="card">
    <div class="row between">
      <h2>评测历史</h2>
      <div class="row">
        <input id="userFilter" list="userOptions" placeholder="按用户名筛选">
        <datalist id="userOptions"></datalist>
        <button class="secondary" id="applyFilter">筛选</button>
        <button class="secondary" id="clearFilter">全部</button>
      </div>
    </div>
    <p class="muted" id="filterHint"></p>
    <div style="overflow-x:auto">
      <table>
        <thead><tr>
          <th>时间</th><th>模式</th><th>用户</th><th>状态</th><th>工具准确率</th>
          <th>忠诚度</th><th>相关性</th><th>召回率</th><th>平均延迟</th>
        </tr></thead>
        <tbody id="historyBody"><tr><td colspan="9" class="muted">暂无评测记录</td></tr></tbody>
      </table>
    </div>
  </section>

  <section class="card hidden" id="detailCard">
    <div class="row between">
      <h2>评测详情 <span class="badge" id="detailBadge"></span></h2>
      <button class="secondary" id="closeDetail">关闭</button>
    </div>
    <div class="detail-grid muted" id="detailMeta"></div>
    <div class="errorbox hidden" id="detailError"></div>
    <div class="warnbox hidden" id="detailWarnings"></div>
    <h3>用例明细</h3>
    <div style="overflow-x:auto">
      <table>
        <thead><tr>
          <th>#</th><th>用户</th><th>问题</th><th>期望工具</th><th>实际工具</th>
          <th>延迟</th><th>忠诚度</th><th>相关性</th><th>召回率</th><th>结果</th>
        </tr></thead>
        <tbody id="caseBody"></tbody>
      </table>
    </div>
    <div id="caseDetails"></div>
  </section>
</main>

<script>
(() => {
  const $ = (id) => document.getElementById(id);
  let config = null;
  let pollTimer = null;
  let currentRun = null;
  let filterUser = "";

  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));

  const fmtTime = (iso) => {
    if (!iso) return "—";
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString("zh-CN", { hour12: false });
  };
  const fmtScore = (value) => {
    if (value === null || value === undefined) return "—";
    return (Number(value) * 100).toFixed(1) + "%";
  };
  const fmtLatency = (value) => {
    if (value === null || value === undefined) return "—";
    return Number(value).toFixed(0) + " ms";
  };
  const statusClass = (status) => "status-" + String(status || "running");
  const statusText = (status) => ({
    running: "运行中", completed: "已完成", failed: "失败"
  }[status] || status || "—");
  const modeText = (mode) => mode === "offline" ? "离线" : mode === "online" ? "在线" : mode;
  const summaryValue = (summary, key) => {
    if (key === "avg_latency_ms") return fmtLatency(summary?.[key]);
    return fmtScore(summary?.[key]);
  };
  const userQuery = () => filterUser ? "?username=" + encodeURIComponent(filterUser) : "";
  const runUsers = (run) => {
    if (run.mode !== "online") return "离线";
    const names = run.users || [];
    return names.length ? names.join("、") : "—";
  };

  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
    const res = await fetch(path, { ...options, headers });
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
    if (!res.ok) {
      const detail = data?.detail ?? data?.error?.message ?? ("HTTP " + res.status);
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  }

  function showRunError(message) {
    $("runError").textContent = message;
    $("runError").classList.remove("hidden");
    setTimeout(() => $("runError").classList.add("hidden"), 8000);
  }

  async function loadConfig() {
    try {
      config = await api("/api/evaluate/config");
      const judge = config.judge || {};
      const evaluated = config.evaluated || {};
      $("configLine").textContent =
        `判分 ${judge.provider}/${judge.model} · 被测 ${evaluated.provider}/${evaluated.model} · ` +
        `离线数据集 ${config.dataset?.cases ?? "?"} 条 · 每用户在线最多 ${config.online_limit} 条`;
      $("enabledBadge").textContent = config.enabled ? "已启用" : "未启用";
      $("enabledBadge").className = "badge " + (config.enabled ? "ok" : "bad");
      $("offlineBtn").disabled = !config.enabled || !config.agent_available || !config.dataset?.exists;
      $("onlineBtn").disabled =
        !config.enabled || !config.db_available || !config.langfuse_available;
      $("runHint").textContent = config.enabled
        ? (!config.dataset?.exists ? "离线数据集缺失" :
           (!config.langfuse_available ? "在线评测需要 Langfuse 可观测性" : ""))
        : (config.reason || "评测未启用");
      const warnings = config.warnings || [];
      $("warningBox").classList.toggle("hidden", warnings.length === 0);
      $("warningBox").textContent = warnings.join("；");
    } catch (err) {
      $("configLine").textContent = "配置加载失败：" + err.message;
    }
  }

  async function loadUsers() {
    try {
      const users = await api("/api/evaluate/users");
      $("userOptions").innerHTML = users.map((u) =>
        `<option value="${esc(u.username)}">${esc(u.cases)} 条用例</option>`
      ).join("");
    } catch { /* 用户列表只是筛选辅助 */ }
  }

  async function loadRuns() {
    try {
      const runs = await api("/api/evaluate/runs" + userQuery());
      renderHistory(runs);
      if (!currentRun && runs.length) await openRun(runs[0].id, false);
      else if (!runs.length) renderMetrics(null);
    } catch (err) {
      showRunError(err.message);
    }
  }

  function renderHistory(runs) {
    const body = $("historyBody");
    $("filterHint").textContent = filterUser
      ? `当前仅显示包含用户「${filterUser}」的在线评测；离线结果不属于个人，已隐藏。`
      : "";
    if (!runs.length) {
      body.innerHTML = '<tr><td colspan="9" class="muted">暂无评测记录</td></tr>';
      return;
    }
    body.innerHTML = runs.map((run) => {
      const s = run.summary || {};
      return `<tr data-id="${esc(run.id)}">
        <td class="num">${esc(fmtTime(run.created_at))}</td>
        <td>${esc(modeText(run.mode))}</td>
        <td>${esc(runUsers(run))}</td>
        <td class="${statusClass(run.status)}">${esc(statusText(run.status))}</td>
        <td class="num">${fmtScore(s.tool_selection_accuracy)}</td>
        <td class="num">${fmtScore(s.faithfulness)}</td>
        <td class="num">${fmtScore(s.answer_relevancy)}</td>
        <td class="num">${fmtScore(s.context_recall)}</td>
        <td class="num">${fmtLatency(s.avg_latency_ms)}</td>
      </tr>`;
    }).join("");
    body.querySelectorAll("tr[data-id]").forEach((row) => {
      row.addEventListener("click", () => openRun(row.dataset.id, true));
    });
  }

  function renderMetrics(run) {
    const summary = run?.summary || {};
    const grid = $("metricGrid");
    const keys = config?.metrics || [];
    grid.innerHTML = keys.map((metric) => {
      return `<div class="metric">
        <div class="label">${esc(metric.label)}</div>
        <div class="value">${esc(summaryValue(summary, metric.key))}</div>
        <div class="note">${run?.mode === "online" ? "在线：判分模型生成判断/参考答案" : "离线：人工数据集"}</div>
      </div>`;
    }).join("");
    $("latestBadge").textContent = run ? `${modeText(run.mode)} · ${statusText(run.status)}` : "暂无结果";
    $("latestMeta").textContent = run
      ? `运行 ${run.id} · 判分 ${run.judge?.provider}/${run.judge?.model} · ` +
        `用例 ${summary.cases ?? 0}（成功 ${summary.succeeded ?? 0}，失败 ${summary.failed ?? 0}）` +
        (run.filtered_by ? ` · 已筛选用户 ${run.filtered_by}` : "")
      : "";
    $("metricNote").textContent = run?.mode === "online"
      ? "在线评测对每个用户取最近 10 条带 Langfuse trace 的问答；工具选择与召回率由判分模型基于 Langfuse 工具记录生成判断和参考答案。"
      : "离线评测使用 data/processed/evaluate_cases.json，工具准确率与召回率为人工标注。";
  }

  async function openRun(runId, scroll) {
    try {
      const run = await api("/api/evaluate/runs/" + encodeURIComponent(runId) + userQuery());
      currentRun = run;
      renderMetrics(run);
      renderDetail(run);
      if (run.status === "running") startPolling(runId);
      if (scroll) $("detailCard").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      showRunError(err.message);
    }
  }

  function renderDetail(run) {
    $("detailCard").classList.remove("hidden");
    $("detailBadge").textContent = `${modeText(run.mode)} · ${statusText(run.status)}`;
    $("detailBadge").className = "badge " + (run.status === "completed" ? "ok" : run.status === "failed" ? "bad" : "");
    const s = run.summary || {};
    $("detailMeta").innerHTML = [
      ["运行 ID", run.id], ["开始", fmtTime(run.created_at)], ["结束", fmtTime(run.finished_at)],
      ["判分", `${run.judge?.provider}/${run.judge?.model}`],
      ["被测", `${run.evaluated?.provider}/${run.evaluated?.model}`],
      ["用例", `${s.cases ?? 0} 条（Ragas 计分 ${s.ragas_scored ?? 0}）`],
      ["用户", runUsers(run)],
    ].map(([k, v]) => `<div><strong>${esc(k)}</strong>：${esc(v ?? "—")}</div>`).join("");

    const errors = (run.metric_errors || []).filter(Boolean);
    $("detailWarnings").classList.toggle("hidden", errors.length === 0);
    $("detailWarnings").textContent = errors.length ? "判分提示：\n" + errors.join("\n") : "";
    $("detailError").classList.toggle("hidden", !run.error);
    $("detailError").textContent = run.error || "";

    const cases = run.cases || [];
    $("caseBody").innerHTML = cases.map((c) => {
      const tools = (c.actual_tools || []).join(", ") || "—";
      const result = c.error ? `<span class="status-failed">失败</span>` :
        (c.tool_selection_correct === null || c.tool_selection_correct === undefined ? "—" :
          (c.tool_selection_correct ? '<span class="status-completed">✓</span>' : '<span class="status-failed">×</span>'));
      return `<tr>
        <td class="num">${esc(c.index + 1)}</td>
        <td>${esc(c.username || (run.mode === "offline" ? "离线" : "—"))}</td>
        <td>${esc(c.question)}</td>
        <td>${esc(c.expected_tool || (c.expected_no_tool ? "不调用工具" : "—"))}</td>
        <td>${esc(tools)}</td>
        <td class="num">${fmtLatency(c.latency_ms)}</td>
        <td class="num">${fmtScore(c.scores?.faithfulness)}</td>
        <td class="num">${fmtScore(c.scores?.answer_relevancy)}</td>
        <td class="num">${fmtScore(c.scores?.context_recall)}</td>
        <td>${result}</td>
      </tr>`;
    }).join("") || '<tr><td colspan="10" class="muted">没有用例</td></tr>';

    $("caseDetails").innerHTML = cases.map((c) => {
      const parts = [];
      if (c.reference) parts.push(`<details><summary>参考回答</summary><pre>${esc(c.reference)}</pre></details>`);
      if (c.answer) parts.push(`<details><summary>回答</summary><pre>${esc(c.answer)}</pre></details>`);
      if (c.judge_reasoning) parts.push(`<details><summary>判分理由</summary><pre>${esc(c.judge_reasoning)}</pre></details>`);
      if (c.error) parts.push(`<div class="errorbox">${esc(c.error)}</div>`);
      return `<div style="margin-top:8px"><span class="muted">#${esc(c.index + 1)} ${esc(c.username || "")}</span> ${esc(c.question)}
        ${parts.join("")}</div>`;
    }).join("");
  }

  function renderProgress(run) {
    const progress = run.progress || {};
    const total = progress.total || 0;
    const completed = progress.completed || 0;
    $("progressBox").classList.remove("hidden");
    $("progressText").textContent = `阶段：${progress.phase || "运行中"} · ${completed}/${total || "?"} 条`;
    const pct = total ? Math.min(100, Math.round(completed / total * 100)) : 5;
    $("progressBar").style.width = pct + "%";
    $("progressPct").textContent = total ? pct + "%" : "";
    $("offlineBtn").disabled = true;
    $("onlineBtn").disabled = true;
  }

  function startPolling(runId) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      try {
        const run = await api("/api/evaluate/runs/" + encodeURIComponent(runId) + userQuery());
        currentRun = run;
        renderMetrics(run);
        renderDetail(run);
        if (run.status === "running") {
          renderProgress(run);
        } else {
          clearInterval(pollTimer); pollTimer = null;
          $("progressBox").classList.add("hidden");
          $("offlineBtn").disabled = false;
          $("onlineBtn").disabled = false;
          await loadUsers();
          await loadRuns();
        }
      } catch (err) {
        clearInterval(pollTimer); pollTimer = null;
        showRunError(err.message);
        $("offlineBtn").disabled = false;
        $("onlineBtn").disabled = false;
      }
    }, 1800);
  }

  async function startRun(mode) {
    $("runError").classList.add("hidden");
    const path = mode === "offline" ? "/api/evaluate/offline" : "/api/evaluate/online";
    try {
      const run = await api(path, { method: "POST" });
      currentRun = null;
      $("progressBox").classList.remove("hidden");
      renderProgress(run);
      startPolling(run.id);
    } catch (err) {
      showRunError(err.message);
    }
  }

  function applyUserFilter() {
    filterUser = $("userFilter").value.trim();
    currentRun = null;
    $("detailCard").classList.add("hidden");
    loadRuns();
  }

  $("offlineBtn").addEventListener("click", () => startRun("offline"));
  $("onlineBtn").addEventListener("click", () => startRun("online"));
  $("applyFilter").addEventListener("click", applyUserFilter);
  $("clearFilter").addEventListener("click", () => {
    $("userFilter").value = "";
    filterUser = "";
    currentRun = null;
    $("detailCard").classList.add("hidden");
    loadRuns();
  });
  $("userFilter").addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); applyUserFilter(); }
  });
  $("closeDetail").addEventListener("click", () => $("detailCard").classList.add("hidden"));

  (async () => {
    await loadConfig();
    await loadUsers();
    await loadRuns();
  })();
})();
</script>
</body>
</html>
"""
