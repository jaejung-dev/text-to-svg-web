let CACHE_KEY = "";

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
}

function assetUrl(asset) {
  const sep = asset.includes("?") ? "&" : "?";
  return `${asset}${sep}v=${encodeURIComponent(CACHE_KEY)}`;
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "pending";
  const abs = Math.abs(Number(value));
  if (abs >= 10) return Number(value).toFixed(2);
  return Number(value).toFixed(4);
}

function renderSummary(data) {
  const summary = data.summary || {};
  const metrics = data.score_order?.length || 1;
  return `
    <article><span>Prompts</span><strong>${escapeHtml(summary.prompts || 0)}</strong></article>
    <article><span>Scored SVGs</span><strong>${escapeHtml(summary.scored_candidates || 0)}</strong></article>
    <article><span>Model tracks</span><strong>${escapeHtml(summary.models || 0)}</strong></article>
    <article><span>Metrics</span><strong>${escapeHtml(metrics)}</strong></article>
  `;
}

function renderRankings(data) {
  const labels = data.score_labels || {};
  const scoreOrder = data.score_order || ["lica_score_v2"];
  const rows = (data.models || [])
    .map((model) => {
      const metricWins = scoreOrder.map((metric) => `
        <div class="rank-metric">
          <span>${escapeHtml(labels[metric] || metric)}</span>
          <strong>${escapeHtml(model.metric_wins?.[metric] || 0)}</strong>
        </div>
      `).join("");
      return `
        <article class="rank-card">
          <span>${escapeHtml(model.label)}</span>
          <strong>${escapeHtml(model.wins || 0)} LicaScore v2 wins</strong>
          <small>${escapeHtml(model.count || 0)} generated SVGs</small>
          <div class="rank-metrics">${metricWins}</div>
        </article>
      `;
    })
    .join("");
  return `
    <div class="section-head">
      <p class="eyebrow">Model Win Count</p>
      <h2>Winner selections by metric</h2>
    </div>
    <div class="rank-grid">${rows}</div>
  `;
}

function renderPromptText(prompt) {
  const translation = prompt.prompt_en
    ? `<p class="prompt-translation"><span>English</span>${escapeHtml(prompt.prompt_en)}</p>`
    : "";
  return `
    <div class="prompt-copy">
      <div class="prompt-id">${escapeHtml(prompt.id.replaceAll("_", " "))}</div>
      <h2>${escapeHtml(prompt.prompt)}</h2>
      ${translation}
    </div>
  `;
}

function renderMetricRows(prompt, candidate) {
  const labels = window.MODEL_COMPARISON_SCORE_LABELS || {};
  const scoreOrder = window.MODEL_COMPARISON_SCORE_ORDER || ["lica_score_v2"];
  const scores = candidate.scores || {};
  return scoreOrder.map((metric) => {
    const isWinner = prompt.winners?.[metric]?.id === candidate.id;
    return `
      <div class="metric-row ${isWinner ? "winner" : ""}">
        <span>${escapeHtml(labels[metric] || metric)}</span>
        <strong>${escapeHtml(formatScore(scores[metric]))}</strong>
      </div>
    `;
  }).join("");
}

function renderCandidate(prompt, candidate) {
  const isWinner = prompt.best?.id === candidate.id;
  const image = candidate.asset
    ? `<img src="${escapeHtml(assetUrl(candidate.asset))}" alt="${escapeHtml(`${candidate.label} output for ${prompt.id}`)}" loading="lazy" />`
    : `<div class="missing-box">Not provided</div>`;
  return `
    <article class="candidate-card ${isWinner ? "winner" : ""} ${candidate.asset ? "" : "missing"}">
      <div class="image-frame">${image}</div>
      <div class="candidate-meta">
        <div>
          <h3>${escapeHtml(candidate.label)}</h3>
          <p>${escapeHtml(candidate.asset || "No Arrow SVG available for this prompt")}</p>
        </div>
        <div class="score-pill ${isWinner ? "winner" : ""}">
          <span>${escapeHtml(isWinner ? "LicaScore v2 winner" : "LicaScore v2")}</span>
          <strong>${escapeHtml(formatScore(candidate.scores?.lica_score_v2 ?? candidate.score))}</strong>
        </div>
        <div class="metric-list">${renderMetricRows(prompt, candidate)}</div>
      </div>
    </article>
  `;
}

function renderWinnerMetrics(prompt) {
  const labels = window.MODEL_COMPARISON_SCORE_LABELS || {};
  const scoreOrder = window.MODEL_COMPARISON_SCORE_ORDER || ["lica_score_v2"];
  return scoreOrder.map((metric) => {
    const winner = prompt.winners?.[metric];
    return `
      <div class="winner-metric">
        <span>${escapeHtml(labels[metric] || metric)}</span>
        <strong>${escapeHtml(winner?.label || "pending")}</strong>
      </div>
    `;
  }).join("");
}

function renderPrompt(prompt) {
  const best = prompt.best;
  return `
    <article class="prompt-card">
      <header class="prompt-head">
        ${renderPromptText(prompt)}
        <aside class="winner-box">
          <span>Best selection</span>
          <strong>${escapeHtml(best?.label || "pending")}</strong>
          <small>${escapeHtml(formatScore(best?.score))}</small>
          <div class="winner-metrics">${renderWinnerMetrics(prompt)}</div>
        </aside>
      </header>
      <div class="candidate-grid">
        ${(prompt.candidates || []).map((candidate) => renderCandidate(prompt, candidate)).join("")}
      </div>
    </article>
  `;
}

async function main() {
  const response = await fetch("data/model-comparison-demo.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load data: ${response.status}`);
  const data = await response.json();
  CACHE_KEY = data.generated_at || String(Date.now());
  window.MODEL_COMPARISON_SCORE_ORDER = data.score_order || ["lica_score_v2"];
  window.MODEL_COMPARISON_SCORE_LABELS = data.score_labels || { lica_score_v2: "LicaScore v2" };
  document.getElementById("summary").innerHTML = renderSummary(data);
  document.getElementById("model-rankings").innerHTML = renderRankings(data);
  document.getElementById("prompts").innerHTML = (data.prompts || []).map(renderPrompt).join("");
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="padding:24px;white-space:pre-wrap">${escapeHtml(err.stack || err)}</pre>`;
});
