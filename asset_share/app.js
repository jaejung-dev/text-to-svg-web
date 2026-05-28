let cacheKey = "";

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
  return `${asset}${sep}v=${encodeURIComponent(cacheKey)}`;
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "pending";
  const numeric = Number(value);
  if (Math.abs(numeric) >= 10) return numeric.toFixed(2);
  return numeric.toFixed(4);
}

function renderSummary(data) {
  const summary = data.summary || {};
  return `
    <article><span>Prompts</span><strong>${escapeHtml(summary.prompts || 0)}</strong></article>
    <article><span>SVGs</span><strong>${escapeHtml(summary.candidates || 0)}</strong></article>
    <article><span>Metrics</span><strong>${escapeHtml(summary.metrics || 0)}</strong></article>
  `;
}

function renderMetricRows(prompt, candidate, data) {
  return (data.score_order || []).map((metric) => {
    const isWinner = prompt.winners?.[metric]?.id === candidate.id;
    return `
      <div class="metric-row ${isWinner ? "winner" : ""}">
        <span>${escapeHtml(data.score_labels?.[metric] || metric)}</span>
        <strong>${escapeHtml(formatScore(candidate.scores?.[metric]))}</strong>
      </div>
    `;
  }).join("");
}

function renderCandidate(prompt, candidate, data) {
  const isWinner = prompt.best?.id === candidate.id;
  return `
    <article class="candidate-card ${isWinner ? "winner" : ""}">
      <div class="image-frame">
        <img src="${escapeHtml(assetUrl(candidate.asset))}" alt="${escapeHtml(candidate.display_label)}" loading="lazy" />
      </div>
      <div class="candidate-meta">
        <div class="candidate-name">${escapeHtml(candidate.display_label)}</div>
        <div class="score-pill ${isWinner ? "winner" : ""}">
          <span>${escapeHtml(isWinner ? "LicaScore winner" : "LicaScore")}</span>
          <strong>${escapeHtml(formatScore(candidate.scores?.lica_score_v2))}</strong>
        </div>
        <div class="metric-list">${renderMetricRows(prompt, candidate, data)}</div>
      </div>
    </article>
  `;
}

function renderPrompt(prompt, data) {
  return `
    <article class="prompt-card">
      <header class="prompt-head">
        <div class="prompt-copy">
          <div class="prompt-id">${escapeHtml(prompt.id.replaceAll("_", " "))}</div>
          <h2>${escapeHtml(prompt.prompt)}</h2>
        </div>
      </header>
      <div class="candidate-grid">
        ${(prompt.candidates || []).map((candidate) => renderCandidate(prompt, candidate, data)).join("")}
      </div>
    </article>
  `;
}

async function main() {
  const response = await fetch("data.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load data: ${response.status}`);
  const data = await response.json();
  cacheKey = JSON.stringify(data.summary || {});
  document.getElementById("summary").innerHTML = renderSummary(data);
  document.getElementById("prompts").innerHTML = (data.prompts || [])
    .map((prompt) => renderPrompt(prompt, data))
    .join("");
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="padding:24px;white-space:pre-wrap">${escapeHtml(err.stack || err)}</pre>`;
});
