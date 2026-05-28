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
  return Number(value).toFixed(4);
}

function renderSummary(data) {
  const summary = data.summary || {};
  return `
    <article><span>Prompts</span><strong>${escapeHtml(summary.prompts || 0)}</strong></article>
    <article><span>Scored SVGs</span><strong>${escapeHtml(summary.scored_candidates || 0)}</strong></article>
    <article><span>Model tracks</span><strong>${escapeHtml(summary.models || 0)}</strong></article>
    <article><span>Score</span><strong>LicaScore v2</strong></article>
  `;
}

function renderRankings(data) {
  const rows = (data.models || [])
    .map((model) => `
      <article class="rank-card">
        <span>${escapeHtml(model.label)}</span>
        <strong>${escapeHtml(model.wins || 0)} wins</strong>
        <small>${escapeHtml(model.count || 0)} generated SVGs</small>
      </article>
    `)
    .join("");
  return `
    <div class="section-head">
      <p class="eyebrow">Model Win Count</p>
      <h2>Best-by-score selections across all prompts</h2>
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
          <span>${escapeHtml(isWinner ? "Winner" : "Score")}</span>
          <strong>${escapeHtml(formatScore(candidate.score))}</strong>
        </div>
      </div>
    </article>
  `;
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
  document.getElementById("summary").innerHTML = renderSummary(data);
  document.getElementById("model-rankings").innerHTML = renderRankings(data);
  document.getElementById("prompts").innerHTML = (data.prompts || []).map(renderPrompt).join("");
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="padding:24px;white-space:pre-wrap">${escapeHtml(err.stack || err)}</pre>`;
});
