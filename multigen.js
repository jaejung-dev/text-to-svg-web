const SCORE_LABELS = {
  lica_score_v1: "LicaScore",
  lica_score_v2: "LicaScore v2",
};

let ASSET_CACHE_KEY = "";

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
  return `${asset}${sep}v=${encodeURIComponent(ASSET_CACHE_KEY)}`;
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "pending";
  return Number(value).toFixed(3);
}

function winnerSeed(prompt, scoreId) {
  return prompt.winners?.[scoreId]?.seed;
}

function renderSummary(data) {
  const prompts = data.prompts || [];
  const agreements = prompts.filter((prompt) => prompt.winner_agreement).length;
  const v1Wins = {};
  const v2Wins = {};
  for (const prompt of prompts) {
    const v1 = prompt.winners?.lica_score_v1?.seed;
    const v2 = prompt.winners?.lica_score_v2?.seed;
    if (v1) v1Wins[v1] = (v1Wins[v1] || 0) + 1;
    if (v2) v2Wins[v2] = (v2Wins[v2] || 0) + 1;
  }
  const top = (rows) => Object.entries(rows).sort((a, b) => b[1] - a[1])[0]?.join(" · ") || "none";
  return `
    <div class="summary-card"><span>Prompts</span><strong>${escapeHtml(prompts.length)}</strong></div>
    <div class="summary-card"><span>Candidates</span><strong>${escapeHtml(data.summary?.candidates || 0)}</strong></div>
    <div class="summary-card"><span>Scorer Agreement</span><strong>${escapeHtml(`${agreements}/${prompts.length}`)}</strong></div>
    <div class="summary-card"><span>Top Seeds</span><strong>${escapeHtml(`v1 ${top(v1Wins)} / v2 ${top(v2Wins)}`)}</strong></div>
  `;
}

function renderWinnerBox(prompt) {
  const v1 = prompt.winners?.lica_score_v1;
  const v2 = prompt.winners?.lica_score_v2;
  const agree = prompt.winner_agreement;
  return `
    <aside class="winner-box">
      <div class="winner-row"><span>LicaScore selects</span><strong>${escapeHtml(v1 ? `seed ${v1.seed}` : "pending")}</strong></div>
      <div class="winner-row"><span>LicaScore v2 selects</span><strong>${escapeHtml(v2 ? `seed ${v2.seed}` : "pending")}</strong></div>
      <div class="agreement ${agree ? "" : "disagree"}">${escapeHtml(agree ? "Scorers agree" : "Scorers disagree")}</div>
    </aside>
  `;
}

function renderCandidate(prompt, candidate) {
  const v1Winner = winnerSeed(prompt, "lica_score_v1") === candidate.seed;
  const v2Winner = winnerSeed(prompt, "lica_score_v2") === candidate.seed;
  const scores = candidate.scores || {};
  const replacement = candidate.replacement_for ? `replacement for ${candidate.replacement_for}` : "";
  const image = candidate.asset
    ? `<img src="${escapeHtml(assetUrl(candidate.asset))}" alt="${escapeHtml(`${prompt.id} seed ${candidate.seed}`)}" loading="lazy" />`
    : `<span class="status-bad">${escapeHtml(candidate.status || "missing")}</span>`;
  return `
    <article class="candidate-card">
      <div class="image-wrap">${image}</div>
      <div class="candidate-meta">
        <div class="seed-line"><strong>Seed ${escapeHtml(candidate.seed)}</strong><span>${escapeHtml(candidate.svg_chars ? `${candidate.svg_chars} chars` : candidate.status)}</span></div>
        ${replacement ? `<div class="replacement-note">${escapeHtml(replacement)}</div>` : ""}
        <div class="score-grid">
          <div class="score-line ${v1Winner ? "winner" : ""}">
            <span>${escapeHtml(SCORE_LABELS.lica_score_v1)}</span>
            <strong>${escapeHtml(formatScore(scores.lica_score_v1))}</strong>
          </div>
          <div class="score-line ${v2Winner ? "winner" : ""}">
            <span>${escapeHtml(SCORE_LABELS.lica_score_v2)}</span>
            <strong>${escapeHtml(formatScore(scores.lica_score_v2))}</strong>
          </div>
        </div>
      </div>
    </article>
  `;
}

function renderPrompt(prompt) {
  const trainSim = prompt.train_sim === null || prompt.train_sim === undefined ? "n/a" : prompt.train_sim;
  const nearest = prompt.nearest_curated_id ? `<span class="tag">nearest ${escapeHtml(prompt.nearest_curated_id)}</span>` : "";
  return `
    <article class="prompt-card">
      <header class="prompt-head">
        <div>
          <div class="prompt-meta">
            <span class="tag">${escapeHtml(prompt.id)}</span>
            <span class="tag">${escapeHtml(prompt.lang)}</span>
            <span class="tag">${escapeHtml(prompt.length_tier)}</span>
            <span class="tag">train sim ${escapeHtml(trainSim)}</span>
            ${nearest}
          </div>
          <p class="prompt-text">${escapeHtml(prompt.prompt)}</p>
        </div>
        ${renderWinnerBox(prompt)}
      </header>
      <div class="candidate-grid">
        ${(prompt.candidates || []).map((candidate) => renderCandidate(prompt, candidate)).join("")}
      </div>
    </article>
  `;
}

async function main() {
  const response = await fetch("data/multigen-demo.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load data: ${response.status}`);
  const data = await response.json();
  ASSET_CACHE_KEY = data.generated_at || String(Date.now());
  document.getElementById("summary").innerHTML = renderSummary(data);
  document.getElementById("prompts").innerHTML = (data.prompts || []).map(renderPrompt).join("");
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="padding:24px;white-space:pre-wrap">${escapeHtml(err.stack || err)}</pre>`;
});
