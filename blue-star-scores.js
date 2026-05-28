const SCORE_LABELS = {
  lica_score_v1: "LicaScore",
  lica_score_v2: "LicaScore v2",
  hpsv21: "HPS v2.1",
  pickscore: "PickScore",
  clipscore: "CLIPScore",
  imagereward: "ImageReward",
  laion_aesthetic: "LAION aesthetic",
};

const SCORE_ORDER = [
  "lica_score_v1",
  "lica_score_v2",
  "hpsv21",
  "pickscore",
  "clipscore",
  "imagereward",
  "laion_aesthetic",
];

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

function winnerFor(data, scoreId) {
  return data.winners?.[scoreId]?.id;
}

function renderSummary(data) {
  const v1 = data.winners?.lica_score_v1;
  const v2 = data.winners?.lica_score_v2;
  return `
    <div class="summary-card"><span>Prompt</span><strong>${escapeHtml(data.prompt)}</strong></div>
    <div class="summary-card"><span>Candidates</span><strong>${escapeHtml(data.candidates?.length || 0)}</strong></div>
    <div class="summary-card"><span>V1 winner</span><strong>${escapeHtml(v1?.label || "pending")}</strong></div>
    <div class="summary-card"><span>V2 winner</span><strong>${escapeHtml(v2?.label || "pending")}</strong></div>
  `;
}

function renderCandidate(data, candidate) {
  const scores = candidate.scores || {};
  const rows = SCORE_ORDER.map((scoreId) => {
    const isWinner = winnerFor(data, scoreId) === candidate.id;
    const primary = scoreId === "lica_score_v1" || scoreId === "lica_score_v2";
    return `
      <div class="score-row ${isWinner ? "winner" : ""} ${primary ? "primary" : ""}">
        <span>${escapeHtml(SCORE_LABELS[scoreId] || scoreId)}</span>
        <strong>${escapeHtml(formatScore(scores[scoreId]))}</strong>
      </div>
    `;
  }).join("");
  return `
    <article class="candidate-card">
      <div class="image-wrap">
        <img src="${escapeHtml(assetUrl(candidate.asset))}" alt="${escapeHtml(candidate.label)}" />
      </div>
      <div class="candidate-body">
        <div class="candidate-title">
          <h2>${escapeHtml(candidate.label)}</h2>
          <code>${escapeHtml(candidate.filename)}</code>
        </div>
        <div class="score-list">${rows}</div>
      </div>
    </article>
  `;
}

async function main() {
  const response = await fetch("data/blue-star-scores.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load data: ${response.status}`);
  const data = await response.json();
  CACHE_KEY = data.generated_at || String(Date.now());
  document.getElementById("summary").innerHTML = renderSummary(data);
  document.getElementById("candidates").innerHTML = (data.candidates || [])
    .map((candidate) => renderCandidate(data, candidate))
    .join("");
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="padding:24px;white-space:pre-wrap">${escapeHtml(err.stack || err)}</pre>`;
});
