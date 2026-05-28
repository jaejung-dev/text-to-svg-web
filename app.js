const ORDER = ["gt", "text-to-svg-base", "text-to-svg-v1", "text-to-svg-v2", "claude", "gemini", "gpt-5.2"];
const MODEL_COMPARISON_ORDER = ["text-to-svg-base", "text-to-svg-v1", "text-to-svg-v2"];

const LABELS = {
  gt: "Ground Truth",
  "text-to-svg-production": "Text-to-SVG",
  "text-to-svg-base": "Base",
  "text-to-svg-v1": "V1",
  "text-to-svg-v2": "V2",
  claude: "Claude",
  gemini: "Gemini",
  "gpt-5.2": "GPT-5.2",
};

const BADGES = {
  gt: "GT",
  "text-to-svg-production": "Ours",
  "text-to-svg-base": "Base",
  "text-to-svg-v1": "V1",
  "text-to-svg-v2": "V2",
  claude: "Ref",
  gemini: "Ref",
  "gpt-5.2": "Ref",
};

let ASSET_CACHE_KEY = "";
let REPORT_SCORE_ORDER = [
  "qwen8b_epoch_3",
  "imscore_hpsv21",
  "imscore_pickscore",
  "imscore_mpsv1",
  "imscore_clipscore",
  "imscore_imagereward",
  "imscore_laion_aesthetic",
];
let REPORT_SCORE_LABELS = {
  qwen8b_epoch_3: "LicaScore",
  imscore_hpsv21: "HPSv2.1",
  imscore_pickscore: "PickScore",
  imscore_mpsv1: "MPSv1",
  imscore_clipscore: "CLIPScore",
  imscore_imagereward: "ImageReward",
  imscore_laion_aesthetic: "LAION aesthetic",
};

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
  const separator = asset.includes("?") ? "&" : "?";
  return `${asset}${separator}v=${encodeURIComponent(ASSET_CACHE_KEY)}`;
}

function itemStatus(item) {
  if (item?.status) return item.status;
  if (item?.asset) return "ok";
  return "pending";
}

function placeholderText(item) {
  const status = itemStatus(item);
  if (status === "error") return "Baseten unavailable";
  if (status === "no_svg") return "No SVG returned";
  return "Baseten pending";
}

function assetElement(item) {
  if (!item?.asset) {
    return `<div class="empty"><strong>${escapeHtml(placeholderText(item))}</strong><span>Placeholder until generation completes.</span></div>`;
  }
  const asset = escapeHtml(assetUrl(item.asset));
  return `<img src="${asset}" alt="${escapeHtml(item.label)}" loading="lazy" />`;
}

function isGeneratedSource(source) {
  return source === "text-to-svg-production"
    || source === "text-to-svg-base"
    || source === "text-to-svg-v1"
    || source === "text-to-svg-v2";
}

function isPrimaryGeneratedSource(source) {
  return source === "text-to-svg-v2";
}

function badgeClass(source) {
  if (source === "gt") return "gt";
  if (isGeneratedSource(source)) return "generated";
  return "ref";
}

function scoreLine(item) {
  if (item?.lica_score == null) return "";
  const label = `Lica ${Number(item.lica_score).toFixed(4)}${item.is_lica_winner ? " · winner" : ""}`;
  return `<div class="score-pill ${item.is_lica_winner ? "winner" : ""}">${escapeHtml(label)}</div>`;
}

function formatScore(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  return Math.abs(number) < 2 ? number.toFixed(3) : number.toFixed(2);
}

function reportScores(item) {
  const scores = item?.report_scores || {};
  const rows = REPORT_SCORE_ORDER
    .filter((key) => scores[key] != null)
    .map((key) => `
      <div class="metric-score">
        <span>${escapeHtml(REPORT_SCORE_LABELS[key] || key)}</span>
        <strong>${escapeHtml(formatScore(scores[key]))}</strong>
      </div>
    `);
  if (!rows.length) return "";
  return `<div class="report-scores">${rows.join("")}</div>`;
}

function resultMeta(source, item) {
  const parts = [];
  if (isGeneratedSource(source) && item?.output_tokens != null) parts.push(`${item.output_tokens} output tokens`);
  if (isGeneratedSource(source) && item?.elapsed_seconds != null) parts.push(`${item.elapsed_seconds}s`);
  if (item?.svg_parse_error) parts.push(`parse: ${item.svg_parse_error}`);
  if (itemStatus(item) !== "ok" && item?.error) parts.push(item.error);
  if (itemStatus(item) === "pending") parts.push("generation pending");
  const meta = parts.length ? `<div class="result-meta">${escapeHtml(parts.join(" · "))}</div>` : "";
  return `${scoreLine(item)}${meta}${reportScores(item)}`;
}

function sampleItems(sample) {
  const bySource = Object.fromEntries((sample.baselines || []).map((item) => [item.source, item]));
  bySource["text-to-svg-production"] = sample.generated || {};
  Object.entries(sample.model_generations || {}).forEach(([source, item]) => {
    bySource[source] = item || {};
  });
  return ORDER.map((source) => ({
    source,
    label: LABELS[source] || source,
    ...bySource[source],
    lica_score: sample.lica_scores?.[source]?.score,
    is_lica_winner: sample.lica_winner === source,
  })).filter((item) => isGeneratedSource(item.source) || item.asset);
}

function renderSample(sample, index) {
  const items = sampleItems(sample);
  const generatedMeta = [
    sample.lica_winner ? `Lica winner: ${LABELS[sample.lica_winner] || sample.lica_winner}` : "Lica pending",
    ...items
      .filter((item) => isGeneratedSource(item.source))
      .map((item) => `${LABELS[item.source] || item.source}: ${itemStatus(item)}`),
  ].join("<br>");

  return `
    <article class="sample-card" data-bucket="${escapeHtml(sample.bucket)}">
      <div class="sample-head">
        <div>
          <div class="sample-title">
            <div class="sample-index">${index + 1}</div>
            <span class="bucket">${escapeHtml(sample.bucket || "sample")}</span>
          </div>
          <p class="prompt">${escapeHtml(sample.prompt)}</p>
        </div>
        <div class="generation-meta">${generatedMeta}</div>
      </div>
      <div class="comparison-grid">
        ${items.map((item) => `
          <div class="result-card ${isPrimaryGeneratedSource(item.source) ? "featured" : ""}">
            <div class="image-wrap">${assetElement(item)}</div>
            <div class="result-body">
              <div class="result-label">
                <span>${escapeHtml(item.label || LABELS[item.source] || item.source)}</span>
                <span class="badge ${badgeClass(item.source)}">${escapeHtml(BADGES[item.source] || "Ref")}</span>
              </div>
              ${resultMeta(item.source, item)}
            </div>
          </div>
        `).join("")}
      </div>
    </article>
  `;
}

function renderPromptPair(pair) {
  function metaLine(result) {
    if (result?.status !== "ok") {
      return result?.error || result?.svg_parse_error || "Generation pending";
    }
    return [
      result.output_tokens != null ? `${result.output_tokens} tok` : null,
      result.elapsed_seconds != null ? `${result.elapsed_seconds}s` : null,
      result.svg_parse_error ? `parse: ${result.svg_parse_error}` : "valid SVG",
    ].filter(Boolean).join(" · ");
  }

  function modelCard(result, title) {
    const meta = metaLine(result);
    const source = result?.source;
    const score = pair.lica_scores?.[source]?.score;
    const isWinner = pair.lica_winner === source;

    return `
      <div class="pair-output-card">
        <div class="pair-image-wrap">${assetElement(result)}</div>
        <div class="pair-output-meta">
          <strong>${escapeHtml(title)}</strong>
          ${score == null ? "" : `<div class="score-pill ${isWinner ? "winner" : ""}">${escapeHtml(`Lica ${Number(score).toFixed(4)}${isWinner ? " · winner" : ""}`)}</div>`}
          <span>${escapeHtml(meta)}</span>
        </div>
      </div>
    `;
  }

  const modelGenerations = pair.model_generations || {};

  return `
    <article class="prompt-pair-card">
      <div class="prompt-panel">
        <div class="pair-kicker">Prompt ${escapeHtml(pair.index)}</div>
        <p>${escapeHtml(pair.prompt)}</p>
      </div>
      <div class="prompt-pair-results">
        <section class="lora-mode-section">
          <div class="lora-mode-head">
            <h3>Base / V1 / V2</h3>
            <span>${escapeHtml(pair.lica_winner ? `Lica winner: ${LABELS[pair.lica_winner] || pair.lica_winner}` : "Lica pending")}</span>
          </div>
          <div class="pair-outputs-grid model-comparison-grid">
            ${MODEL_COMPARISON_ORDER.map((source) => modelCard(
              modelGenerations[source] || { source, label: LABELS[source], status: "pending" },
              LABELS[source] || source,
            )).join("")}
          </div>
        </section>
      </div>
    </article>
  `;
}

async function main() {
  const data = await fetch(`data/site-data.json?v=${Date.now()}`).then((response) => {
    if (!response.ok) throw new Error(`Failed to load site data: ${response.status}`);
    return response.json();
  });
  ASSET_CACHE_KEY = data.generated_at || String(Date.now());
  REPORT_SCORE_ORDER = data.report_score_order || REPORT_SCORE_ORDER;
  REPORT_SCORE_LABELS = data.report_score_labels || REPORT_SCORE_LABELS;

  const gallery = document.getElementById("gallery");
  const filter = document.getElementById("bucket-filter");
  document.getElementById("sample-count").textContent = data.samples.length;
  document.getElementById("updated-at").textContent = `Generated ${data.generated_at || "unknown"}`;
  document.getElementById("bucket-summary").innerHTML = ["simple", "medium", "complex"].map((bucket) => {
    const count = data.samples.filter((sample) => sample.bucket === bucket).length;
    return `<div class="bucket-chip"><strong>${count}</strong><span>${escapeHtml(bucket)}</span></div>`;
  }).join("");
  document.getElementById("hero-strip").innerHTML = data.samples
    .slice(0, 5)
    .map((sample) => sample.model_generations?.["text-to-svg-v2"]?.asset || sample.generated?.asset)
    .filter(Boolean)
    .map((asset) => `<div class="strip-tile"><img src="${escapeHtml(assetUrl(asset))}" alt="Generated preview" loading="lazy" /></div>`)
    .join("");
  document.getElementById("prompt-pairs-grid").innerHTML = (data.prompt_pairs || [])
    .map(renderPromptPair)
    .join("");

  function draw() {
    const bucket = filter.value;
    const samples = data.samples.filter((sample) => bucket === "all" || sample.bucket === bucket);
    gallery.innerHTML = samples.map(renderSample).join("");
  }

  filter.addEventListener("change", draw);
  draw();
}

main().catch((error) => {
  document.body.innerHTML = `<pre style="padding: 32px; color: #b42318; white-space: pre-wrap">${escapeHtml(error.stack || error)}</pre>`;
  console.error(error);
});
