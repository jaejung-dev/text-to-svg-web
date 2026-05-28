const ORDER = ["gt", "text-to-svg-base", "text-to-svg-v1", "text-to-svg-v2", "claude", "gemini", "gpt-5.2"];
const MODEL_COMPARISON_ORDER = ["text-to-svg-base", "text-to-svg-v1", "text-to-svg-v2"];
const VALIDATION_WINNER_ORDER = ["text-to-svg-base", "text-to-svg-v1", "text-to-svg-v2", "claude", "gemini", "gpt-5.2"];

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
    .map((key) => {
      const isWinner = item?.metric_winners?.[key]?.source === item.source;
      return `
      <div class="metric-score ${isWinner ? "winner" : ""}">
        <span>${escapeHtml(REPORT_SCORE_LABELS[key] || key)}</span>
        <strong>${escapeHtml(formatScore(scores[key]))}${isWinner ? " ★" : ""}</strong>
      </div>
    `;
    });
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

function sourceMap(payload) {
  const bySource = Object.fromEntries((payload.baselines || []).map((item) => [item.source, item]));
  bySource["text-to-svg-production"] = payload.generated || {};
  Object.entries(payload.model_generations || {}).forEach(([source, item]) => {
    bySource[source] = item || {};
  });
  return bySource;
}

function metricWinners(payload, sources) {
  const bySource = sourceMap(payload);
  const winners = {};
  REPORT_SCORE_ORDER.forEach((metric) => {
    let best = null;
    sources.forEach((source) => {
      const item = bySource[source];
      const value = item?.report_scores?.[metric];
      const score = Number(value);
      if (!item || !Number.isFinite(score)) return;
      if (itemStatus(item) !== "ok" && !item.asset) return;
      if (!best || score > best.score) {
        best = { source, score };
      }
    });
    if (best) winners[metric] = best;
  });
  return winners;
}

function licaScoreFor(item, payload, source) {
  const reportValue = item?.report_scores?.qwen8b_epoch_3;
  if (reportValue != null) return reportValue;
  return payload?.lica_scores?.[source]?.score;
}

function sampleItems(sample) {
  const bySource = sourceMap(sample);
  const winners = metricWinners(sample, VALIDATION_WINNER_ORDER);
  return ORDER.map((source) => {
    const item = {
      source,
      label: LABELS[source] || source,
      ...bySource[source],
      metric_winners: winners,
    };
    item.lica_score = licaScoreFor(item, sample, source);
    item.is_lica_winner = winners.qwen8b_epoch_3?.source === source;
    return item;
  }).filter((item) => isGeneratedSource(item.source) || item.asset);
}

function summarizeMetricWinners(items, sources) {
  const summary = {};
  REPORT_SCORE_ORDER.forEach((metric) => {
    summary[metric] = { total: 0, counts: Object.fromEntries(sources.map((source) => [source, 0])) };
  });
  items.forEach((payload) => {
    const winners = metricWinners(payload, sources);
    REPORT_SCORE_ORDER.forEach((metric) => {
      const winner = winners[metric]?.source;
      if (!winner) return;
      summary[metric].total += 1;
      summary[metric].counts[winner] = (summary[metric].counts[winner] || 0) + 1;
    });
  });
  return summary;
}

function topWinner(summary, sources) {
  const ranked = sources
    .map((source) => ({ source, count: summary.counts[source] || 0 }))
    .sort((a, b) => b.count - a.count);
  const top = ranked[0];
  if (!top || top.count === 0) return "No scores";
  return `${LABELS[top.source] || top.source} ${top.count}/${summary.total}`;
}

function breakdown(summary, sources) {
  return sources
    .filter((source) => summary.counts[source])
    .map((source) => `${LABELS[source] || source}: ${summary.counts[source]}`)
    .join(" · ") || "No wins";
}

function renderScoreSummary(data) {
  const validation = summarizeMetricWinners(data.samples || [], VALIDATION_WINNER_ORDER);
  const promptPairs = summarizeMetricWinners(data.prompt_pairs || [], MODEL_COMPARISON_ORDER);
  const rows = REPORT_SCORE_ORDER.map((metric) => `
    <tr>
      <th>${escapeHtml(REPORT_SCORE_LABELS[metric] || metric)}</th>
      <td><strong>${escapeHtml(topWinner(validation[metric], VALIDATION_WINNER_ORDER))}</strong><span>${escapeHtml(breakdown(validation[metric], VALIDATION_WINNER_ORDER))}</span></td>
      <td><strong>${escapeHtml(topWinner(promptPairs[metric], MODEL_COMPARISON_ORDER))}</strong><span>${escapeHtml(breakdown(promptPairs[metric], MODEL_COMPARISON_ORDER))}</span></td>
    </tr>
  `);
  return `
    <div class="score-summary-head">
      <div>
        <p class="eyebrow">Score Winners</p>
        <h2>Best scores are counted without GT.</h2>
      </div>
      <p>Validation excludes Ground Truth from winner counts. Prompt pairs compare Base, V1, and V2 only. A star marks the metric winner inside each card.</p>
    </div>
    <div class="score-summary-scroll">
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Validation gallery</th>
            <th>Prompt pairs</th>
          </tr>
        </thead>
        <tbody>${rows.join("")}</tbody>
      </table>
    </div>
  `;
}

function renderSample(sample, index) {
  const items = sampleItems(sample);
  const licaWinner = metricWinners(sample, VALIDATION_WINNER_ORDER).qwen8b_epoch_3?.source;
  const generatedMeta = [
    licaWinner ? `Lica winner: ${LABELS[licaWinner] || licaWinner}` : "Lica pending",
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
  const winners = metricWinners(pair, MODEL_COMPARISON_ORDER);

  function modelCard(result, title) {
    const source = result?.source;
    const enriched = {
      ...result,
      metric_winners: winners,
    };
    enriched.lica_score = licaScoreFor(enriched, pair, source);
    enriched.is_lica_winner = winners.qwen8b_epoch_3?.source === source;

    return `
      <div class="pair-output-card">
        <div class="pair-image-wrap">${assetElement(enriched)}</div>
        <div class="pair-output-meta">
          <strong>${escapeHtml(title)}</strong>
          ${resultMeta(source, enriched)}
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
            <span>${escapeHtml(winners.qwen8b_epoch_3 ? `Lica winner: ${LABELS[winners.qwen8b_epoch_3.source] || winners.qwen8b_epoch_3.source}` : "Lica pending")}</span>
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
  document.getElementById("score-summary").innerHTML = renderScoreSummary(data);
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
