const ORDER = ["gt", "text-to-svg-production", "claude", "gemini", "gpt-5.2"];

const LABELS = {
  gt: "Ground Truth",
  "text-to-svg-production": "Text-to-SVG",
  claude: "Claude",
  gemini: "Gemini",
  "gpt-5.2": "GPT-5.2",
};

const BADGES = {
  gt: "GT",
  "text-to-svg-production": "Ours",
  claude: "Ref",
  gemini: "Ref",
  "gpt-5.2": "Ref",
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

function assetElement(item) {
  if (!item?.asset) {
    return `<div class="empty">Not generated yet</div>`;
  }
  const asset = escapeHtml(item.asset);
  return `<img src="${asset}" alt="${escapeHtml(item.label)}" loading="lazy" />`;
}

function resultMeta(source, item) {
  if (source !== "text-to-svg-production") return "";
  const parts = [];
  if (item?.output_tokens != null) parts.push(`${item.output_tokens} output tokens`);
  if (item?.elapsed_seconds != null) parts.push(`${item.elapsed_seconds}s`);
  if (item?.svg_parse_error) parts.push(`parse: ${item.svg_parse_error}`);
  return parts.length ? `<div class="result-meta">${escapeHtml(parts.join(" · "))}</div>` : "";
}

function sampleItems(sample) {
  const bySource = Object.fromEntries((sample.baselines || []).map((item) => [item.source, item]));
  bySource["text-to-svg-production"] = sample.generated || {};
  return ORDER.map((source) => ({
    source,
    label: LABELS[source] || source,
    ...bySource[source],
  })).filter((item) => item.source === "text-to-svg-production" || item.asset);
}

function renderSample(sample, index) {
  const items = sampleItems(sample);
  const generated = sample.generated || {};
  const generatedMeta = generated.status === "ok"
    ? [
      generated.input_tokens != null ? `Input ${generated.input_tokens}` : null,
      generated.output_tokens != null ? `Output ${generated.output_tokens}` : null,
      generated.elapsed_seconds != null ? `${generated.elapsed_seconds}s` : null,
    ].filter(Boolean).join("<br>")
    : "Generation pending";

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
          <div class="result-card ${item.source === "text-to-svg-production" ? "featured" : ""}">
            <div class="image-wrap">${assetElement(item)}</div>
            <div class="result-body">
              <div class="result-label">
                <span>${escapeHtml(item.label || LABELS[item.source] || item.source)}</span>
                <span class="badge ${item.source === "gt" ? "gt" : item.source === "text-to-svg-production" ? "generated" : "ref"}">${escapeHtml(BADGES[item.source] || "Ref")}</span>
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
  const generated = pair.generated || {};
  const meta = generated.status === "ok"
    ? [
      generated.output_tokens != null ? `${generated.output_tokens} output tokens` : null,
      generated.elapsed_seconds != null ? `${generated.elapsed_seconds}s` : null,
      generated.svg_parse_error ? `parse: ${generated.svg_parse_error}` : "valid SVG",
    ].filter(Boolean).join(" · ")
    : "Generation pending";

  return `
    <article class="prompt-pair-card">
      <div class="prompt-panel">
        <div class="pair-kicker">Prompt ${escapeHtml(pair.index)}</div>
        <p>${escapeHtml(pair.prompt)}</p>
      </div>
      <div class="pair-output-card">
        <div class="pair-image-wrap">${assetElement(generated)}</div>
        <div class="pair-output-meta">
          <strong>Text-to-SVG Production</strong>
          <span>${escapeHtml(meta)}</span>
        </div>
      </div>
    </article>
  `;
}

async function main() {
  const data = await fetch(`data/site-data.json?v=${Date.now()}`).then((response) => {
    if (!response.ok) throw new Error(`Failed to load site data: ${response.status}`);
    return response.json();
  });

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
    .map((sample) => sample.generated?.asset)
    .filter(Boolean)
    .map((asset) => `<div class="strip-tile"><img src="${escapeHtml(asset)}" alt="Generated preview" loading="lazy" /></div>`)
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
