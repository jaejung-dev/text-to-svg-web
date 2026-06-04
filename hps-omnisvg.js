function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function pct(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function renderHeadline(d) {
  const h = d.headline;
  const cards = [
    { v: pct(h.gt_preferred_rate), s: "Ground-truth preferred", note: "over the generated SVG" },
    { v: fmt(h.hps_vs_bon_reward_pearson, 2), s: "Corr. with V19 reward", note: "weak → measures something else" },
    { v: `${fmt(h.score_range[0], 2)}–${fmt(h.score_range[1], 2)}`, s: "Full score range", note: "narrow band, low spread" },
    { v: `+${fmt(h.median_margin, 3)}`, s: "Median GT−gen margin", note: "small separation" },
    { v: h.n, s: "Prompts scored", note: "GT + generated each" },
  ];
  return cards.map((c) => `
    <div class="stat">
      <strong>${escapeHtml(c.v)}</strong>
      <span>${escapeHtml(c.s)}</span>
      <small>${escapeHtml(c.note)}</small>
    </div>
  `).join("");
}

function renderTakeaways() {
  const items = [
    { warn: false, h: "Direction is correct", p: "Ranks the human ground-truth above the generated SVG 96.4% of the time, so it is not broken." },
    { warn: true, h: "Coarse, not fine-grained", p: "All scores sit in ~0.06–0.32 with a ~0.06 median margin — weak for separating close Best-of-N candidates." },
    { warn: true, h: "Aesthetic over fidelity", p: "Its misses reward clean, bright, decorated vector art over prompt-faithfulness." },
    { warn: true, h: "Saturation / brightness bias", p: "Muted, pastel and beige palettes get systematically low absolute scores." },
    { warn: true, h: "Out-of-distribution prompts", p: "Trained on short captions; OmniSVG prompts are nearly all long (>50 words)." },
    { warn: false, h: "Complementary signal", p: "Only 0.22 correlation with the existing reward — potentially useful as an extra, not a replacement." },
  ];
  return items.map((i) => `
    <div class="takeaway ${i.warn ? "warn" : ""}">
      <h3>${escapeHtml(i.h)}</h3>
      <p>${escapeHtml(i.p)}</p>
    </div>
  `).join("");
}

function renderHist(bins, cls) {
  const max = Math.max(1, ...bins.map((b) => b.count));
  const bars = bins.map((b) => {
    const neg = cls === "margin" && b.x1 <= 0;
    const h = (b.count / max) * 100;
    return `<div class="bar ${neg ? "neg" : ""}" style="height:${h}%" title="${b.x0}–${b.x1}: ${b.count}"></div>`;
  }).join("");
  const lo = bins[0].x0;
  const hi = bins[bins.length - 1].x1;
  return `
    <div class="hist ${cls}">${bars}</div>
    <div class="axis"><span>${fmt(lo, 2)}</span><span>${fmt(hi, 2)}</span></div>
  `;
}

function renderCharts(d) {
  return `
    <div class="chart-card">
      <h3>Ground-truth scores</h3>
      <p class="sub">mean ${fmt(d.dist.gt.mean)} · median ${fmt(d.dist.gt.median)}</p>
      ${renderHist(d.histograms.gt, "gt")}
    </div>
    <div class="chart-card">
      <h3>Generated scores</h3>
      <p class="sub">mean ${fmt(d.dist.generated.mean)} · median ${fmt(d.dist.generated.median)}</p>
      ${renderHist(d.histograms.generated, "gen")}
    </div>
    <div class="chart-card">
      <h3>Margin (GT − generated)</h3>
      <p class="sub">red = model prefers generated (a miss) · ${pct(d.headline.gen_preferred_rate)} of cases</p>
      ${renderHist(d.histograms.margin, "margin")}
    </div>
  `;
}

function sliceTable(title, obj, colKey) {
  const rows = Object.entries(obj).map(([k, v]) => `
    <tr>
      <td>${escapeHtml(k)}</td>
      <td>${escapeHtml(v.n)}</td>
      <td>${fmt(v.gt_mean)}</td>
      <td>${fmt(v.gen_mean)}</td>
      <td>${pct(v.gt_preferred_rate)}</td>
    </tr>
  `).join("");
  return `
    <div class="slice-card">
      <h3>${escapeHtml(title)}</h3>
      <table>
        <thead>
          <tr><th>${escapeHtml(colKey)}</th><th>n</th><th>GT</th><th>gen</th><th>GT&nbsp;pref</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function renderCase(c) {
  const gtWin = c.margin > 0;
  return `
    <article class="case ${c.tags.includes("miss") ? "miss" : ""}" data-tags="${escapeHtml(c.tags.join(" "))}">
      <div class="case-imgs">
        <figure class="${gtWin ? "win" : ""}">
          <figcaption>GT</figcaption>
          <img src="${escapeHtml(c.gt_thumb)}" alt="ground truth ${escapeHtml(c.id)}" loading="lazy" />
        </figure>
        <figure class="${gtWin ? "" : "win"}">
          <figcaption>GEN</figcaption>
          <img src="${escapeHtml(c.gen_thumb)}" alt="generated ${escapeHtml(c.id)}" loading="lazy" />
        </figure>
      </div>
      <div class="case-body">
        <div class="score-row">
          <span class="pill gt">GT ${fmt(c.gt_score)}</span>
          <span class="pill gen">GEN ${fmt(c.gen_score)}</span>
          <span class="pill margin ${gtWin ? "pos" : "neg"}">${gtWin ? "+" : ""}${fmt(c.margin)}</span>
        </div>
        <p class="case-prompt">${escapeHtml(c.prompt)}</p>
        <div class="tags">
          ${c.tags.map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join("")}
          <span class="tag">${escapeHtml(c.color_words)} color words</span>
        </div>
      </div>
    </article>
  `;
}

function renderDownloads(raw) {
  return `
    <a class="dl" href="${escapeHtml(raw.csv)}" download>
      <strong>scores.csv</strong>
      <span>${escapeHtml(raw.rows)} rows · spreadsheet-ready</span>
    </a>
    <a class="dl" href="${escapeHtml(raw.jsonl)}" download>
      <strong>scores.jsonl</strong>
      <span>${escapeHtml(raw.rows)} rows · one JSON object per line</span>
    </a>
    <div class="dl meta">
      <strong>join key</strong>
      <span>${escapeHtml(raw.join_key)}</span>
    </div>
    <div class="dl meta">
      <strong>columns</strong>
      <span>${escapeHtml(raw.columns.join(", "))}</span>
    </div>
  `;
}

async function renderScoresPreview(raw) {
  const body = document.querySelector("#scores-preview tbody");
  const note = document.getElementById("scores-preview-note");
  let rows = [];
  try {
    const res = await fetch(raw.jsonl, { cache: "no-store" });
    const text = await res.text();
    rows = text.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
  } catch (err) {
    note.textContent = "Preview unavailable — use the download links above.";
    return;
  }
  const PREVIEW = 25;
  const draw = (limit) => {
    body.innerHTML = rows.slice(0, limit).map((r) => {
      const pos = r.margin_gt_minus_gen > 0;
      return `
        <tr>
          <td class="mono">${escapeHtml(r.id)}</td>
          <td>${fmt(r.gt_score)}</td>
          <td>${fmt(r.gen_score)}</td>
          <td class="${pos ? "pos" : "neg"}">${pos ? "+" : ""}${fmt(r.margin_gt_minus_gen)}</td>
          <td>${fmt(r.bon_reward)}</td>
        </tr>
      `;
    }).join("");
  };
  draw(PREVIEW);
  if (rows.length > PREVIEW) {
    note.innerHTML = `Showing first ${PREVIEW} of ${rows.length}. <a href="#" id="show-all-scores">Show all ${rows.length}</a> (or download above).`;
    note.querySelector("#show-all-scores").addEventListener("click", (e) => {
      e.preventDefault();
      draw(rows.length);
      note.textContent = `Showing all ${rows.length} rows.`;
    });
  }
}

function renderMethod(d) {
  const items = [
    `Checkpoint: <code>${escapeHtml(d.checkpoint)}</code> — the only surviving copy lives in the lica-svg-score Baseten assets, not in the lica-score repo.`,
    `Dataset: ${escapeHtml(d.dataset)}.`,
    `Each SVG rasterised at 512×512; scored with raw HPS cosine against the prompt, matching the deployed scoring convention.`,
    `Train-time eval of this checkpoint: pairwise ${fmt(d.checkpoint_train_eval.pairwise_accuracy, 3)}, MRR ${fmt(d.checkpoint_train_eval.mrr, 3)} (weakest on the "simple" bucket at ${fmt(d.checkpoint_train_eval.by_bucket.simple.pairwise_accuracy, 3)}).`,
    `This page shows curated cases only (misses + score extremes). Full per-example scores: <code>hps_omnisvg_test/results/scores.jsonl</code>.`,
  ];
  return items.map((i) => `<li>${i}</li>`).join("");
}

function wireFilters() {
  const chips = document.querySelectorAll(".chip");
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      chips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      const filter = chip.dataset.filter;
      document.querySelectorAll(".case").forEach((card) => {
        const tags = card.dataset.tags.split(" ");
        card.style.display = filter === "all" || tags.includes(filter) ? "" : "none";
      });
    });
  });
}

async function main() {
  const res = await fetch("data/hps-omnisvg.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load data: ${res.status}`);
  const d = await res.json();

  document.getElementById("headline").innerHTML = renderHeadline(d);
  document.getElementById("takeaways").innerHTML = renderTakeaways();
  document.getElementById("charts").innerHTML = renderCharts(d);
  document.getElementById("slices").innerHTML =
    sliceTable("By color-word count in prompt", d.by_color_word_count, "colors") +
    sliceTable("By prompt length", d.by_prompt_length, "length");
  document.getElementById("downloads").innerHTML = renderDownloads(d.raw_scores);
  document.getElementById("cases").innerHTML = d.examples.map(renderCase).join("");
  document.getElementById("method-list").innerHTML = renderMethod(d);
  wireFilters();
  renderScoresPreview(d.raw_scores);
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="padding:24px;white-space:pre-wrap">${escapeHtml(err.stack || err)}</pre>`;
});
