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

function renderTakeaways(d) {
  const bias = d.bias?.correlations_generated_score_vs_image_features || {};
  const miss = d.bias?.miss_vs_win_render_traits || {};
  const items = [
    { warn: false, h: "Direction is correct", p: "Our fine-tuned HPS ranks the human ground-truth above the generated SVG 96.4% of the time, so it is not broken." },
    {
      warn: true, h: "Mild colorfulness preference (verified)",
      p: `No brightness bias (corr ${fmt(bias.gen_score_vs_brightness, 2)}), but a weak preference for more saturated / colorful renders (saturation ${fmt(bias.gen_score_vs_saturation, 2)}, colorfulness ${fmt(bias.gen_score_vs_colorfulness, 2)}).`,
    },
    {
      warn: true, h: "Aesthetic over fidelity (verified)",
      p: `When the model wrongly prefers the generated SVG, that render is on average MORE saturated than the ground truth (+${fmt(miss["mean_gen_minus_gt_saturation_on_miss"], 3)}); when it is right, the generated render is less saturated (${fmt(miss["mean_gen_minus_gt_saturation_on_win"], 3)}). So extra saturation can flip its judgment.`,
    },
    { warn: true, h: "Out-of-distribution prompts", p: "Trained on short captions; OmniSVG prompts are nearly all long (>50 words)." },
    {
      warn: false, h: "Complementary, not redundant",
      p: `\"Complementary\" = it does NOT just echo the existing V19 reward: the generated HPS score correlates only ~${fmt(d.headline.hps_vs_bon_reward_pearson, 2)} with bon_reward. Useful as an extra signal, not a drop-in replacement.`,
    },
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

const METRIC_LABELS = {
  hpsv21: "HPSv2.1", pickscore: "PickScore", clipscore: "CLIPScore",
  imagereward: "ImageReward", laion_aesthetic: "LAION aes.",
};
const METRIC_DECIMALS = {
  hpsv21: 3, pickscore: 2, clipscore: 1, imagereward: 2, laion_aesthetic: 2,
};

function scoreTableRow(label, gtVal, genVal, decimals, opts = {}) {
  const gtWin = gtVal > genVal;
  return `
    <tr class="${opts.lead ? "lead" : ""}">
      <th>${escapeHtml(label)}</th>
      <td class="num ${gtWin ? "win" : ""}">${fmt(gtVal, decimals)}${gtWin ? '<i class="chk">✓</i>' : ""}</td>
      <td class="num ${gtWin ? "" : "win"}">${fmt(genVal, decimals)}${gtWin ? "" : '<i class="chk">✓</i>'}</td>
    </tr>
  `;
}

function renderScoreTable(c) {
  const ms = c.metric_scores || {};
  const order = ["hpsv21", "pickscore", "clipscore", "imagereward", "laion_aesthetic"];
  const rows = [scoreTableRow("Our HPS (LoRA)", c.gt_score, c.gen_score, 3, { lead: true })];
  for (const id of order) {
    if (ms[id]) rows.push(scoreTableRow(METRIC_LABELS[id], ms[id].gt, ms[id].gen, METRIC_DECIMALS[id]));
  }
  return `
    <table class="case-scores">
      <thead><tr><th>metric</th><th>GT</th><th>GEN</th></tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table>
  `;
}

function renderDisagreementCard(c) {
  const ms = c.metric_scores || {};
  const ft = c.finetuned_hps;
  const pt = ms.hpsv21;
  const order = ["pickscore", "clipscore", "imagereward", "laion_aesthetic"];
  const ftGtWin = ft.winner === "gt";
  // both HPS rows first (the disagreement), then the other metrics for context
  const rows = [
    scoreTableRow("Fine-tuned HPS", ft.gt, ft.gen, 3, { lead: true }),
    pt ? scoreTableRow("Pretrained HPSv2.1", pt.gt, pt.gen, 3, { lead: true }) : "",
  ];
  for (const id of order) {
    if (ms[id]) rows.push(scoreTableRow(METRIC_LABELS[id], ms[id].gt, ms[id].gen, METRIC_DECIMALS[id]));
  }
  const headline = ftGtWin
    ? `Fine-tuned picks <b class="gt-txt">GT</b>, pretrained picks <b class="gen-txt">GEN</b>`
    : `Fine-tuned picks <b class="gen-txt">GEN</b>, pretrained picks <b class="gt-txt">GT</b>`;
  const sub = ftGtWin
    ? "Fine-tuning recovered the human reference the base model missed."
    : "Rare regression: fine-tuning moved away from the human reference.";
  return `
    <article class="case ${ftGtWin ? "" : "miss"}" data-dir="${escapeHtml(c.direction)}">
      <div class="case-imgs">
        <figure class="${ftGtWin ? "win" : ""}">
          <figcaption>GT</figcaption>
          <img src="${escapeHtml(c.gt_thumb)}" alt="ground truth ${escapeHtml(c.id)}" loading="lazy" />
        </figure>
        <figure class="${ftGtWin ? "" : "win"}">
          <figcaption>GEN</figcaption>
          <img src="${escapeHtml(c.gen_thumb)}" alt="generated ${escapeHtml(c.id)}" loading="lazy" />
        </figure>
      </div>
      <div class="case-body">
        <div class="verdict ${ftGtWin ? "" : "miss"}">
          <span>${headline}</span>
          <small>${escapeHtml(sub)}</small>
        </div>
        <table class="case-scores">
          <thead><tr><th>metric</th><th>GT</th><th>GEN</th></tr></thead>
          <tbody>${rows.join("")}</tbody>
        </table>
        <p class="case-prompt">${escapeHtml(c.prompt)}</p>
        <div class="tags">
          <span class="tag">${escapeHtml(c.color_words)} color words</span>
        </div>
      </div>
    </article>
  `;
}

function renderCase(c) {
  const gtWin = c.margin > 0;
  const ms = c.metric_scores || {};
  const gtVotes = Object.values(ms).filter((m) => m.winner === "gt").length;
  const total = Object.keys(ms).length;
  const verdict = gtWin
    ? `Our HPS picks <b class="gt-txt">ground-truth</b>`
    : `Our HPS picks <b class="gen-txt">generated</b> — a miss`;
  const consensus = total
    ? `${gtVotes}/${total} other metrics agree with GT`
    : "";
  return `
    <article class="case ${c.tags.includes("miss") ? "miss" : ""}" data-tags="${escapeHtml(c.tags.join(" "))}">
      <div class="case-imgs">
        <figure class="${gtWin ? "win" : ""}">
          <figcaption>GT${gtWin ? " ✓" : ""}</figcaption>
          <img src="${escapeHtml(c.gt_thumb)}" alt="ground truth ${escapeHtml(c.id)}" loading="lazy" />
        </figure>
        <figure class="${gtWin ? "" : "win"}">
          <figcaption>GEN${gtWin ? "" : " ✓"}</figcaption>
          <img src="${escapeHtml(c.gen_thumb)}" alt="generated ${escapeHtml(c.id)}" loading="lazy" />
        </figure>
      </div>
      <div class="case-body">
        <div class="verdict ${gtWin ? "" : "miss"}">
          <span>${verdict}</span>
          ${consensus ? `<small>${escapeHtml(consensus)}</small>` : ""}
        </div>
        ${renderScoreTable(c)}
        <p class="case-prompt">${escapeHtml(c.prompt)}</p>
        <div class="tags">
          ${c.tags.map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join("")}
          <span class="tag">${escapeHtml(c.color_words)} color words</span>
        </div>
      </div>
    </article>
  `;
}

function renderMultimetric(mm) {
  if (!mm) return "";
  const order = ["hpsv21", "pickscore", "clipscore", "imagereward", "laion_aesthetic"];
  const rows = order.map((id) => {
    const m = mm.per_metric[id];
    const rate = m.gt_preferred_rate;
    return `
      <tr>
        <td>${escapeHtml(m.label)}</td>
        <td class="bar-cell">
          ${pct(rate)}
          <span class="fill" style="width:${(rate * 100).toFixed(0)}%"></span>
        </td>
        <td>${pct(1 - rate)}</td>
        <td>${fmt(m.gen_score_vs_bon_reward_pearson, 2)}</td>
      </tr>
    `;
  }).join("");
  const c = mm.gt_consensus;
  return `
    <div class="slice-card wide">
      <h3>Who wins per metric — ground-truth vs V19-generated (n=${mm.n})</h3>
      <p class="muted small">Each of the 5 deployed scorers decides, per prompt, whether the human SVG or the generated SVG scores higher. Higher "GT wins" = the metric trusts the human reference more.</p>
      <table>
        <thead>
          <tr><th>metric</th><th>GT wins</th><th>GEN wins</th><th>corr. w/ V19 reward</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <div class="slice-card">
      <h3>Cross-metric consensus on the winner</h3>
      <p class="muted small">Out of 5 metrics, how many pick the ground-truth?</p>
      <table>
        <tbody>
          <tr><td>All 5 prefer ground-truth</td><td>${c.all5_prefer_gt}</td></tr>
          <tr><td>Majority (≥3) prefer ground-truth</td><td>${c["majority_prefer_gt(>=3)"]}</td></tr>
          <tr><td>All 5 prefer the generated SVG</td><td class="neg">${c.all5_prefer_generated}</td></tr>
        </tbody>
      </table>
      <p class="muted small">The ${c.all5_prefer_generated} cases where every metric prefers the generated SVG are the strongest "generation actually beat the reference" candidates.</p>
    </div>
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

function renderDisagreements(dis) {
  const summary = document.getElementById("disagree-summary");
  const grid = document.getElementById("disagree-cases");
  if (!dis || !dis.cards?.length) {
    if (summary) summary.innerHTML = "<p class='muted'>No disagreement data available.</p>";
    return;
  }
  summary.innerHTML = `
    <div class="stat"><strong>${dis.n}</strong><span>total disagreements</span><small>out of 1,000 prompts</small></div>
    <div class="stat"><strong>${dis.n_finetuned_gt_pretrained_gen}</strong><span>Fine-tuned → GT, base → GEN</span><small>fine-tuning recovered the reference</small></div>
    <div class="stat"><strong>${dis.n_finetuned_gen_pretrained_gt}</strong><span>Fine-tuned → GEN, base → GT</span><small>rare regressions</small></div>
  `;
  const draw = (filter) => {
    grid.innerHTML = dis.cards
      .filter((c) => filter === "all" || c.direction === filter)
      .map(renderDisagreementCard)
      .join("");
  };
  draw("all");
  document.querySelectorAll(".disagree-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".disagree-chip").forEach((x) => x.classList.remove("active"));
      chip.classList.add("active");
      draw(chip.dataset.filter);
    });
  });
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
  document.getElementById("takeaways").innerHTML = renderTakeaways(d);
  document.getElementById("multimetric").innerHTML = renderMultimetric(d.multimetric);
  document.getElementById("charts").innerHTML = renderCharts(d);
  document.getElementById("slices").innerHTML =
    sliceTable("By color-word count in prompt", d.by_color_word_count, "colors") +
    sliceTable("By prompt length", d.by_prompt_length, "length");
  document.getElementById("downloads").innerHTML = renderDownloads(d.raw_scores);
  document.getElementById("cases").innerHTML = d.examples.map(renderCase).join("");
  renderDisagreements(d.disagreements);
  document.getElementById("method-list").innerHTML = renderMethod(d);
  wireFilters();
  renderScoresPreview(d.raw_scores);
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="padding:24px;white-space:pre-wrap">${escapeHtml(err.stack || err)}</pre>`;
});
