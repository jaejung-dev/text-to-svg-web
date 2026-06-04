"""Build the data + curated thumbnails for the HPS-on-OmniSVG report page.

Reads the scoring output produced by
/home/ubuntu/hps_omnisvg_test/run_hps_omnisvg_test.py and emits:
  - data/hps-omnisvg.json         (summary stats + curated example metadata)
  - assets/hps-omnisvg-renders/   (small GT/generated PNG thumbnails for the
                                   curated examples only, kept lightweight)

We intentionally show only the *important* cases (model misses, saturation-bias
cases, score extremes) instead of all 1000, to keep the page fast and readable.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

WEB = Path("/home/ubuntu/text-to-svg-web")
RESULTS = Path("/home/ubuntu/hps_omnisvg_test/results")
EXAMPLES = Path("/home/ubuntu/hps_omnisvg_test/extracted/omnisvg_1k_v19_examples/examples")
THUMB_DIR = WEB / "assets" / "hps-omnisvg-renders"
DATA_OUT = WEB / "data" / "hps-omnisvg.json"
SCORES_DIR = WEB / "data" / "hps-omnisvg-scores"
THUMB_PX = 320

MULTIMETRIC_IN = RESULTS / "multimetric.jsonl"
MULTIMETRIC_REPORT = RESULTS / "multimetric_report.json"
BIAS_REPORT = RESULTS / "bias.json"
METRIC_IDS = ["hpsv21", "pickscore", "clipscore", "imagereward", "laion_aesthetic"]

# Columns published in the downloadable raw-score files. `id` matches the
# OmniSVG example id (folder name in omnisvg_1k_v19_examples/examples/<id>/),
# so teammates can join these scores back onto the source dataset. The licascore_*
# columns are our fine-tuned HPS; the rest are the deployed lica-svg-imscore stack.
SCORE_COLUMNS = (
    ["id"]
    + ["licascore_hps_gt", "licascore_hps_gen"]
    + [f"{m}_gt" for m in METRIC_IDS]
    + [f"{m}_gen" for m in METRIC_IDS]
    + ["bon_reward", "prompt"]
)

COLOR_WORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
    "white", "gray", "grey", "brown", "cyan", "magenta", "gold", "silver",
    "teal", "navy", "beige", "maroon", "violet", "turquoise", "pastel",
}


def load_scores() -> list[dict]:
    return [json.loads(l) for l in (RESULTS / "scores.jsonl").read_text().splitlines() if l.strip()]


def make_thumb(example_id: str, kind: str) -> str | None:
    """kind in {ground_truth, generated}. Returns web-relative asset path."""
    src = EXAMPLES / example_id / f"{kind}.rendered.png"
    if not src.exists():
        return None
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    out = THUMB_DIR / f"{example_id}-{kind}.png"
    if not out.exists():
        img = Image.open(src).convert("RGB")
        img.thumbnail((THUMB_PX, THUMB_PX))
        # paste onto white so transparent/edge SVGs never look black
        canvas = Image.new("RGB", (THUMB_PX, THUMB_PX), (255, 255, 255))
        canvas.paste(img, ((THUMB_PX - img.width) // 2, (THUMB_PX - img.height) // 2))
        canvas.save(out, format="PNG", optimize=True)
    return f"assets/hps-omnisvg-renders/{out.name}"


def color_count(prompt: str) -> int:
    return len(set(re.findall(r"[a-z]+", prompt.lower())) & COLOR_WORDS)


def stats(arr: np.ndarray) -> dict:
    return {
        "n": int(arr.size), "mean": round(float(arr.mean()), 4), "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4), "p25": round(float(np.percentile(arr, 25)), 4),
        "median": round(float(np.median(arr)), 4), "p75": round(float(np.percentile(arr, 75)), 4),
        "max": round(float(arr.max()), 4),
    }


def histogram(values: np.ndarray, lo: float, hi: float, bins: int = 24) -> list[dict]:
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    return [
        {"x0": round(float(edges[i]), 4), "x1": round(float(edges[i + 1]), 4), "count": int(counts[i])}
        for i in range(bins)
    ]


def curate(results: list[dict], multimetric: dict[str, dict]) -> list[dict]:
    """Pick the important cases: model misses + saturation-bias evidence + extremes."""
    res = [r for r in results if not (np.isnan(r["gt_score"]) or np.isnan(r["gen_score"]))]
    by_margin = sorted(res, key=lambda r: r["margin_gt_minus_gen"])

    picked: dict[str, dict] = {}

    def add(rec, tag):
        if rec["id"] not in picked:
            picked[rec["id"]] = {**rec, "tags": []}
        if tag not in picked[rec["id"]]["tags"]:
            picked[rec["id"]]["tags"].append(tag)

    for r in by_margin[:6]:                       # model misses: prefers generated over GT
        add(r, "miss")
    for r in by_margin[-4:]:                       # clearest GT wins
        add(r, "gt_win")
    for r in sorted(res, key=lambda r: r["gen_score"])[:5]:   # lowest generated scores (often pastel/muted)
        add(r, "low_score")
    for r in sorted(res, key=lambda r: r["gen_score"], reverse=True)[:4]:  # highest generated
        add(r, "high_score")

    cards = []
    for rec in picked.values():
        gt_thumb = make_thumb(rec["id"], "ground_truth")
        gen_thumb = make_thumb(rec["id"], "generated")
        if not gt_thumb or not gen_thumb:
            continue
        mm = multimetric.get(rec["id"], {})
        metric_scores = {}
        for m in METRIC_IDS:
            if mm.get("gt") and mm.get("gen"):
                gt_v = mm["gt"][m]
                gen_v = mm["gen"][m]
                metric_scores[m] = {
                    "gt": round(gt_v, 4),
                    "gen": round(gen_v, 4),
                    "winner": "gt" if gt_v > gen_v else "gen",
                }
        cards.append({
            "id": rec["id"],
            "prompt": rec["prompt"],
            "gt_score": round(rec["gt_score"], 4),
            "gen_score": round(rec["gen_score"], 4),
            "margin": round(rec["margin_gt_minus_gen"], 4),
            "bon_reward": round(rec["bon_reward"], 4) if rec.get("bon_reward") is not None else None,
            "color_words": color_count(rec["prompt"]),
            "gt_thumb": gt_thumb,
            "gen_thumb": gen_thumb,
            "tags": rec["tags"],
            "model_prefers": "generated" if rec["margin_gt_minus_gen"] < 0 else "ground_truth",
            "metric_scores": metric_scores,
        })
    cards.sort(key=lambda c: c["margin"])
    return cards


def load_multimetric() -> dict[str, dict]:
    """id -> {gt: {metric: score}, gen: {...}, bon_reward}. Empty if not run yet."""
    if not MULTIMETRIC_IN.exists():
        return {}
    out = {}
    for line in MULTIMETRIC_IN.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def export_raw_scores(results: list[dict], multimetric: dict[str, dict]) -> dict:
    """Publish the full per-SVG scores for downstream data-matching (JSONL + CSV).

    `id` is the OmniSVG example id, so teammates can join on it directly. Includes
    our fine-tuned HPS (licascore_hps_*) plus all 5 deployed imscore metrics.
    """
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    base = sorted(results, key=lambda r: r["id"])

    def flat_row(r: dict) -> dict:
        mm = multimetric.get(r["id"], {})
        gt = mm.get("gt", {})
        gen = mm.get("gen", {})
        row = {
            "id": r["id"],
            "licascore_hps_gt": r["gt_score"],
            "licascore_hps_gen": r["gen_score"],
            "bon_reward": r.get("bon_reward"),
            "prompt": r["prompt"],
        }
        for m in METRIC_IDS:
            row[f"{m}_gt"] = gt.get(m)
            row[f"{m}_gen"] = gen.get(m)
        return row

    rows = [flat_row(r) for r in base]

    jsonl_path = SCORES_DIR / "scores.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: r.get(k) for k in SCORE_COLUMNS}, ensure_ascii=False) + "\n")

    csv_path = SCORES_DIR / "scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in SCORE_COLUMNS})

    return {
        "jsonl": f"data/hps-omnisvg-scores/{jsonl_path.name}",
        "csv": f"data/hps-omnisvg-scores/{csv_path.name}",
        "rows": len(rows),
        "columns": SCORE_COLUMNS,
        "join_key": "id (OmniSVG example id = examples/<id>/ folder name)",
        "metrics_included": ["licascore_hps (fine-tuned)"] + METRIC_IDS,
    }


def main() -> int:
    report = json.loads((RESULTS / "report.json").read_text())
    results = load_scores()
    multimetric = load_multimetric()
    multimetric_report = json.loads(MULTIMETRIC_REPORT.read_text()) if MULTIMETRIC_REPORT.exists() else None
    bias = json.loads(BIAS_REPORT.read_text()) if BIAS_REPORT.exists() else None
    res = [r for r in results if not (np.isnan(r["gt_score"]) or np.isnan(r["gen_score"]))]

    gt = np.array([r["gt_score"] for r in res])
    gen = np.array([r["gen_score"] for r in res])
    margin = np.array([r["margin_gt_minus_gen"] for r in res])

    payload = {
        "generated_at": report.get("checkpoint_train_eval") and __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "checkpoint": "hps_lora_caption_epoch_002.pt (HPSv2.1 + LoRA, lica-score)",
        "dataset": "omnisvg_1k_v19_examples (1000 prompts; ground-truth SVG vs V19 Best-of-8 generated SVG)",
        "headline": {
            "n": len(res),
            "gt_preferred_rate": report["analysis"]["gt_preferred_rate"],
            "gen_preferred_rate": report["analysis"]["gen_preferred_rate"],
            "hps_vs_bon_reward_pearson": report["analysis"]["hps_gen_vs_bon_reward_pearson"],
            "score_range": [round(float(min(gt.min(), gen.min())), 4), round(float(max(gt.max(), gen.max())), 4)],
            "median_margin": round(float(np.median(margin)), 4),
        },
        "dist": {
            "gt": stats(gt),
            "generated": stats(gen),
            "margin": stats(margin),
        },
        "histograms": {
            "gt": histogram(gt, 0.05, 0.35),
            "generated": histogram(gen, 0.05, 0.35),
            "margin": histogram(margin, -0.1, 0.2),
        },
        "by_color_word_count": report["analysis"]["by_color_word_count"],
        "by_prompt_length": report["analysis"]["by_prompt_length"],
        "checkpoint_train_eval": report["checkpoint_train_eval"],
        "multimetric": multimetric_report,
        "bias": bias,
        "raw_scores": export_raw_scores(results, multimetric),
        "examples": curate(results, multimetric),
    }

    DATA_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_thumbs = len(list(THUMB_DIR.glob("*.png"))) if THUMB_DIR.exists() else 0
    print(f"wrote {DATA_OUT} ({len(payload['examples'])} curated cards, {n_thumbs} thumbnails)")
    print(f"wrote raw scores: {payload['raw_scores']['rows']} rows -> "
          f"{payload['raw_scores']['jsonl']} + {payload['raw_scores']['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
