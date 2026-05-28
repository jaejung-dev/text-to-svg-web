from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/home/ubuntu/data_for_demo")
SOURCE_DATA_PATH = SOURCE_ROOT / "demo-data.json"
DATA_PATH = PROJECT / "data" / "model-comparison-demo.json"
ASSET_ROOT = PROJECT / "assets" / "model-comparison"

EXCLUDED_PROMPTS = {"potted_plant_growing", "classic_sailboat", "cartoon_girl_waving"}
MODEL_ORDER = ["arrow", "gpt", "gemini", "claude", "qwen_lora"]
MODEL_LABELS = {
    "arrow": "Arrow 1.1",
    "gpt": "GPT-5.4",
    "gemini": "Gemini 3.1 Pro Preview",
    "claude": "Claude Opus 4.7",
    "qwen_lora": "Qwen Base",
}
SHORT_LABELS = {
    "arrow": "Arrow",
    "gpt": "GPT",
    "gemini": "Gemini",
    "claude": "Claude",
    "qwen_lora": "Qwen",
}
SCORE_ORDER = [
    "lica_score_v2",
    "hpsv21",
    "pickscore",
    "clipscore",
    "imagereward",
    "laion_aesthetic",
]
SCORE_LABELS = {
    "lica_score_v2": "LicaScore",
    "hpsv21": "HPS v2.1",
    "pickscore": "PickScore",
    "clipscore": "CLIPScore",
    "imagereward": "ImageReward",
    "laion_aesthetic": "LAION aesthetic",
}
ASSET_OVERRIDES = {
    ("sakura_tree", "arrow"): "assets/model-comparison/arrow/sakura_tree_with_bg.svg",
}
SCORE_OVERRIDES = {
    ("sakura_tree", "arrow"): {
        "lica_score_v2": 0.1552734375,
        "hpsv21": 0.1482030153274536,
        "pickscore": 18.770793914794922,
        "clipscore": 10.886899948120117,
        "imagereward": -1.8555684089660645,
        "laion_aesthetic": 4.778088092803955,
    },
}


def copy_asset(source_asset: str, model_id: str, prompt_id: str) -> str | None:
    source = SOURCE_ROOT / source_asset
    if not source.exists():
        return None
    target = ASSET_ROOT / model_id / f"{prompt_id}.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return str(target.relative_to(PROJECT))


def old_score_lookup() -> dict[tuple[str, str], dict[str, Any]]:
    if not DATA_PATH.exists():
        return {}
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for prompt in data.get("prompts", []):
        for candidate in prompt.get("candidates", []):
            lookup[(prompt["id"], candidate["id"])] = candidate.get("scores", {})
    return lookup


def prompt_text(row: dict[str, Any]) -> tuple[str, str | None]:
    if row["id"] == "horse_logo_ar" and row.get("prompt_en"):
        return row["prompt_en"], None
    return row["prompt"], row.get("prompt_en")


def candidate_scores(
    prompt_id: str,
    candidate: dict[str, Any],
    previous_scores: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    if (prompt_id, candidate["id"]) in SCORE_OVERRIDES:
        return dict(SCORE_OVERRIDES[(prompt_id, candidate["id"])])
    scores = dict(previous_scores.get((prompt_id, candidate["id"]), {}))
    if candidate.get("score") is not None:
        score = candidate["score"]
        scores["lica_score_v2"] = score.get("score") if isinstance(score, dict) else score
    return scores


def compute_winners(prompt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    winners: dict[str, dict[str, Any]] = {}
    for metric in SCORE_ORDER:
        scored = [
            candidate
            for candidate in prompt.get("candidates", [])
            if candidate.get("scores", {}).get(metric) is not None
        ]
        if not scored:
            continue
        winner = max(scored, key=lambda candidate: candidate["scores"][metric])
        winners[metric] = {
            "id": winner["id"],
            "label": winner["label"],
            "score": winner["scores"][metric],
        }
    return winners


def refresh_summary(data: dict[str, Any]) -> None:
    metric_win_counts = {
        metric: {model_id: 0 for model_id in MODEL_ORDER}
        for metric in SCORE_ORDER
    }
    scored_by_metric = {metric: 0 for metric in SCORE_ORDER}

    for prompt in data["prompts"]:
        prompt["winners"] = compute_winners(prompt)
        prompt["best"] = prompt.get("winners", {}).get("lica_score_v2")
        for metric in SCORE_ORDER:
            for candidate in prompt["candidates"]:
                if candidate.get("scores", {}).get(metric) is not None:
                    scored_by_metric[metric] += 1
            winner = prompt.get("winners", {}).get(metric)
            if winner:
                metric_win_counts[metric][winner["id"]] += 1

    for model in data["models"]:
        model["wins"] = metric_win_counts["lica_score_v2"].get(model["id"], 0)
        model["metric_wins"] = {
            metric: metric_win_counts[metric].get(model["id"], 0)
            for metric in SCORE_ORDER
        }

    candidates = [candidate for prompt in data["prompts"] for candidate in prompt["candidates"]]
    scored_candidates = [
        candidate for candidate in candidates if candidate.get("scores", {}).get("lica_score_v2") is not None
    ]
    data["summary"] = {
        "prompts": len(data["prompts"]),
        "models": len(data["models"]),
        "candidates": len(candidates),
        "scored_candidates": len(scored_candidates),
        "missing_candidates": len(candidates) - len(scored_candidates),
        "winner_counts": metric_win_counts["lica_score_v2"],
        "scored_by_metric": scored_by_metric,
        "metric_win_counts": metric_win_counts,
    }


def build_data() -> dict[str, Any]:
    source = json.loads(SOURCE_DATA_PATH.read_text(encoding="utf-8"))
    previous_scores = old_score_lookup()
    prompts: list[dict[str, Any]] = []

    for row in source.get("prompts", []):
        if row["id"] in EXCLUDED_PROMPTS:
            continue

        prompt, prompt_en = prompt_text(row)
        prompt_row: dict[str, Any] = {
            "id": row["id"],
            "prompt": prompt,
            "prompt_en": prompt_en,
            "candidates": [],
        }
        by_model = {candidate["id"]: candidate for candidate in row.get("candidates", [])}
        for model_id in MODEL_ORDER:
            candidate = by_model.get(model_id)
            if not candidate:
                continue
            copied_asset = ASSET_OVERRIDES.get((row["id"], model_id)) or copy_asset(
                candidate.get("asset", ""),
                model_id,
                row["id"],
            )
            output = {
                "id": model_id,
                "label": MODEL_LABELS[model_id],
                "asset": copied_asset,
                "status": candidate.get("status", "missing"),
                "elapsed_seconds": candidate.get("elapsed_seconds"),
                "scores": candidate_scores(row["id"], candidate, previous_scores),
            }
            if candidate.get("error"):
                output["error"] = candidate["error"]
            if candidate.get("score_error"):
                output["score_error"] = candidate["score_error"]
            prompt_row["candidates"].append(output)
        prompts.append(prompt_row)

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE_DATA_PATH.relative_to(SOURCE_ROOT.parent)),
        "score_name": "LicaScore raw cosine",
        "score_order": SCORE_ORDER,
        "score_labels": SCORE_LABELS,
        "models": [
            {
                "id": model_id,
                "label": MODEL_LABELS[model_id],
                "short_label": SHORT_LABELS[model_id],
                "count": 0,
                "wins": 0,
                "metric_wins": {},
            }
            for model_id in MODEL_ORDER
        ],
        "prompts": prompts,
    }

    for model in data["models"]:
        model["count"] = sum(
            1
            for prompt in prompts
            for candidate in prompt["candidates"]
            if candidate["id"] == model["id"] and candidate.get("status") == "ok"
        )
    refresh_summary(data)
    return data


def main() -> int:
    data = build_data()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"synced {data['summary']['prompts']} prompts and {data['summary']['candidates']} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
