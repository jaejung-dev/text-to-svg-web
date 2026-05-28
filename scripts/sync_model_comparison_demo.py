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
    ("ocean_seashells", "arrow"): "assets/model-comparison/arrow/ocean_seashells_with_bg.svg",
    ("ocean_seashells", "gpt"): "assets/model-comparison/gpt/ocean_seashells_with_bg.svg",
    ("ocean_seashells", "gemini"): "assets/model-comparison/gemini/ocean_seashells_with_bg.svg",
    ("ocean_seashells", "claude"): "assets/model-comparison/claude/ocean_seashells_with_bg.svg",
    ("ocean_seashells", "qwen_lora"): "assets/model-comparison/qwen_lora/ocean_seashells_with_bg.svg",
    ("sakura_tree", "arrow"): "assets/model-comparison/arrow/sakura_tree_with_bg.svg",
    ("fall_leaves", "gpt"): "assets/model-comparison/gpt/fall_leaves_with_bg.svg",
    ("fall_leaves", "gemini"): "assets/model-comparison/gemini/fall_leaves_with_bg.svg",
    ("fall_leaves", "claude"): "assets/model-comparison/claude/fall_leaves_with_bg.svg",
    ("fall_leaves", "qwen_lora"): "assets/model-comparison/qwen_lora/fall_leaves_with_bg.svg",
}
SCORE_OVERRIDES = {
    ("ocean_seashells", "arrow"): {
        "lica_score_v2": 0.2177734375,
        "hpsv21": 0.2089029997587204,
        "pickscore": 19.552858352661133,
        "clipscore": 31.021326065063477,
        "imagereward": 1.2282456159591675,
        "laion_aesthetic": 4.984477519989014,
    },
    ("ocean_seashells", "gpt"): {
        "lica_score_v2": 0.17578125,
        "hpsv21": 0.17387910187244415,
        "pickscore": 19.041229248046875,
        "clipscore": 25.827226638793945,
        "imagereward": -0.034814316779375076,
        "laion_aesthetic": 4.6870927810668945,
    },
    ("ocean_seashells", "gemini"): {
        "lica_score_v2": 0.1826171875,
        "hpsv21": 0.18354783952236176,
        "pickscore": 18.98903465270996,
        "clipscore": 26.019304275512695,
        "imagereward": -0.40928927063941956,
        "laion_aesthetic": 4.609762191772461,
    },
    ("ocean_seashells", "claude"): {
        "lica_score_v2": 0.1904296875,
        "hpsv21": 0.17998123168945312,
        "pickscore": 18.29886817932129,
        "clipscore": 28.778955459594727,
        "imagereward": -0.32853248715400696,
        "laion_aesthetic": 4.807924270629883,
    },
    ("ocean_seashells", "qwen_lora"): {
        "lica_score_v2": 0.15234375,
        "hpsv21": 0.15438392758369446,
        "pickscore": 18.422311782836914,
        "clipscore": 21.15886878967285,
        "imagereward": -1.3935719728469849,
        "laion_aesthetic": 4.752846717834473,
    },
    ("sakura_tree", "arrow"): {
        "lica_score_v2": 0.1552734375,
        "hpsv21": 0.14358976483345032,
        "pickscore": 18.812637329101562,
        "clipscore": 4.162923812866211,
        "imagereward": -1.7316429615020752,
        "laion_aesthetic": 4.385939121246338,
    },
    ("fall_leaves", "gpt"): {
        "lica_score_v2": 0.220703125,
        "hpsv21": 0.21799176931381226,
        "pickscore": 20.9012451171875,
        "clipscore": 26.891820907592773,
        "imagereward": -0.04102025553584099,
        "laion_aesthetic": 5.329805850982666,
    },
    ("fall_leaves", "gemini"): {
        "lica_score_v2": 0.2001953125,
        "hpsv21": 0.2069924920797348,
        "pickscore": 20.50295639038086,
        "clipscore": 27.879928588867188,
        "imagereward": -0.3255920112133026,
        "laion_aesthetic": 5.406494617462158,
    },
    ("fall_leaves", "claude"): {
        "lica_score_v2": 0.2109375,
        "hpsv21": 0.21386389434337616,
        "pickscore": 21.455570220947266,
        "clipscore": 26.584550857543945,
        "imagereward": 0.0529426708817482,
        "laion_aesthetic": 5.0993499755859375,
    },
    ("fall_leaves", "qwen_lora"): {
        "lica_score_v2": 0.18359375,
        "hpsv21": 0.17080853879451752,
        "pickscore": 19.74491310119629,
        "clipscore": 28.96927833557129,
        "imagereward": -1.2774423360824585,
        "laion_aesthetic": 5.29179573059082,
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
