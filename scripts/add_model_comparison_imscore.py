from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests
import yaml


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "data" / "model-comparison-demo.json"
PROD_ENV_PATH = Path("/home/ubuntu/web-app/ml-engine/app/prod-env.yaml")
IMSCORE_ENDPOINT = "https://model-wnpgy843.api.baseten.co/environments/production/predict"

SCORE_ORDER = [
    "lica_score_v2",
    "hpsv21",
    "pickscore",
    "clipscore",
    "imagereward",
    "laion_aesthetic",
]
IMSCORE_IDS = ["hpsv21", "pickscore", "clipscore", "imagereward", "laion_aesthetic"]
SCORE_LABELS = {
    "lica_score_v2": "LicaScore v2",
    "hpsv21": "HPS v2.1",
    "pickscore": "PickScore",
    "clipscore": "CLIPScore",
    "imagereward": "ImageReward",
    "laion_aesthetic": "LAION aesthetic",
}


def load_prod_env(path: Path = PROD_ENV_PATH) -> dict[str, str]:
    data = yaml.safe_load(path.read_text()) or {}
    env = data.get("env_variables") or {}
    for key, value in env.items():
        if value is not None:
            os.environ.setdefault(key, str(value))
    return {key: str(value) for key, value in env.items() if value is not None}


def ok_candidates(prompt: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in prompt.get("candidates", [])
        if candidate.get("status") == "ok" and candidate.get("asset")
    ]


def ensure_score_maps(data: dict[str, Any]) -> None:
    for prompt in data.get("prompts", []):
        for candidate in prompt.get("candidates", []):
            scores = candidate.setdefault("scores", {})
            if candidate.get("score") is not None:
                scores.setdefault("lica_score_v2", candidate["score"])


def call_imscore(
    prompt: dict[str, Any],
    candidates: list[dict[str, Any]],
    api_key: str,
    timeout: float,
) -> dict[str, dict[str, float]]:
    response = requests.post(
        IMSCORE_ENDPOINT,
        headers={"Authorization": f"Api-Key {api_key}"},
        json={
            "prompt": prompt["prompt"],
            "metrics": IMSCORE_IDS,
            "candidates": [
                {
                    "id": candidate["id"],
                    "svg": (PROJECT / candidate["asset"]).read_text(encoding="utf-8"),
                }
                for candidate in candidates
            ],
        },
        timeout=timeout,
    )
    if not response.ok:
        raise requests.HTTPError(response.text[:500], response=response)
    raw_scores = response.json().get("scores", {})
    return {
        candidate_id: {metric: float(value) for metric, value in scores.items()}
        for candidate_id, scores in raw_scores.items()
    }


def score_prompt(prompt: dict[str, Any], api_key: str, timeout: float, force: bool) -> None:
    candidates = ok_candidates(prompt)
    missing = [
        candidate
        for candidate in candidates
        if force or any(candidate.get("scores", {}).get(metric) is None for metric in IMSCORE_IDS)
    ]
    if not missing:
        return
    try:
        scored = call_imscore(prompt, missing, api_key, timeout)
    except Exception as exc:
        if len(missing) == 1:
            missing[0]["imscore_error"] = repr(exc)
            return
        for candidate in missing:
            try:
                scored = call_imscore(prompt, [candidate], api_key, timeout)
                candidate.setdefault("scores", {}).update(scored.get(candidate["id"], {}))
                candidate.pop("imscore_error", None)
            except Exception as single_exc:
                candidate["imscore_error"] = repr(single_exc)
        return

    for candidate in missing:
        candidate.setdefault("scores", {}).update(scored.get(candidate["id"], {}))
        candidate.pop("imscore_error", None)


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
        metric: {model["id"]: 0 for model in data.get("models", [])}
        for metric in SCORE_ORDER
    }
    scored_by_metric = {metric: 0 for metric in SCORE_ORDER}

    for prompt in data.get("prompts", []):
        prompt["winners"] = compute_winners(prompt)
        if prompt.get("winners", {}).get("lica_score_v2"):
            prompt["best"] = prompt["winners"]["lica_score_v2"]
        for metric in SCORE_ORDER:
            for candidate in prompt.get("candidates", []):
                if candidate.get("scores", {}).get(metric) is not None:
                    scored_by_metric[metric] += 1
            winner = prompt.get("winners", {}).get(metric)
            if winner:
                metric_win_counts[metric][winner["id"]] += 1

    data["score_order"] = SCORE_ORDER
    data["score_labels"] = SCORE_LABELS
    data.setdefault("summary", {})["scored_by_metric"] = scored_by_metric
    data["summary"]["metric_win_counts"] = metric_win_counts
    for model in data.get("models", []):
        model["wins"] = metric_win_counts["lica_score_v2"].get(model["id"], 0)
        model["metric_wins"] = {
            metric: metric_win_counts[metric].get(model["id"], 0) for metric in SCORE_ORDER
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("BASETEN_API_KEY", ""))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("IMSCORE_TIMEOUT", "300")))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    load_prod_env()
    api_key = args.api_key or os.environ.get("BASETEN_API_KEY")
    if not api_key:
        raise RuntimeError("Set BASETEN_API_KEY or pass --api-key.")

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    ensure_score_maps(data)
    for prompt in data.get("prompts", []):
        score_prompt(prompt, api_key, args.timeout, args.force)
        refresh_summary(data)
        DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(prompt["id"], prompt.get("winners", {}).get("lica_score_v2"), flush=True)

    refresh_summary(data)
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data["summary"].get("scored_by_metric", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
