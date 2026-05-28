from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cairosvg
import requests
import torch


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "data" / "blue-star-scores.json"
ASSET_DIR = PROJECT / "assets" / "blue-star-scores"
RENDER_DIR = PROJECT / "assets" / "blue-star-score-renders"
PROMPT = "a blue star icon"
LICA_SCORE_ENDPOINT = "https://model-32p625xq.api.baseten.co/environments/production/predict"
IMSCORE_ENDPOINT = "https://model-wnpgy843.api.baseten.co/environments/production/predict"
LICA_SCORE_V1_ID = "lica_score_v1"
LICA_SCORE_V2_ID = "lica_score_v2"
IMSCORE_IDS = ["hpsv21", "pickscore", "clipscore", "imagereward", "laion_aesthetic"]
DEFAULT_CHECKPOINT = Path(
    "/mnt/local/lica-score-runs/hps-lora-caption-20260528-091359/"
    "hps_lora_caption_epoch_002.pt"
)
SOURCE_FILES = [
    {
        "id": "candidate_qz4y",
        "label": "Candidate qz4y",
        "path": Path("/home/ubuntu/qz4yZIiumouV6qffbdhC.svg"),
    },
    {
        "id": "candidate_hhAv",
        "label": "Candidate hhAv",
        "path": Path("/home/ubuntu/hhAvzKdYHszN3VqUm79S.svg"),
    },
    {
        "id": "candidate_u379",
        "label": "Candidate u379",
        "path": Path("/home/ubuntu/u3793uzasWyNaupF36H7.svg"),
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def copy_assets() -> list[dict[str, Any]]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for spec in SOURCE_FILES:
        destination = ASSET_DIR / spec["path"].name
        shutil.copyfile(spec["path"], destination)
        svg = destination.read_text(encoding="utf-8")
        candidates.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "filename": spec["path"].name,
                "source_path": str(spec["path"]),
                "asset": str(destination.relative_to(PROJECT)),
                "svg_chars": len(svg),
                "scores": {},
            }
        )
    return candidates


def call_lica_score(candidates: list[dict[str, Any]], api_key: str, timeout: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for index in range(0, len(candidates), 2):
        chunk = candidates[index : index + 2]
        response = requests.post(
            LICA_SCORE_ENDPOINT,
            headers={"Authorization": f"Api-Key {api_key}"},
            json={
                "prompt": PROMPT,
                "score_mode": "raw_cosine",
                "candidates": [
                    {
                        "id": candidate["id"],
                        "source": candidate["id"],
                        "svg": (PROJECT / candidate["asset"]).read_text(encoding="utf-8"),
                    }
                    for candidate in chunk
                ],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        raw_scores = payload.get("scores", [])
        if isinstance(raw_scores, dict):
            for key, row in raw_scores.items():
                scores[key] = float(row["score"] if isinstance(row, dict) else row)
        else:
            for row in raw_scores:
                scores[row["id"]] = float(row["score"])
    return scores


def call_imscore(candidates: list[dict[str, Any]], api_key: str, timeout: int) -> dict[str, dict[str, float]]:
    response = requests.post(
        IMSCORE_ENDPOINT,
        headers={"Authorization": f"Api-Key {api_key}"},
        json={
            "prompt": PROMPT,
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
    response.raise_for_status()
    payload = response.json()
    raw_scores = payload.get("scores", {})
    return {
        candidate_id: {metric: float(value) for metric, value in scores.items()}
        for candidate_id, scores in raw_scores.items()
    }


def render_svg(candidate: dict[str, Any]) -> str:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    render_path = RENDER_DIR / f"{Path(candidate['asset']).stem}.png"
    cairosvg.svg2png(
        bytestring=(PROJECT / candidate["asset"]).read_bytes(),
        write_to=str(render_path),
        output_width=512,
        output_height=512,
    )
    return str(render_path)


def load_lica_v2_model(checkpoint: Path, device: torch.device) -> tuple[Any, str]:
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, "/home/ubuntu/ml-platform/other-projects/lica-score/src")
    from scripts.add_hps_lora_v2_scores import load_hps_lora_model

    return load_hps_lora_model(checkpoint, device)


@torch.no_grad()
def score_lica_v2(candidates: list[dict[str, Any]], checkpoint: Path) -> dict[str, float]:
    sys.path.insert(0, "/home/ubuntu/ml-platform/other-projects/lica-score/src")
    from lica_score.train_hps_lora import load_render_tensor, score_candidate_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, precision = load_lica_v2_model(checkpoint, device)
    tensors = []
    for candidate in candidates:
        tensor = load_render_tensor(render_svg(candidate))
        if tensor.shape[-2:] != (512, 512):
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        tensors.append(tensor)
    values = score_candidate_batch(
        model,
        torch.stack(tensors),
        [PROMPT] * len(candidates),
        device=device,
        precision=precision,
    ).tolist()
    return {candidate["id"]: float(value) for candidate, value in zip(candidates, values)}


def build_winners(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    winners: dict[str, dict[str, Any]] = {}
    for score_id in [LICA_SCORE_V1_ID, LICA_SCORE_V2_ID, *IMSCORE_IDS]:
        scored = [
            candidate
            for candidate in candidates
            if candidate.get("scores", {}).get(score_id) is not None
        ]
        if not scored:
            continue
        winner = max(scored, key=lambda candidate: candidate["scores"][score_id])
        winners[score_id] = {
            "id": winner["id"],
            "label": winner["label"],
            "score": winner["scores"][score_id],
        }
    return winners


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("BASETEN_API_KEY", ""))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if not args.api_key:
        raise RuntimeError("Set BASETEN_API_KEY or pass --api-key.")

    candidates = copy_assets()
    started = time.time()
    v1_scores = call_lica_score(candidates, args.api_key, args.timeout)
    im_scores = call_imscore(candidates, args.api_key, args.timeout)
    v2_scores = score_lica_v2(candidates, args.checkpoint)
    for candidate in candidates:
        candidate["scores"][LICA_SCORE_V1_ID] = v1_scores.get(candidate["id"])
        candidate["scores"][LICA_SCORE_V2_ID] = v2_scores.get(candidate["id"])
        candidate["scores"].update(im_scores.get(candidate["id"], {}))

    data = {
        "generated_at": utc_now(),
        "prompt": PROMPT,
        "score_order": [LICA_SCORE_V1_ID, LICA_SCORE_V2_ID, *IMSCORE_IDS],
        "score_labels": {
            LICA_SCORE_V1_ID: "LicaScore",
            LICA_SCORE_V2_ID: "LicaScore v2",
            "hpsv21": "HPS v2.1",
            "pickscore": "PickScore",
            "clipscore": "CLIPScore",
            "imagereward": "ImageReward",
            "laion_aesthetic": "LAION aesthetic",
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "winners": build_winners(candidates),
        "candidates": candidates,
    }
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"winners": data["winners"], "elapsed_seconds": data["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
