from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import cairosvg
import requests
import torch


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "data" / "multigen-demo.json"
ASSET_DIR = PROJECT / "assets" / "multigen"
RENDER_DIR = PROJECT / "assets" / "multigen-renders"
DESIGNARENA_PATH = Path("/home/ubuntu/designarena_curated (1).jsonl")
TEXT_TO_SVG_V2_ENDPOINT = "https://model-qklg45d3.api.baseten.co/environments/production/predict"
LICA_SCORE_ENDPOINT = "https://model-32p625xq.api.baseten.co/environments/production/predict"
MANUAL_PROMPTS = {
    "example_dna_icon": {
        "prompt": "DNA Icon",
        "lang": "en",
        "length_tier": "short",
        "source": "provided_examples",
        "train_sim": 0.7874,
        "nearest_curated_id": "cur_0022",
    },
    "example_baby_bicycle": {
        "prompt": "a cute baby riding a bicycle",
        "lang": "en",
        "length_tier": "short",
        "source": "provided_examples",
        "train_sim": None,
        "nearest_curated_id": None,
    },
}
DEFAULT_PROMPT_IDS = [
    "example_dna_icon",
    "example_baby_bicycle",
    "cur_0008",
    "cur_0099",
    "cur_0009",
    "cur_0056",
    "cur_0192",
    "cur_0018",
]
DEFAULT_SEEDS = [2026052801, 2026052802, 2026052803, 2026052804]
LICA_SCORE_V1_ID = "lica_score_v1"
LICA_SCORE_V2_ID = "lica_score_v2"
SVG_RE = re.compile(r"<svg\b[\s\S]*?</svg>", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def parse_json_objects(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        obj, end = decoder.raw_decode(text, index)
        rows.append(obj)
        index = end
    return rows


def svg_parse_error(svg: str) -> str | None:
    try:
        ElementTree.fromstring(svg)
    except ElementTree.ParseError as exc:
        return str(exc)
    return None


def extract_svg(payload: dict[str, Any]) -> str:
    svg = payload.get("svg") or ""
    if svg:
        return svg.strip()
    if payload.get("svg_base64"):
        try:
            return base64.b64decode(str(payload["svg_base64"])).decode("utf-8").strip()
        except Exception:
            return ""
    text = payload.get("text") or payload.get("raw") or payload.get("completion") or ""
    final_text = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    match = SVG_RE.search(final_text)
    return match.group(0).strip() if match else ""


def asset_path(prompt_id: str, seed: int) -> str:
    return f"assets/multigen/{prompt_id}-seed-{seed}.svg"


def render_path(asset: str) -> str:
    out_path = RENDER_DIR / f"{Path(asset).stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        svg = (PROJECT / asset).read_text(encoding="utf-8")
        cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            write_to=str(out_path),
            output_width=512,
            output_height=512,
        )
    return str(out_path)


def generation_complete(candidate: dict[str, Any] | None) -> bool:
    return bool(
        candidate
        and candidate.get("status") == "ok"
        and candidate.get("asset")
        and (PROJECT / str(candidate["asset"])).exists()
    )


def call_generation(prompt: str, seed: int, api_key: str, timeout: int) -> dict[str, Any]:
    started = time.time()
    response = requests.post(
        TEXT_TO_SVG_V2_ENDPOINT,
        headers={"Authorization": f"Api-Key {api_key}"},
        json={"description": prompt, "seed": seed},
        timeout=timeout,
    )
    elapsed = time.time() - started
    response.raise_for_status()
    payload = response.json()
    payload["elapsed_seconds"] = elapsed
    return payload


def build_candidate(prompt_id: str, seed: int, payload: dict[str, Any]) -> dict[str, Any]:
    svg = extract_svg(payload)
    asset = asset_path(prompt_id, seed)
    parse_error = svg_parse_error(svg) if svg else "No SVG returned."
    status = "ok" if svg and not parse_error else ("invalid_svg" if svg else "no_svg")
    if svg:
        (PROJECT / asset).parent.mkdir(parents=True, exist_ok=True)
        (PROJECT / asset).write_text(svg, encoding="utf-8")
    return {
        "id": f"{prompt_id}-seed-{seed}",
        "seed": seed,
        "asset": asset if svg else None,
        "status": status,
        "svg_chars": len(svg),
        "sha256": hashlib.sha256(svg.encode("utf-8")).hexdigest() if svg else None,
        "elapsed_seconds": round(float(payload.get("elapsed_seconds", 0.0)), 3),
        "output_tokens": payload.get("output_tokens"),
        "input_tokens": payload.get("input_tokens"),
        "svg_parse_error": parse_error,
        "scores": {},
    }


def call_lica_score(prompt: str, candidates: list[dict[str, str]], api_key: str, timeout: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for index in range(0, len(candidates), 2):
        chunk = candidates[index : index + 2]
        response = requests.post(
            LICA_SCORE_ENDPOINT,
            headers={"Authorization": f"Api-Key {api_key}"},
            json={
                "prompt": prompt,
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


def load_lica_score_v2_model(checkpoint: Path, device: torch.device) -> tuple[Any, str]:
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, "/home/ubuntu/ml-platform/other-projects/lica-score/src")
    from scripts.add_hps_lora_v2_scores import load_hps_lora_model

    return load_hps_lora_model(checkpoint, device)


@torch.no_grad()
def score_lica_v2(
    checkpoint: Path,
    prompts: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, float]:
    sys.path.insert(0, "/home/ubuntu/ml-platform/other-projects/lica-score/src")
    from lica_score.train_hps_lora import load_render_tensor, score_candidate_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, precision = load_lica_score_v2_model(checkpoint, device)
    rows: list[tuple[str, str, str]] = []
    for prompt in prompts:
        for candidate in prompt["candidates"]:
            if candidate.get("status") == "ok" and candidate.get("asset"):
                rows.append((candidate["id"], prompt["prompt"], render_path(candidate["asset"])))

    scores: dict[str, float] = {}
    for index in range(0, len(rows), batch_size):
        chunk = rows[index : index + batch_size]
        tensors = []
        for _candidate_id, _prompt, path in chunk:
            tensor = load_render_tensor(path)
            if tensor.shape[-2:] != (512, 512):
                tensor = torch.nn.functional.interpolate(
                    tensor.unsqueeze(0),
                    size=(512, 512),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            tensors.append(tensor)
        images = torch.stack(tensors)
        batch_prompts = [prompt for _candidate_id, prompt, _path in chunk]
        values = score_candidate_batch(
            model,
            images,
            batch_prompts,
            device=device,
            precision=precision,
        ).tolist()
        for (candidate_id, _prompt, _path), value in zip(chunk, values):
            scores[candidate_id] = float(value)
    return scores


def select_winners(prompt: dict[str, Any]) -> None:
    for metric in (LICA_SCORE_V1_ID, LICA_SCORE_V2_ID):
        scored = [
            candidate
            for candidate in prompt["candidates"]
            if candidate.get("scores", {}).get(metric) is not None
        ]
        if scored:
            winner = max(scored, key=lambda candidate: candidate["scores"][metric])
            prompt.setdefault("winners", {})[metric] = {
                "candidate_id": winner["id"],
                "seed": winner["seed"],
                "score": winner["scores"][metric],
            }
    winners = prompt.get("winners", {})
    prompt["winner_agreement"] = (
        winners.get(LICA_SCORE_V1_ID, {}).get("candidate_id")
        == winners.get(LICA_SCORE_V2_ID, {}).get("candidate_id")
    )


def read_existing() -> dict[str, Any]:
    if DATA_PATH.exists():
        return json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("BASETEN_API_KEY", ""))
    parser.add_argument("--prompt-ids", default=",".join(DEFAULT_PROMPT_IDS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--score-timeout", type=int, default=300)
    parser.add_argument("--force-generation", action="store_true")
    parser.add_argument("--force-scoring", action="store_true")
    parser.add_argument(
        "--lica-v2-checkpoint",
        type=Path,
        default=Path(
            "/mnt/local/lica-score-runs/hps-lora-caption-20260528-091359/"
            "hps_lora_caption_epoch_002.pt"
        ),
    )
    args = parser.parse_args()
    if not args.api_key:
        raise RuntimeError("Set BASETEN_API_KEY or pass --api-key.")

    selected_ids = [value.strip() for value in args.prompt_ids.split(",") if value.strip()]
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    design_rows = {row["id"]: row for row in parse_json_objects(DESIGNARENA_PATH)}
    data = read_existing()
    prompts_by_id = {prompt["id"]: prompt for prompt in data.get("prompts", [])}
    prompts = []
    for prompt_id in selected_ids:
        source_row = design_rows.get(prompt_id) or MANUAL_PROMPTS[prompt_id]
        prompt = prompts_by_id.get(prompt_id) or {
            "id": prompt_id,
            "prompt": source_row["prompt"],
            "lang": source_row.get("lang"),
            "length_tier": source_row.get("length_tier"),
            "train_sim": source_row.get("train_sim"),
            "source": source_row.get("source"),
            "nearest_curated_id": source_row.get("nearest_curated_id"),
            "candidates": [],
        }
        prompt["prompt"] = source_row["prompt"]
        prompt["lang"] = source_row.get("lang")
        prompt["length_tier"] = source_row.get("length_tier")
        prompt["train_sim"] = source_row.get("train_sim")
        prompt["source"] = source_row.get("source")
        prompt["nearest_curated_id"] = source_row.get("nearest_curated_id")
        prompt["candidate_count"] = len(seeds)
        prompts.append(prompt)

    tasks = []
    for prompt in prompts:
        by_seed = {candidate["seed"]: candidate for candidate in prompt.get("candidates", [])}
        for seed in seeds:
            if not args.force_generation and generation_complete(by_seed.get(seed)):
                continue
            tasks.append((prompt, seed))
    print(f"[generate] tasks remaining: {len(tasks)}", flush=True)

    def generate_one(task: tuple[dict[str, Any], int]) -> tuple[dict[str, Any], int, dict[str, Any]]:
        prompt, seed = task
        try:
            payload = call_generation(prompt["prompt"], seed, args.api_key, args.timeout)
            candidate = build_candidate(prompt["id"], seed, payload)
        except Exception as exc:
            candidate = {
                "id": f"{prompt['id']}-seed-{seed}",
                "seed": seed,
                "asset": None,
                "status": "error",
                "error": repr(exc),
                "scores": {},
            }
        return prompt, seed, candidate

    if tasks:
        with futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            for prompt, seed, candidate in executor.map(generate_one, tasks):
                by_seed = {row["seed"]: row for row in prompt.get("candidates", [])}
                by_seed[seed] = candidate
                prompt["candidates"] = [by_seed[seed] for seed in seeds if seed in by_seed]
                print(
                    f"[generate] {prompt['id']} seed={seed} {candidate.get('status')} "
                    f"chars={candidate.get('svg_chars')} elapsed={candidate.get('elapsed_seconds')}",
                    flush=True,
                )

    for prompt in prompts:
        ok_candidates = [
            candidate
            for candidate in prompt.get("candidates", [])
            if candidate.get("status") == "ok" and candidate.get("asset")
        ]
        needs_v1 = args.force_scoring or any(
            candidate.get("scores", {}).get(LICA_SCORE_V1_ID) is None for candidate in ok_candidates
        )
        if ok_candidates and needs_v1:
            v1_scores = call_lica_score(
                prompt["prompt"],
                ok_candidates,
                args.api_key,
                args.score_timeout,
            )
            for candidate in ok_candidates:
                candidate.setdefault("scores", {})[LICA_SCORE_V1_ID] = v1_scores.get(candidate["id"])
            print(f"[score-v1] {prompt['id']} scored {len(v1_scores)}", flush=True)

    needs_v2 = args.force_scoring or any(
        candidate.get("status") == "ok"
        and candidate.get("scores", {}).get(LICA_SCORE_V2_ID) is None
        for prompt in prompts
        for candidate in prompt.get("candidates", [])
    )
    if needs_v2:
        v2_scores = score_lica_v2(args.lica_v2_checkpoint, prompts, batch_size=16)
        for prompt in prompts:
            for candidate in prompt.get("candidates", []):
                if candidate.get("id") in v2_scores:
                    candidate.setdefault("scores", {})[LICA_SCORE_V2_ID] = v2_scores[candidate["id"]]
        print(f"[score-v2] scored {len(v2_scores)}", flush=True)

    for prompt in prompts:
        select_winners(prompt)

    summary = {
        "prompts": len(prompts),
        "candidates": sum(len(prompt.get("candidates", [])) for prompt in prompts),
        "agreements": sum(1 for prompt in prompts if prompt.get("winner_agreement")),
    }
    data = {
        "generated_at": utc_now(),
        "source_file": str(DESIGNARENA_PATH),
        "model": {
            "id": "text-to-svg-v2",
            "label": "Text-to-SVG V2",
            "endpoint": TEXT_TO_SVG_V2_ENDPOINT,
        },
        "seeds": seeds,
        "score_labels": {
            LICA_SCORE_V1_ID: "LicaScore",
            LICA_SCORE_V2_ID: "LicaScore v2",
        },
        "summary": summary,
        "prompts": prompts,
    }
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"updated {DATA_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
