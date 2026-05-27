from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import requests

from build_site_data import PROMPT_PAIR_TEXTS, build_site_data


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "data" / "site-data.json"
ASSETS_DIR = PROJECT / "assets"

LICA_ENDPOINT = "https://model-32p625xq.api.baseten.co/environments/production/predict"
MODEL_SPECS = [
    {
        "id": "text-to-svg-base",
        "label": "Base",
        "endpoint": "https://model-wpj7zz8w.api.baseten.co/environments/production/predict",
    },
    {
        "id": "text-to-svg-v1",
        "label": "V1",
        "endpoint": "https://model-qklg45d3.api.baseten.co/deployment/wpjkvkp/predict",
    },
    {
        "id": "text-to-svg-v2",
        "label": "V2",
        "endpoint": "https://model-qklg45d3.api.baseten.co/environments/production/predict",
    },
]
BASELINE_ORDER = ["gt", "claude", "gemini", "gpt-5.2"]
MODEL_ORDER = [spec["id"] for spec in MODEL_SPECS]
SVG_SOURCES = set(MODEL_ORDER)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def svg_parse_error(svg: str) -> str | None:
    try:
        ElementTree.fromstring(svg)
    except ElementTree.ParseError as exc:
        return str(exc)
    return None


def read_data(args: argparse.Namespace) -> dict[str, Any]:
    if DATA_PATH.exists() and not args.rebuild_base_data:
        data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    else:
        data = build_site_data(
            argparse.Namespace(
                endpoint="",
                api_key=args.api_key,
                seed=20260521,
                timeout=args.timeout,
                limit=args.limit,
                force=False,
                skip_generation=True,
                prompt_pair_seed=2026052101,
                force_prompt_pairs=False,
                skip_prompt_pairs=True,
            )
        )
    sync_prompt_pairs(data)
    data.setdefault("model_comparison", {})
    data["model_comparison"].update(
        {
            "models": MODEL_SPECS,
            "model_order": MODEL_ORDER,
            "lica_endpoint": LICA_ENDPOINT,
            "generation_payload": {"description": "PROMPT"},
            "last_started_at": utc_now(),
        }
    )
    return data


def sync_prompt_pairs(data: dict[str, Any]) -> None:
    pairs_by_id = {pair.get("id"): pair for pair in data.get("prompt_pairs", [])}
    prompt_pairs: list[dict[str, Any]] = []
    for index, prompt in enumerate(PROMPT_PAIR_TEXTS, start=1):
        pair_id = f"prompt-pair-{index:02d}"
        pair = dict(pairs_by_id.get(pair_id, {}))
        pair.update({"id": pair_id, "index": index, "prompt": prompt})
        prompt_pairs.append(pair)
    data["prompt_pairs"] = prompt_pairs


def save_data(data: dict[str, Any], lock: threading.Lock | None = None) -> None:
    def write() -> None:
        data["generated_at"] = utc_now()
        DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if lock is None:
        write()
    else:
        with lock:
            write()


def asset_path_for(kind: str, item_id: str, model_id: str) -> str:
    safe_id = item_id.replace("/", "-")
    return f"assets/{model_id}-{safe_id}.svg"


def complete_generation(item: dict[str, Any] | None) -> bool:
    return bool(item and item.get("status") == "ok" and item.get("asset") and (PROJECT / item["asset"]).exists())


def call_generation(endpoint: str, api_key: str, prompt: str, timeout: int) -> dict[str, Any]:
    started = time.time()
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Api-Key {api_key}"},
        json={"description": prompt},
        timeout=timeout,
    )
    elapsed = time.time() - started
    response.raise_for_status()
    payload = response.json()
    payload["elapsed_seconds"] = elapsed
    return payload


def generation_item(
    *,
    spec: dict[str, Any],
    asset: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    svg = result.get("svg") or ""
    if svg:
        (PROJECT / asset).write_text(svg, encoding="utf-8")
    return {
        "source": spec["id"],
        "label": spec["label"],
        "asset": asset if svg else None,
        "status": "ok" if svg else "no_svg",
        "endpoint": spec["endpoint"],
        "svg_parse_error": result.get("svg_parse_error") or (svg_parse_error(svg) if svg else "No SVG returned."),
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "elapsed_seconds": round(float(result.get("elapsed_seconds", 0.0)), 3),
        "svg_chars": len(svg),
        "sha256": hashlib.sha256(svg.encode("utf-8")).hexdigest() if svg else None,
        "engine": result.get("engine"),
        "lora_enabled": result.get("lora_enabled"),
    }


def iter_generation_targets(data: dict[str, Any], bucket: str, include_prompt_pairs: bool) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for sample in data.get("samples", []):
        if bucket != "all" and sample.get("bucket") != bucket:
            continue
        targets.append({"kind": "sample", "id": sample["id"], "prompt": sample["prompt"], "payload": sample})
    if include_prompt_pairs:
        for pair in data.get("prompt_pairs", []):
            targets.append({"kind": "prompt_pair", "id": pair["id"], "prompt": pair["prompt"], "payload": pair})
    return targets


def run_model_generation(
    *,
    spec: dict[str, Any],
    targets: list[dict[str, Any]],
    api_key: str,
    timeout: int,
    force: bool,
    data: dict[str, Any],
    save_lock: threading.Lock,
) -> None:
    total = len(targets)
    for index, target in enumerate(targets, start=1):
        payload = target["payload"]
        generations = payload.setdefault("model_generations", {})
        asset = asset_path_for(target["kind"], target["id"], spec["id"])
        existing = generations.get(spec["id"])
        if not force and complete_generation(existing):
            print(f"[{spec['id']}] {index}/{total} cached {target['id']}", flush=True)
            continue
        print(f"[{spec['id']}] {index}/{total} generating {target['id']}", flush=True)
        try:
            result = call_generation(spec["endpoint"], api_key, target["prompt"], timeout)
            item = generation_item(spec=spec, asset=asset, result=result)
        except Exception as exc:
            item = {
                "source": spec["id"],
                "label": spec["label"],
                "asset": None,
                "status": "error",
                "endpoint": spec["endpoint"],
                "error": repr(exc),
            }
        generations[spec["id"]] = item
        print(
            f"[{spec['id']}] {target['id']} {item.get('status')} "
            f"tokens={item.get('output_tokens')} elapsed={item.get('elapsed_seconds')} "
            f"parse={item.get('svg_parse_error') or item.get('error')}",
            flush=True,
        )
        save_data(data, save_lock)


def encode_asset(asset: str) -> str:
    data = (PROJECT / asset).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def svg_from_asset(asset: str) -> str:
    return (PROJECT / asset).read_text(encoding="utf-8")


def candidate_payload(source: str, asset: str) -> dict[str, Any]:
    payload = {"id": source, "source": source}
    if source in SVG_SOURCES or asset.lower().endswith(".svg"):
        payload["svg"] = svg_from_asset(asset)
    else:
        payload["image"] = encode_asset(asset)
    return payload


def sample_candidates(sample: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sample.get("baselines", []):
        if item.get("asset"):
            rows.append({"source": item["source"], "asset": item["asset"]})
    for model_id in MODEL_ORDER:
        item = sample.get("model_generations", {}).get(model_id, {})
        if item.get("status") == "ok" and item.get("asset"):
            rows.append({"source": model_id, "asset": item["asset"]})
    return rows


def prompt_pair_candidates(pair: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        item = pair.get("model_generations", {}).get(model_id, {})
        if item.get("status") == "ok" and item.get("asset"):
            rows.append({"source": model_id, "asset": item["asset"]})
    return rows


def call_lica_score(api_key: str, prompt: str, candidates: list[dict[str, Any]], timeout: int) -> dict[str, Any]:
    response = requests.post(
        LICA_ENDPOINT,
        headers={"Authorization": f"Api-Key {api_key}"},
        json={
            "prompt": prompt,
            "score_mode": "raw_cosine",
            "candidates": candidates,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def score_one(
    *,
    item: dict[str, Any],
    candidates: list[dict[str, Any]],
    api_key: str,
    timeout: int,
) -> dict[str, Any]:
    scores: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for index in range(0, len(candidates), 2):
        chunk = candidates[index : index + 2]
        try:
            result = call_lica_score(
                api_key,
                item["prompt"],
                [candidate_payload(row["source"], row["asset"]) for row in chunk],
                timeout,
            )
            for row in result.get("scores", []):
                scores[row["id"]] = row
        except Exception as exc:
            errors.append(repr(exc))
    winner = None
    if scores:
        winner = max(scores.items(), key=lambda pair: pair[1]["score"])[0]
    return {
        "scores": scores,
        "winner": winner,
        "errors": errors,
        "scored_at": utc_now(),
        "score_mode": "raw_cosine",
    }


def score_targets(
    *,
    data: dict[str, Any],
    targets: list[dict[str, Any]],
    api_key: str,
    timeout: int,
    force: bool,
    workers: int,
    save_lock: threading.Lock,
) -> None:
    score_specs = []
    for target in targets:
        payload = target["payload"]
        candidates = sample_candidates(payload) if target["kind"] == "sample" else prompt_pair_candidates(payload)
        if not candidates:
            continue
        existing = payload.get("lica_scores", {})
        expected = {row["source"] for row in candidates}
        if not force and expected and expected.issubset(set(existing)) and payload.get("lica_winner"):
            print(f"[score] cached {target['id']}", flush=True)
            continue
        score_specs.append((target, candidates))

    print(f"[score] tasks remaining: {len(score_specs)}", flush=True)
    if not score_specs:
        return

    def run(spec: tuple[dict[str, Any], list[dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, Any]]:
        target, candidates = spec
        started = time.time()
        result = score_one(item=target, candidates=candidates, api_key=api_key, timeout=timeout)
        result["elapsed_seconds"] = round(time.time() - started, 3)
        return target, result

    completed = 0
    with futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {executor.submit(run, spec): spec for spec in score_specs}
        for future in futures.as_completed(future_map):
            target, result = future.result()
            payload = target["payload"]
            payload["lica_scores"] = result["scores"]
            payload["lica_winner"] = result["winner"]
            payload["lica_score_errors"] = result["errors"]
            payload["lica_scored_at"] = result["scored_at"]
            payload["lica_score_elapsed_seconds"] = result["elapsed_seconds"]
            completed += 1
            print(
                f"[score] {target['id']} winner={result['winner']} "
                f"errors={len(result['errors'])} elapsed={result['elapsed_seconds']} "
                f"({completed}/{len(score_specs)})",
                flush=True,
            )
            save_data(data, save_lock)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate base/v1/v2 outputs and LicaScore rankings.")
    parser.add_argument("--api-key", default=os.getenv("BASETEN_API_KEY", ""))
    parser.add_argument("--bucket", choices=["all", "simple", "medium", "complex"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--score-timeout", type=int, default=300)
    parser.add_argument("--score-workers", type=int, default=2)
    parser.add_argument("--force-generation", action="store_true")
    parser.add_argument("--force-scoring", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--skip-prompt-pairs", action="store_true")
    parser.add_argument("--rebuild-base-data", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("Set BASETEN_API_KEY or pass --api-key.")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = read_data(args)
    targets = iter_generation_targets(data, args.bucket, not args.skip_prompt_pairs)
    if args.limit:
        targets = targets[: args.limit]
    save_lock = threading.Lock()
    save_data(data, save_lock)

    if not args.skip_generation:
        with futures.ThreadPoolExecutor(max_workers=len(MODEL_SPECS)) as executor:
            submitted = [
                executor.submit(
                    run_model_generation,
                    spec=spec,
                    targets=targets,
                    api_key=args.api_key,
                    timeout=args.timeout,
                    force=args.force_generation,
                    data=data,
                    save_lock=save_lock,
                )
                for spec in MODEL_SPECS
            ]
            for future in futures.as_completed(submitted):
                future.result()

    if not args.skip_scoring:
        score_targets(
            data=data,
            targets=targets,
            api_key=args.api_key,
            timeout=args.score_timeout,
            force=args.force_scoring,
            workers=args.score_workers,
            save_lock=save_lock,
        )

    data["model_comparison"]["last_completed_at"] = utc_now()
    save_data(data, save_lock)
    print(f"done {DATA_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
