from __future__ import annotations

import argparse
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


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "data" / "site-data.json"
ENDPOINT = "https://model-wl1pmo0q.api.baseten.co/environments/production/predict"
RUNS_PER_MODE = 4
SETTINGS_LABEL = "default sampling (temperature 0.9, top_p 0.95, top_k 50)"


def svg_parse_error(svg: str) -> str | None:
    try:
        ElementTree.fromstring(svg)
    except ElementTree.ParseError as exc:
        return str(exc)
    return None


def asset_path(index: int, mode: str, run: int) -> str:
    return f"assets/lora-toggle-prompt-pair-{index:02d}-{mode}-{run:02d}.svg"


def existing_by_run(items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(item.get("run", 0)): item for item in items if item.get("run")}


def is_complete(item: dict[str, Any] | None) -> bool:
    return bool(
        item
        and item.get("status") == "ok"
        and item.get("settings") == SETTINGS_LABEL
        and item.get("asset")
        and (PROJECT / str(item["asset"])).exists()
    )


def ensure_grid(pair: dict[str, Any]) -> dict[str, Any]:
    grid = pair.setdefault("lora_toggle", {})
    grid["label"] = "SGLang LoRA ON/OFF default sampling"
    grid["settings"] = SETTINGS_LABEL
    grid["runs_per_mode"] = RUNS_PER_MODE
    grid["endpoint"] = ENDPOINT
    grid.setdefault("on", [])
    grid.setdefault("off", [])
    return grid


def save_data(data: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        data["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        data["lora_toggle_endpoint"] = ENDPOINT
        data["lora_toggle_settings"] = SETTINGS_LABEL
        data.pop("lora_toggle_temperature", None)
        DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def call_model(api_key: str, prompt: str, mode: str) -> dict[str, Any]:
    payload = {
        "description": prompt,
        "response_format": "text",
        "disable_lora": mode == "off",
    }
    started = time.time()
    response = requests.post(
        ENDPOINT,
        headers={"Authorization": f"Api-Key {api_key}"},
        json=payload,
        timeout=900,
    )
    elapsed = time.time() - started
    response.raise_for_status()
    result = response.json()
    result["elapsed_seconds"] = elapsed
    return result


def run_task(api_key: str, spec: tuple[dict[str, Any], str, int]) -> tuple[dict[str, Any], str, int, dict[str, Any]]:
    pair, mode, run = spec
    index = int(pair["index"])
    asset = asset_path(index, mode, run)
    print(f"[{index:02d}] {mode.upper()} run {run}: start", flush=True)

    try:
        result = call_model(api_key, pair["prompt"], mode)
        svg = result.get("svg") or ""
        if svg:
            (PROJECT / asset).write_text(svg, encoding="utf-8")

        item = {
            "source": f"sglang-lora-{mode}",
            "label": f"LoRA {mode.upper()} #{run}",
            "asset": asset if svg else None,
            "status": "ok" if svg else "no_svg",
            "mode": mode,
            "run": run,
            "settings": SETTINGS_LABEL,
            "disable_lora": mode == "off",
            "lora_enabled": result.get("lora_enabled"),
            "adapter": result.get("adapter"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "elapsed_seconds": round(float(result.get("elapsed_seconds", 0.0)), 3),
            "svg_parse_error": result.get("svg_parse_error")
            or (svg_parse_error(svg) if svg else "Model output did not contain a <svg>...</svg> block."),
            "svg_chars": len(svg),
            "sha256": hashlib.sha256(svg.encode("utf-8")).hexdigest(),
        }
    except Exception as exc:
        item = {
            "source": f"sglang-lora-{mode}",
            "label": f"LoRA {mode.upper()} #{run}",
            "asset": None,
            "status": "error",
            "mode": mode,
            "run": run,
            "settings": SETTINGS_LABEL,
            "disable_lora": mode == "off",
            "error": repr(exc),
        }

    return pair, mode, run, item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("BASETEN_API_KEY")
    if not api_key:
        raise RuntimeError("Set BASETEN_API_KEY.")

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    save_lock = threading.Lock()
    specs: list[tuple[dict[str, Any], str, int]] = []

    for pair in data.get("prompt_pairs", []):
        grid = ensure_grid(pair)
        for mode in ("on", "off"):
            by_run = existing_by_run(grid[mode])
            for run in range(1, RUNS_PER_MODE + 1):
                if not args.force and is_complete(by_run.get(run)):
                    continue
                specs.append((pair, mode, run))
    save_data(data, save_lock)

    print(f"tasks remaining: {len(specs)}", flush=True)
    if not specs:
        return 0

    completed = 0
    with futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {executor.submit(run_task, api_key, spec): spec for spec in specs}
        for future in futures.as_completed(future_map):
            pair, mode, run, item = future.result()
            grid = ensure_grid(pair)
            by_run = existing_by_run(grid[mode])
            by_run[run] = item
            grid[mode] = [by_run[index] for index in sorted(by_run)]
            save_data(data, save_lock)

            completed += 1
            index = int(pair["index"])
            print(
                f"[{index:02d}] {mode.upper()} run {run}: {item.get('status')} "
                f"tokens={item.get('output_tokens')} elapsed={item.get('elapsed_seconds')} "
                f"parse={item.get('svg_parse_error')} ({completed}/{len(specs)})",
                flush=True,
            )

    print(f"done {DATA_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
