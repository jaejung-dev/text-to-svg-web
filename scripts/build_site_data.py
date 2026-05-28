from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPORT = Path("/home/ubuntu/lica-score-web/report-data.json")
SOURCE_ASSETS = Path("/home/ubuntu/lica-score-web")
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_PATH = PROJECT_ROOT / "data" / "site-data.json"
DEFAULT_ENDPOINT = "https://model-qklg45d3.api.baseten.co/environments/production/predict"
BASELINE_ORDER = ["gt", "claude", "gemini", "gpt-5.2"]
REPORT_SCORE_ORDER = [
    "qwen8b_epoch_3",
    "imscore_hpsv21",
    "imscore_hpsv3",
    "imscore_pickscore",
    "imscore_mpsv1",
    "imscore_clipscore",
    "imscore_imagereward",
    "imscore_laion_aesthetic",
]
REPORT_SCORE_LABELS = {
    "qwen8b_epoch_3": "LicaScore",
    "imscore_hpsv21": "HPSv2.1",
    "imscore_hpsv3": "HPSv3",
    "imscore_pickscore": "PickScore",
    "imscore_mpsv1": "MPSv1",
    "imscore_clipscore": "CLIPScore",
    "imscore_imagereward": "ImageReward",
    "imscore_laion_aesthetic": "LAION aesthetic",
}
PROMPT_PAIR_TEXTS = [
    "Design a modern logo to illustrate the concept of SAP BASIS Support. You may include the shape of the SAP logo, but without the letters SAP.",
    "Create an icon for an sftp dlp feature",
    "illustrate a mountain scene with an eagle above",
    "Make a design",
    "Create a single SVG tile for a 2D top-down game map. This tile represents a luxury ornate crimson carpet/rug viewed from directly above, used for the executive floor of a cyberpunk corporate tower owned by a powerful female CEO. It should evoke dark opulence — think Persian rug meets corporate noir.",
    "i want a logo for GhostVector",
    'A simple, flat design icon for a digital "Word Cruncher." Two-color palette: Deep Navy and Bright Teal. The icon features a stylized book being compressed into a diamond shape, symbolizing the refinement of language. High contrast, white background, minimalist vector style.',
    "Create a professional logo for a communications company called Bell",
    "Design a modern logo to illustrate the concept of SAP BASIS Support. You may include the shape of the SAP logo, but without the letters SAP.",
    'The logo is a clean, elegant wordmark that reads "Calidora" in a flowing script. It is set in a deep navy blue against a white background, giving it a refined, understated feel. The capital "C" is oversized and dramatic, beginning with a tall upward loop and sweeping down into a long, curved underline that subtly supports the rest of the word. That underline tapers off naturally near the end, creating movement without feeling busy. The lettering is smooth and connected, with soft, rounded strokes and consistent line weight - more modern script than ornate calligraphy. It feels feminine but not overly delicate. There are no additional symbols, icons, or embellishments; the personality comes entirely from the typography. Overall vibe: Minimal. Confident. Slightly romantic. Boutique-luxury without trying too hard.',
]


def asset_name(group_id: str, source: str, original_path: str) -> str:
    suffix = Path(original_path).suffix or ".png"
    digest = hashlib.sha1(f"{group_id}:{source}:{original_path}".encode("utf-8")).hexdigest()[:10]
    safe_source = source.replace(".", "-").replace(" ", "-")
    return f"{group_id}-{safe_source}-{digest}{suffix}"


def copy_asset(group_id: str, entry: dict[str, Any]) -> str:
    source_path = SOURCE_ASSETS / entry["image"]
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    name = asset_name(group_id, entry["source"], entry["image"])
    dest = ASSETS_DIR / name
    if not dest.exists():
        shutil.copy2(source_path, dest)
    return f"assets/{name}"


def svg_parse_error(svg: str) -> str | None:
    try:
        ElementTree.fromstring(svg)
    except ElementTree.ParseError as exc:
        return str(exc)
    return None


def call_text_to_svg(endpoint: str, api_key: str, prompt: str, seed: int, timeout: int) -> dict[str, Any]:
    started = time.time()
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Api-Key {api_key}"},
        json={
            "description": prompt,
            "response_format": "b64_json",
            "seed": seed,
        },
        timeout=timeout,
    )
    elapsed = time.time() - started
    response.raise_for_status()
    payload = response.json()
    payload["elapsed_seconds"] = elapsed
    return payload


def load_existing_data() -> dict[str, Any]:
    if not DATA_PATH.exists():
        return {}
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def load_existing_generations(existing_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        sample["id"]: sample["generated"]
        for sample in existing_data.get("samples", [])
        if sample.get("generated", {}).get("asset")
    }


def load_existing_prompt_pairs(existing_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        pair["id"]: pair["generated"]
        for pair in existing_data.get("prompt_pairs", [])
        if pair.get("generated", {}).get("asset")
    }


def build_prompt_pairs(args: argparse.Namespace, api_key: str, existing_data: dict[str, Any]) -> list[dict[str, Any]]:
    existing_pairs = load_existing_prompt_pairs(existing_data)
    pairs: list[dict[str, Any]] = []

    for index, prompt in enumerate(PROMPT_PAIR_TEXTS, start=1):
        pair_id = f"prompt-pair-{index:02d}"
        generated_asset = f"assets/{pair_id}.svg"
        generated_path = PROJECT_ROOT / generated_asset
        generated = existing_pairs.get(pair_id, {})

        if generated_path.exists() and generated and not args.force_prompt_pairs:
            print(f"[prompt {index}/{len(PROMPT_PAIR_TEXTS)}] cached generation {pair_id}")
        elif args.skip_generation or args.skip_prompt_pairs:
            generated = {
                "source": "text-to-svg-production",
                "label": "Text-to-SVG Production",
                "asset": generated_asset if generated_path.exists() else None,
                "status": "cached" if generated_path.exists() else "not_generated",
            }
        else:
            if not api_key:
                raise RuntimeError("Set BASETEN_API_KEY or pass --api-key to generate prompt pair outputs.")
            print(f"[prompt {index}/{len(PROMPT_PAIR_TEXTS)}] generating text-to-SVG output for {pair_id}")
            result = call_text_to_svg(
                endpoint=args.endpoint,
                api_key=api_key,
                prompt=prompt,
                seed=args.prompt_pair_seed + index - 1,
                timeout=args.timeout,
            )
            svg = result.get("svg") or ""
            if not svg:
                raise RuntimeError(f"No SVG returned for {pair_id}: {result}")
            generated_path.write_text(svg, encoding="utf-8")
            generated = {
                "source": "text-to-svg-production",
                "label": "Text-to-SVG Production",
                "asset": generated_asset,
                "status": "ok",
                "svg_parse_error": result.get("svg_parse_error") or svg_parse_error(svg),
                "input_tokens": result.get("input_tokens"),
                "output_tokens": result.get("output_tokens"),
                "elapsed_seconds": round(float(result.get("elapsed_seconds", 0.0)), 3),
                "seed": args.prompt_pair_seed + index - 1,
            }
            print(
                "  "
                f"out={generated['output_tokens']} tokens, "
                f"{generated['elapsed_seconds']}s, "
                f"parse={generated['svg_parse_error']}"
            )

        pairs.append(
            {
                "id": pair_id,
                "index": index,
                "prompt": prompt,
                "generated": generated,
            }
        )

    return pairs


def build_site_data(args: argparse.Namespace) -> dict[str, Any]:
    report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    existing_data = load_existing_data()
    existing_generations = load_existing_generations(existing_data)
    api_key = args.api_key or os.getenv("BASETEN_API_KEY", "")

    samples: list[dict[str, Any]] = []
    selected_groups = report["groups"][: args.limit] if args.limit else report["groups"]
    for index, group in enumerate(selected_groups):
        group_id = group["group_id"]
        entries_by_source = {entry["source"]: entry for entry in group["entries"]}
        baselines = []
        for source in BASELINE_ORDER:
            if source not in entries_by_source:
                continue
            entry = entries_by_source[source]
            baselines.append(
                {
                    "source": source,
                    "label": "Ground Truth" if source == "gt" else source,
                    "asset": copy_asset(group_id, entry),
                    "report_scores": entry.get("scores", {}),
                }
            )

        generated_asset = f"assets/text-to-svg-production-{group_id}.svg"
        generated_path = PROJECT_ROOT / generated_asset
        generated = existing_generations.get(group_id, {})
        if generated_path.exists() and generated and not args.force:
            print(f"[{index + 1}/{len(selected_groups)}] cached generation {group_id}")
        elif args.skip_generation:
            generated = {
                "source": "text-to-svg-production",
                "label": "Text-to-SVG Production",
                "asset": None,
                "status": "not_generated",
            }
        else:
            if not api_key:
                raise RuntimeError("Set BASETEN_API_KEY or pass --api-key to generate production outputs.")
            print(f"[{index + 1}/{len(selected_groups)}] generating text-to-SVG output for {group_id}")
            result = call_text_to_svg(
                endpoint=args.endpoint,
                api_key=api_key,
                prompt=group["prompt"],
                seed=args.seed + index,
                timeout=args.timeout,
            )
            svg = result.get("svg") or ""
            if not svg:
                raise RuntimeError(f"No SVG returned for {group_id}: {result}")
            generated_path.write_text(svg, encoding="utf-8")
            generated = {
                "source": "text-to-svg-production",
                "label": "Text-to-SVG Production",
                "asset": generated_asset,
                "status": "ok",
                "svg_parse_error": result.get("svg_parse_error") or svg_parse_error(svg),
                "input_tokens": result.get("input_tokens"),
                "output_tokens": result.get("output_tokens"),
                "elapsed_seconds": round(float(result.get("elapsed_seconds", 0.0)), 3),
                "seed": args.seed + index,
            }
            print(
                "  "
                f"out={generated['output_tokens']} tokens, "
                f"{generated['elapsed_seconds']}s, "
                f"parse={generated['svg_parse_error']}"
            )

        samples.append(
            {
                "id": group_id,
                "bucket": group.get("bucket"),
                "prompt": group["prompt"],
                "generated": generated,
                "baselines": baselines,
            }
        )

    prompt_pairs = build_prompt_pairs(args=args, api_key=api_key, existing_data=existing_data)

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source_report": str(SOURCE_REPORT),
        "generation_endpoint": args.endpoint,
        "report_score_order": REPORT_SCORE_ORDER,
        "report_score_labels": REPORT_SCORE_LABELS,
        "samples": samples,
        "prompt_pairs": prompt_pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Text-to-SVG comparison website data.")
    parser.add_argument("--endpoint", default=os.getenv("BASETEN_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--api-key", default=os.getenv("BASETEN_API_KEY", ""))
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--prompt-pair-seed", type=int, default=2026052101)
    parser.add_argument("--force-prompt-pairs", action="store_true")
    parser.add_argument("--skip-prompt-pairs", action="store_true")
    args = parser.parse_args()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = build_site_data(args)
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {DATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
