from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "data" / "site-data.json"
LORA_PATH = Path("/home/ubuntu/lora-only")
BASE_MODEL = "Qwen/Qwen3.5-35B-A3B"
RUNS_PER_PROMPT = 2
SETTINGS_LABEL = "HF PEFT local default sampling (temperature 0.9, top_p 0.95, top_k 50)"
SYSTEM_PROMPT = (
    "You are an expert SVG code generator. Given a description of an image, "
    "generate clean, well-structured SVG code with separate elements for each "
    "visual concept. Use descriptive group IDs and organize elements logically. "
    "Output only the SVG code between <svg> and </svg> tags."
)
SVG_RE = re.compile(r"<svg\b[\s\S]*?</svg>", re.IGNORECASE)


def asset_path(index: int, run: int) -> str:
    return f"assets/hf-lora-local-prompt-pair-{index:02d}-{run:02d}.svg"


def svg_parse_error(svg: str) -> str | None:
    try:
        ElementTree.fromstring(svg)
    except ElementTree.ParseError as exc:
        return str(exc)
    return None


def extract_svg(completion: str) -> str:
    # Thinking mode can mention SVG tags in reasoning. Only extract from the
    # final answer after </think> when present.
    final_text = completion.rsplit("</think>", 1)[-1] if "</think>" in completion else completion
    match = SVG_RE.search(final_text)
    if match:
        return match.group(0).strip()
    return ""


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
    grid = pair.setdefault("hf_lora_local", {})
    grid["label"] = "HF LoRA local"
    grid["settings"] = SETTINGS_LABEL
    grid["runs_per_prompt"] = RUNS_PER_PROMPT
    grid["base_model"] = BASE_MODEL
    grid["lora_path"] = str(LORA_PATH)
    grid["no_stop_constraint"] = True
    grid.setdefault("runs", [])
    return grid


def register_existing_asset(pair: dict[str, Any], run: int) -> dict[str, Any] | None:
    index = int(pair["index"])
    asset = asset_path(index, run)
    path = PROJECT / asset
    if not path.exists():
        return None
    svg = path.read_text(encoding="utf-8")
    return {
        "source": "hf-lora-local",
        "label": f"HF LoRA local #{run}",
        "asset": asset,
        "status": "ok" if svg else "no_svg",
        "run": run,
        "settings": SETTINGS_LABEL,
        "stop_constraint": None,
        "input_tokens": None,
        "output_tokens": None,
        "elapsed_seconds": None,
        "svg_parse_error": svg_parse_error(svg) if svg else "Empty SVG file.",
        "svg_chars": len(svg),
        "sha256": hashlib.sha256(svg.encode("utf-8")).hexdigest(),
        "cached": True,
    }


def save_data(data: dict[str, Any]) -> None:
    data["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data["hf_lora_local_settings"] = SETTINGS_LABEL
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_prompt(pair: dict[str, Any], tokenizer: Any) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Generate SVG code for: {pair['prompt']}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def update_item(pair: dict[str, Any], run: int, item: dict[str, Any]) -> None:
    grid = ensure_grid(pair)
    by_run = existing_by_run(grid["runs"])
    by_run[run] = item
    grid["runs"] = [by_run[index] for index in sorted(by_run)]


def make_item(
    *,
    index: int,
    run: int,
    svg: str,
    completion: str,
    input_tokens: int,
    output_tokens: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    asset = asset_path(index, run)
    if svg:
        (PROJECT / asset).write_text(svg, encoding="utf-8")
    return {
        "source": "hf-lora-local",
        "label": f"HF LoRA local #{run}",
        "asset": asset if svg else None,
        "status": "ok" if svg else "no_svg",
        "run": run,
        "settings": SETTINGS_LABEL,
        "stop_constraint": None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "svg_parse_error": svg_parse_error(svg) if svg else "Model output did not contain a final <svg>...</svg> block.",
        "svg_chars": len(svg),
        "sha256": hashlib.sha256(svg.encode("utf-8")).hexdigest(),
        "raw_text_prefix": completion[:500] if not svg else None,
    }


def generate_batch(
    *,
    specs: list[tuple[dict[str, Any], int]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
) -> list[tuple[dict[str, Any], int, dict[str, Any]]]:
    texts = [build_prompt(pair, tokenizer) for pair, _run in specs]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    input_width = int(inputs["input_ids"].shape[1])
    input_token_counts = [int(count) for count in inputs["attention_mask"].sum(dim=1).tolist()]

    started = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            top_k=50,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - started

    results = []
    for offset, (pair, run) in enumerate(specs):
        generated_ids = output_ids[offset, input_width:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        svg = extract_svg(completion)
        item = make_item(
            index=int(pair["index"]),
            run=run,
            svg=svg,
            completion=completion,
            input_tokens=input_token_counts[offset],
            output_tokens=int(generated_ids.shape[0]),
            elapsed_seconds=elapsed,
        )
        results.append((pair, run, item))
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    specs: list[tuple[dict[str, Any], int]] = []
    for pair in data.get("prompt_pairs", []):
        grid = ensure_grid(pair)
        by_run = existing_by_run(grid["runs"])
        for run in range(1, RUNS_PER_PROMPT + 1):
            if not by_run.get(run):
                cached = register_existing_asset(pair, run)
                if cached:
                    by_run[run] = cached
                    grid["runs"] = [by_run[index] for index in sorted(by_run)]
            if not args.force and is_complete(by_run.get(run)):
                continue
            specs.append((pair, run))
    save_data(data)

    print(f"tasks remaining: {len(specs)}", flush=True)
    if not specs:
        return 0

    print(f"loading tokenizer {BASE_MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"loading base model {BASE_MODEL}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"applying LoRA {LORA_PATH}", flush=True)
    model = PeftModel.from_pretrained(model, str(LORA_PATH)).eval()

    completed = 0
    batch_size = max(1, args.batch_size)
    for start in range(0, len(specs), batch_size):
        batch = specs[start : start + batch_size]
        labels = ", ".join(f"{pair['index']:02d}#{run}" for pair, run in batch)
        print(f"batch start: {labels}", flush=True)
        try:
            results = generate_batch(
                specs=batch,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=args.max_new_tokens,
            )
        except RuntimeError as exc:
            if len(batch) == 1 or "out of memory" not in str(exc).lower():
                raise
            print(f"batch OOM, retrying individually: {labels}", flush=True)
            torch.cuda.empty_cache()
            results = []
            for single in batch:
                results.extend(
                    generate_batch(
                        specs=[single],
                        tokenizer=tokenizer,
                        model=model,
                        max_new_tokens=args.max_new_tokens,
                    )
                )

        for pair, run, item in results:
            update_item(pair, run, item)
            save_data(data)
            completed += 1
            print(
                f"[{int(pair['index']):02d}] HF run {run}: {item['status']} "
                f"tokens={item['output_tokens']} elapsed={item['elapsed_seconds']} "
                f"parse={item['svg_parse_error']} ({completed}/{len(specs)})",
                flush=True,
            )

    print(f"done {DATA_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
