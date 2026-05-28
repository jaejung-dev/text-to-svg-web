from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "data" / "site-data.json"
LOCAL_ROOT = Path("/mnt/local")
LORA_PATH = LOCAL_ROOT / "lora-only-20260527-150651"
BASE_MODEL = "Qwen/Qwen3.5-35B-A3B"
SOURCE_ID = "hf-v2"
SETTINGS_LABEL = "HF PEFT v2 local defaults (temperature 0.9, top_p 0.95, top_k 50)"
SYSTEM_PROMPT = (
    "You are an expert SVG code generator. Given a description of an image, "
    "generate clean, well-structured SVG code with separate elements for each "
    "visual concept. Use descriptive group IDs and organize elements logically. "
    "Output only the SVG code between <svg> and </svg> tags."
)
SVG_RE = re.compile(r"<svg\b[\s\S]*?</svg>", re.IGNORECASE)


def force_local_caches() -> None:
    cache_env = {
        "HF_HOME": LOCAL_ROOT / "hf-cache",
        "HF_HUB_CACHE": LOCAL_ROOT / "hf-cache" / "hub",
        "HUGGINGFACE_HUB_CACHE": LOCAL_ROOT / "hf-cache" / "hub",
        "TRANSFORMERS_CACHE": LOCAL_ROOT / "hf-cache" / "transformers",
        "TORCH_HOME": LOCAL_ROOT / "torch-cache",
        "XDG_CACHE_HOME": LOCAL_ROOT / "xdg-cache",
        "PIP_CACHE_DIR": LOCAL_ROOT / "pip-cache",
    }
    for key, path in cache_env.items():
        os.environ[key] = str(path)
        path.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def asset_path(pair_id: str) -> str:
    return f"assets/{SOURCE_ID}-{pair_id}.svg"


def svg_parse_error(svg: str) -> str | None:
    try:
        ElementTree.fromstring(svg)
    except ElementTree.ParseError as exc:
        return str(exc)
    return None


def extract_svg(completion: str) -> str:
    final_text = completion.rsplit("</think>", 1)[-1] if "</think>" in completion else completion
    match = SVG_RE.search(final_text)
    if match:
        return match.group(0).strip()
    return ""


def build_prompt(pair: dict[str, Any], tokenizer: Any) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Generate SVG code for: {pair['prompt']}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def complete_generation(item: dict[str, Any] | None) -> bool:
    return bool(
        item
        and item.get("status") == "ok"
        and item.get("asset")
        and (PROJECT / str(item["asset"])).exists()
        and item.get("source") == SOURCE_ID
    )


def save_data(data: dict[str, Any]) -> None:
    data["generated_at"] = utc_now()
    data["hf_v2_settings"] = {
        "label": "HF-v2",
        "source": SOURCE_ID,
        "base_model": BASE_MODEL,
        "lora_path": str(LORA_PATH),
        "settings": SETTINGS_LABEL,
    }
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_item(
    *,
    pair: dict[str, Any],
    svg: str,
    completion: str,
    input_tokens: int,
    output_tokens: int,
    elapsed_seconds: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    asset = asset_path(pair["id"])
    if svg:
        (PROJECT / asset).write_text(svg, encoding="utf-8")
    parse_error = svg_parse_error(svg) if svg else "Model output did not contain a final <svg>...</svg> block."
    return {
        "source": SOURCE_ID,
        "label": "HF-v2",
        "asset": asset if svg else None,
        "status": "ok" if svg and not parse_error else ("invalid_svg" if svg else "no_svg"),
        "base_model": BASE_MODEL,
        "lora_path": str(LORA_PATH),
        "settings": SETTINGS_LABEL,
        "max_new_tokens": max_new_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "svg_parse_error": parse_error,
        "svg_chars": len(svg),
        "sha256": hashlib.sha256(svg.encode("utf-8")).hexdigest() if svg else None,
        "engine": "hf-peft-local",
        "raw_text_prefix": completion[:500] if not svg else None,
    }


def generate_batch(
    *,
    pairs: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    texts = [build_prompt(pair, tokenizer) for pair in pairs]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    input_width = int(inputs["input_ids"].shape[1])
    input_token_counts = [int(count) for count in inputs["attention_mask"].sum(dim=1).tolist()]

    started = time.time()
    import torch

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
    for offset, pair in enumerate(pairs):
        generated_ids = output_ids[offset, input_width:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        svg = extract_svg(completion)
        item = make_item(
            pair=pair,
            svg=svg,
            completion=completion,
            input_tokens=input_token_counts[offset],
            output_tokens=int(generated_ids.shape[0]),
            elapsed_seconds=elapsed,
            max_new_tokens=max_new_tokens,
        )
        results.append((pair, item))
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    force_local_caches()
    if not LORA_PATH.exists():
        raise FileNotFoundError(f"Missing LoRA adapter at {LORA_PATH}")

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    pairs = []
    for pair in data.get("prompt_pairs", []):
        generations = pair.setdefault("model_generations", {})
        if not args.force and complete_generation(generations.get(SOURCE_ID)):
            continue
        pairs.append(pair)
    save_data(data)
    print(f"HF-v2 tasks remaining: {len(pairs)}", flush=True)
    if not pairs:
        return 0

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        labels = ", ".join(str(pair["index"]).zfill(2) for pair in batch)
        print(f"HF-v2 batch start: {labels}", flush=True)
        try:
            results = generate_batch(
                pairs=batch,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=args.max_new_tokens,
            )
        except RuntimeError as exc:
            if len(batch) == 1 or "out of memory" not in str(exc).lower():
                raise
            print(f"HF-v2 batch OOM, retrying individually: {labels}", flush=True)
            torch.cuda.empty_cache()
            results = []
            for single in batch:
                results.extend(
                    generate_batch(
                        pairs=[single],
                        tokenizer=tokenizer,
                        model=model,
                        max_new_tokens=args.max_new_tokens,
                    )
                )

        for pair, item in results:
            pair.setdefault("model_generations", {})[SOURCE_ID] = item
            save_data(data)
            completed += 1
            print(
                f"[{int(pair['index']):02d}] HF-v2: {item['status']} "
                f"tokens={item['output_tokens']} elapsed={item['elapsed_seconds']} "
                f"parse={item['svg_parse_error']} ({completed}/{len(pairs)})",
                flush=True,
            )

    print(f"done {DATA_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
