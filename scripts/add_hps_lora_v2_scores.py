from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cairosvg
import torch


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "data" / "site-data.json"
RENDER_DIR = PROJECT / "assets" / "hps-lora-v2-renders"
LICA_SCORE_V2_ID = "lica_hps_lora_caption_epoch_2"
LICA_SCORE_V2_LABEL = "LicaScore v2"
DEFAULT_CHECKPOINT = Path(
    "/mnt/local/lica-score-runs/hps-lora-caption-20260528-091359/"
    "hps_lora_caption_epoch_002.pt"
)
LICA_SCORE_SRC = Path("/home/ubuntu/ml-platform/other-projects/lica-score/src")


def force_local_caches() -> None:
    cache_root = Path("/mnt/local")
    cache_env = {
        "HF_HOME": cache_root / "hf-cache",
        "HF_HUB_CACHE": cache_root / "hf-cache" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hf-cache" / "hub",
        "TRANSFORMERS_CACHE": cache_root / "hf-cache" / "transformers",
        "TORCH_HOME": cache_root / "torch-cache",
        "XDG_CACHE_HOME": cache_root / "xdg-cache",
    }
    for key, path in cache_env.items():
        os.environ[key] = str(path)
        path.mkdir(parents=True, exist_ok=True)


def ensure_score_metadata(data: dict[str, Any]) -> None:
    order = list(data.get("report_score_order") or [])
    if LICA_SCORE_V2_ID not in order:
        try:
            insert_at = order.index("qwen8b_epoch_3") + 1
        except ValueError:
            insert_at = 0
        order.insert(insert_at, LICA_SCORE_V2_ID)
    data["report_score_order"] = order
    labels = dict(data.get("report_score_labels") or {})
    labels[LICA_SCORE_V2_ID] = LICA_SCORE_V2_LABEL
    data["report_score_labels"] = labels


def render_svg(asset: str) -> str:
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


def image_path_for_asset(asset: str) -> str:
    if asset.lower().endswith(".svg"):
        return render_svg(asset)
    return str(PROJECT / asset)


def candidate_items(payload: dict[str, Any], *, include_baselines: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if include_baselines:
        for item in payload.get("baselines", []):
            if item.get("asset"):
                rows.append(item)
    for item in (payload.get("model_generations") or {}).values():
        if item.get("status") == "ok" and item.get("asset"):
            rows.append(item)
    return rows


def source_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = {item["source"]: item for item in payload.get("baselines", []) if item.get("source")}
    if payload.get("generated"):
        items["text-to-svg-production"] = payload["generated"]
    items.update(payload.get("model_generations") or {})
    return items


def update_report_score_winners(payload: dict[str, Any], sources: list[str], score_order: list[str]) -> None:
    items = source_map(payload)
    winners = {}
    for score_id in score_order:
        best = None
        for source in sources:
            item = items.get(source) or {}
            value = (item.get("report_scores") or {}).get(score_id)
            if value is None:
                continue
            score = float(value)
            if best is None or score > best["score"]:
                best = {"source": source, "score": score}
        if best is not None:
            winners[score_id] = best
    payload["report_score_winners"] = winners
    if "qwen8b_epoch_3" in winners:
        payload["lica_winner"] = winners["qwen8b_epoch_3"]["source"]


def load_hps_lora_model(checkpoint_path: Path, device: torch.device) -> tuple[Any, str]:
    force_local_caches()
    sys.path.insert(0, str(LICA_SCORE_SRC))
    from imscore.hps.model import HPSv2
    from lica_score.train_hps_lora import apply_hps_lora

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = dict(checkpoint["args"])
    checkpoint_args["device"] = str(device)
    args = SimpleNamespace(**checkpoint_args)

    model = HPSv2.from_pretrained(args.hps_pretrained_id).to(device)
    apply_hps_lora(model, args)
    missing, unexpected = model.load_state_dict(checkpoint["trainable_state"], strict=False)
    unexpected = [key for key in unexpected if key]
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:8]}")
    # Missing keys are expected because the checkpoint stores only trainable LoRA/logit-scale weights.
    del missing
    model.eval()
    return model, args.precision


@torch.no_grad()
def score_rows(
    model: Any,
    rows: list[tuple[dict[str, Any], str, str]],
    *,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> None:
    from lica_score.train_hps_lora import load_render_tensor, score_candidate_batch

    for index in range(0, len(rows), batch_size):
        chunk = rows[index : index + batch_size]
        tensors = []
        for _item, _prompt, path in chunk:
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
        prompts = [prompt for _item, prompt, _path in chunk]
        scores = score_candidate_batch(
            model,
            images,
            prompts,
            device=device,
            precision=precision,
        ).tolist()
        for (item, _prompt, _path), score in zip(chunk, scores):
            report_scores = dict(item.get("report_scores") or {})
            report_scores[LICA_SCORE_V2_ID] = float(score)
            item["report_scores"] = report_scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Add HPS-LoRA epoch 2 scores to text-to-svg-web.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    ensure_score_metadata(data)
    rows: list[tuple[dict[str, Any], str, str]] = []

    for sample in data.get("samples", []):
        for item in candidate_items(sample, include_baselines=True):
            if not args.force and (item.get("report_scores") or {}).get(LICA_SCORE_V2_ID) is not None:
                continue
            rows.append((item, sample["prompt"], image_path_for_asset(item["asset"])))
    for pair in data.get("prompt_pairs", []):
        for item in candidate_items(pair, include_baselines=False):
            if not args.force and (item.get("report_scores") or {}).get(LICA_SCORE_V2_ID) is not None:
                continue
            rows.append((item, pair["prompt"], image_path_for_asset(item["asset"])))

    print(f"[hps-lora-v2] rows to score: {len(rows)}", flush=True)
    if rows:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, precision = load_hps_lora_model(args.checkpoint, device)
        score_rows(
            model,
            rows,
            batch_size=max(1, args.batch_size),
            device=device,
            precision=precision,
        )

    score_order = data.get("report_score_order", [])
    validation_sources = [
        "text-to-svg-base",
        "text-to-svg-v1",
        "text-to-svg-v2",
        "claude",
        "gemini",
        "gpt-5.2",
    ]
    prompt_pair_sources = ["text-to-svg-base", "text-to-svg-v1", "text-to-svg-v2", "hf-v2"]
    for sample in data.get("samples", []):
        update_report_score_winners(sample, validation_sources, score_order)
    for pair in data.get("prompt_pairs", []):
        update_report_score_winners(pair, prompt_pair_sources, score_order)

    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("updated", DATA_PATH, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
