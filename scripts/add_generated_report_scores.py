from __future__ import annotations

import json
import os
import sys
import gc
import argparse
from pathlib import Path
from typing import Any

import cairosvg
import torch


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "data" / "site-data.json"
RENDER_DIR = PROJECT / "assets" / "generated-score-renders"
SCORE_CACHE_PATH = RENDER_DIR / "score-cache.json"
LICA_SCORE_VARIANT = "qwen8b_epoch_3"
GENERATED_SOURCES = ["text-to-svg-base", "text-to-svg-v1", "text-to-svg-v2"]
IMSCORE_IDS = [
    "imscore_hpsv21",
    "imscore_pickscore",
    "imscore_mpsv1",
    "imscore_clipscore",
    "imscore_imagereward",
    "imscore_laion_aesthetic",
]


def force_local_caches() -> None:
    """Keep large downloaded scorer weights off the root disk."""
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


def render_svg(asset: str, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        svg = (PROJECT / asset).read_text(encoding="utf-8")
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(out_path), output_width=512, output_height=512)
    return str(out_path)


def build_groups(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for sample in data.get("samples", []):
        candidates = {}
        for source in GENERATED_SOURCES:
            item = sample.get("model_generations", {}).get(source) or {}
            if item.get("status") != "ok" or not item.get("asset"):
                continue
            render_path = RENDER_DIR / f"{sample['id']}-{source}.png"
            candidates[source] = {
                "source": source,
                "render_path": render_svg(item["asset"], render_path),
            }
        if candidates:
            groups[sample["id"]] = {"prompt": sample["prompt"], "candidates": candidates}
    for pair in data.get("prompt_pairs", []):
        candidates = {}
        for source in GENERATED_SOURCES:
            item = pair.get("model_generations", {}).get(source) or {}
            if item.get("status") != "ok" or not item.get("asset"):
                continue
            render_path = RENDER_DIR / f"{pair['id']}-{source}.png"
            candidates[source] = {
                "source": source,
                "render_path": render_svg(item["asset"], render_path),
            }
        if candidates:
            groups[pair["id"]] = {"prompt": pair["prompt"], "candidates": candidates}
    return groups


def patch_imscore_compat(add_baselines: Any) -> None:
    # imscore's MPS helper targets older transformers internals; current CLIP
    # outputs can be handled by forcing tuple-style returns after load.
    add_baselines.patch_clip_text_return_dict = lambda: None
    from transformers import BertTokenizer

    if isinstance(getattr(BertTokenizer, "additional_special_tokens_ids", None), property):
        delattr(BertTokenizer, "additional_special_tokens_ids")
    if not getattr(BertTokenizer.add_special_tokens, "_lica_patched_ids", False):
        original_add_special_tokens = BertTokenizer.add_special_tokens

        def add_special_tokens_with_ids(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_add_special_tokens(self, *args, **kwargs)
            additional_tokens = []
            if args and isinstance(args[0], dict):
                additional_tokens = args[0].get("additional_special_tokens", [])
            if not additional_tokens:
                additional_tokens = self.special_tokens_map.get("additional_special_tokens", [])
            self.__dict__["additional_special_tokens_ids"] = self.convert_tokens_to_ids(
                additional_tokens
            )
            return result

        add_special_tokens_with_ids._lica_patched_ids = True  # type: ignore[attr-defined]
        BertTokenizer.add_special_tokens = add_special_tokens_with_ids
    original_loader = add_baselines.load_imscore_model

    def load_imscore_model(spec: dict[str, Any], device: torch.device) -> Any:
        model = original_loader(spec, device)
        if spec["id"] == "imscore_mpsv1":
            xclip = model.mps.model
            original_get_text_features = xclip.get_text_features
            original_get_image_features = xclip.get_image_features

            def get_text_features_tuple(*args: Any, **kwargs: Any) -> Any:
                kwargs["return_dict"] = False
                return original_get_text_features(*args, **kwargs)

            def get_image_features_tuple(*args: Any, **kwargs: Any) -> Any:
                kwargs["return_dict"] = False
                return original_get_image_features(*args, **kwargs)

            xclip.get_text_features = get_text_features_tuple
            xclip.get_image_features = get_image_features_tuple
        if spec["id"] == "imscore_pickscore":
            original_get_image_features = model.model.get_image_features
            original_get_text_features = model.model.get_text_features

            def first_feature_tensor(output: Any, feature_name: str) -> torch.Tensor:
                if isinstance(output, torch.Tensor):
                    return output
                if hasattr(output, "image_embeds"):
                    return output.image_embeds
                if hasattr(output, "text_embeds"):
                    return output.text_embeds
                if hasattr(output, "pooler_output"):
                    return output.pooler_output
                if isinstance(output, (tuple, list)):
                    for item in output:
                        if isinstance(item, torch.Tensor):
                            return item
                raise TypeError(f"Unsupported PickScore {feature_name} feature output: {type(output)!r}")

            def get_image_features_tensor(*args: Any, **kwargs: Any) -> torch.Tensor:
                return first_feature_tensor(original_get_image_features(*args, **kwargs), "image")

            def get_text_features_tensor(*args: Any, **kwargs: Any) -> torch.Tensor:
                return first_feature_tensor(original_get_text_features(*args, **kwargs), "text")

            model.model.get_image_features = get_image_features_tensor
            model.model.get_text_features = get_text_features_tensor
        return model

    add_baselines.load_imscore_model = load_imscore_model


def apply_scores_to_payload(payload: dict[str, Any], group_id: str, all_scores: dict[str, dict[str, dict[str, float]]]) -> None:
    for source in GENERATED_SOURCES:
        item = payload.get("model_generations", {}).get(source) or {}
        if item.get("status") != "ok":
            continue
        report_scores = dict(item.get("report_scores") or {})
        lica_score = payload.get("lica_scores", {}).get(source, {}).get("score")
        if lica_score is not None:
            report_scores[LICA_SCORE_VARIANT] = float(lica_score)
        for score_id, score_by_group in all_scores.items():
            value = score_by_group.get(group_id, {}).get(source)
            if value is not None:
                report_scores[score_id] = float(value)
        item["report_scores"] = {
            key: report_scores[key]
            for key in payload.get("report_score_order", [])
            if key in report_scores
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merge-cache-only",
        action="store_true",
        help="Only merge existing score-cache.json into site-data.json without running missing scorers.",
    )
    args = parser.parse_args()
    force_local_caches()
    sys.path.insert(0, "/home/ubuntu/lica-score-web")
    import add_baselines
    patch_imscore_compat(add_baselines)

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    groups = build_groups(data)
    if not groups:
        raise RuntimeError("No generated model candidates found to score.")

    imscore_specs = []
    for spec in add_baselines.IMSCORE_MODELS:
        if spec["id"] not in set(IMSCORE_IDS):
            continue
        spec = dict(spec)
        imscore_specs.append(spec)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if SCORE_CACHE_PATH.exists():
        all_scores: dict[str, dict[str, dict[str, float]]] = json.loads(
            SCORE_CACHE_PATH.read_text(encoding="utf-8")
        )
    else:
        all_scores = {}

    if not args.merge_cache_only:
        for spec in imscore_specs:
            if spec["id"] in all_scores:
                print(f"[score] {spec['id']} cached", flush=True)
                continue
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[score] {spec['id']} on {len(groups)} groups", flush=True)
            all_scores[spec["id"]] = add_baselines.score_imscore_model(spec, groups, device)
            SCORE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            SCORE_CACHE_PATH.write_text(json.dumps(all_scores, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for sample in data.get("samples", []):
        sample["report_score_order"] = data.get("report_score_order", [])
        apply_scores_to_payload(sample, sample["id"], all_scores)
        sample.pop("report_score_order", None)
    for pair in data.get("prompt_pairs", []):
        pair["report_score_order"] = data.get("report_score_order", [])
        apply_scores_to_payload(pair, pair["id"], all_scores)
        pair.pop("report_score_order", None)

    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("updated", DATA_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
