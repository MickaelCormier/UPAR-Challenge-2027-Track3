from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
PRIOR_SOURCE = (HERE.parent / "sample_code_submission" / "assets" / "mean_pose.json")

# COCO is dataset index 0 in the ViTPose+ mixture-of-experts head
COCO_DATASET_INDEX = 0


class CocoWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, dataset_index: int) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("dataset_index", torch.tensor([dataset_index], dtype=torch.int64))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        index = self.dataset_index.expand(pixel_values.shape[0])
        return self.model(pixel_values=pixel_values, dataset_index=index).heatmaps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="usyd-community/vitpose-plus-small",
                        help="HF checkpoint (default: vitpose-plus-small, 126 MB)")
    parser.add_argument("--out", type=Path, default=ASSETS / "vitpose.torchscript")
    args = parser.parse_args()

    try:
        from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor
    except ImportError:
        print("This export step needs transformers: pip install transformers")
        return 1

    ASSETS.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    print(f"[*] loading {args.model} ...")
    model = VitPoseForPoseEstimation.from_pretrained(args.model).eval()
    processor = VitPoseImageProcessor.from_pretrained(args.model)

    size = processor.size
    height, width = int(size["height"]), int(size["width"])
    wrapper = CocoWrapper(model, COCO_DATASET_INDEX).eval()

    example = torch.zeros(1, 3, height, width)
    print(f"[*] tracing with input {tuple(example.shape)} ...")
    traced = torch.jit.trace(wrapper, example, strict=False)
    traced = torch.jit.freeze(traced)
    traced.save(args.out)
    print(f"[*] saved {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")

    meta = {
        "source_checkpoint": args.model,
        "dataset_index": COCO_DATASET_INDEX,
        "input_height": height,
        "input_width": width,
        "image_mean": list(processor.image_mean),
        "image_std": list(processor.image_std),
        "rescale_factor": float(processor.rescale_factor),
        "normalize_factor": float(getattr(processor, "normalize_factor", 200.0)),
        "padding_factor": 1.25,
        "num_keypoints": int(model.config.num_labels),
        "heatmap_scale_factor": int(model.config.scale_factor),
    }
    (ASSETS / "preprocess.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[*] wrote {ASSETS / 'preprocess.json'}")

    if PRIOR_SOURCE.is_file():
        (ASSETS / "mean_pose.json").write_bytes(PRIOR_SOURCE.read_bytes())
        print(f"[*] copied fallback prior from {PRIOR_SOURCE.name}")

    reloaded = torch.jit.load(args.out, map_location="cpu")
    out = reloaded(example)
    print(f"[*] reload check: heatmaps {tuple(out.shape)}")

    total = sum(p.stat().st_size for p in ASSETS.rglob("*") if p.is_file()) / 1e6
    print(f"\n[OK] assets/ holds {total:.0f} MB. The submission is ready to zip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
