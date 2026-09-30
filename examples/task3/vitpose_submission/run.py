from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "assets", "vitpose.torchscript")
META_PATH = os.path.join(HERE, "assets", "preprocess.json")
PRIOR_PATH = os.path.join(HERE, "assets", "mean_pose.json")

BATCH_SIZE = 8
DARK_KERNEL = 11

_MODEL = None
_DEVICE = None
_META: dict[str, Any] = {}
_PRIOR: list[list[float]] | None = None
_MEAN: np.ndarray | None = None
_STD: np.ndarray | None = None


def load_model() -> None:
    global _MODEL, _DEVICE, _META, _PRIOR, _MEAN, _STD
    if _MODEL is not None:
        return

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} is missing. Run `python export_model.py` before zipping: "
            "the evaluation container cannot download anything."
        )

    with open(META_PATH, "r", encoding="utf-8") as fh:
        _META = json.load(fh)
    with open(PRIOR_PATH, "r", encoding="utf-8") as fh:
        _PRIOR = json.load(fh)["mean_pose_normalized"]

    _MEAN = np.asarray(_META["image_mean"], dtype=np.float32).reshape(3, 1, 1)
    _STD = np.asarray(_META["image_std"], dtype=np.float32).reshape(3, 1, 1)

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    _MODEL = torch.jit.load(MODEL_PATH, map_location=_DEVICE)
    _MODEL.eval()
    print(f"[submission] ViTPose ready on {_DEVICE} "
          f"({_META['input_width']}x{_META['input_height']} input)", flush=True)


def box_to_center_scale(box, out_w: int, out_h: int,
                        normalize_factor: float, padding_factor: float):
    x, y, w, h = [float(v) for v in box[:4]]
    aspect = out_w / out_h
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect
    scale = np.array([w / normalize_factor, h / normalize_factor], dtype=np.float32)
    return center, scale * padding_factor


def warp_matrix(center: np.ndarray, scale: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    size_input = center * 2.0
    size_dst = np.array([out_w, out_h], dtype=np.float32) - 1.0
    size_target = scale * 200.0
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[0, 0] = scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] + 0.5 * size_target[0])
    matrix[1, 1] = scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[1] + 0.5 * size_target[1])
    return matrix


def warp_affine(image: np.ndarray, matrix: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    bottom = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    inverse = np.linalg.inv(np.vstack([matrix, bottom]))[:2].astype(np.float32)
    ys, xs = np.meshgrid(np.arange(out_h, dtype=np.float32),
                         np.arange(out_w, dtype=np.float32), indexing="ij")
    src_x = inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]
    src_y = inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]

    height, width = image.shape[:2]
    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = src_x - x0, src_y - y0

    inside = (src_x >= 0) & (src_x <= width - 1) & (src_y >= 0) & (src_y <= height - 1)
    x0c, x1c = np.clip(x0, 0, width - 1), np.clip(x1, 0, width - 1)
    y0c, y1c = np.clip(y0, 0, height - 1), np.clip(y1, 0, height - 1)

    top = image[y0c, x0c] * (1 - wx)[..., None] + image[y0c, x1c] * wx[..., None]
    bottom = image[y1c, x0c] * (1 - wx)[..., None] + image[y1c, x1c] * wx[..., None]
    return (top * (1 - wy)[..., None] + bottom * wy[..., None]) * inside[..., None]


def preprocess(image: np.ndarray, box) -> tuple:
    out_w, out_h = int(_META["input_width"]), int(_META["input_height"])
    center, scale = box_to_center_scale(
        box, out_w, out_h, float(_META["normalize_factor"]), float(_META["padding_factor"]))
    crop = warp_affine(image, warp_matrix(center, scale, out_w, out_h), out_h, out_w)
    tensor = crop.transpose(2, 0, 1) * float(_META["rescale_factor"])
    return (((tensor - _MEAN) / _STD).astype(np.float32), center, scale)


def _gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 / (sigma * sigma) * x * x)
    return (kernel / kernel.sum()).astype(np.float32)


def _gaussian_blur(heatmaps: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = (kernel_size - 1) // 2
    kernel = _gaussian_kernel1d(0.8, radius)
    padded = np.pad(heatmaps, ((0, 0), (0, 0), (radius, radius), (radius, radius)),
                    mode="symmetric")
    blurred = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 2, padded)
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 3, blurred)


def decode_heatmaps(heatmaps: np.ndarray, centers: np.ndarray, scales: np.ndarray):
    batch, num_kpts, height, width = heatmaps.shape
    flat = heatmaps.reshape(batch, num_kpts, -1)
    idx = np.argmax(flat, axis=2)
    scores = np.max(flat, axis=2)

    coords = np.zeros((batch, num_kpts, 2), dtype=np.float32)
    coords[:, :, 0] = idx % width
    coords[:, :, 1] = idx // width
    coords = np.where((scores > 0.0)[..., None], coords, -1.0)

    blurred = np.clip(_gaussian_blur(heatmaps, DARK_KERNEL), 0.001, 50.0)
    log_hm = np.log(blurred)
    padded = np.pad(log_hm, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge").reshape(-1)

    stride = (width + 2) * (height + 2)
    base = (coords[..., 0] + 1 + (coords[..., 1] + 1) * (width + 2))
    base = base + stride * np.arange(batch * num_kpts).reshape(batch, num_kpts)
    index = base.astype(np.int64).reshape(-1, 1)

    center_v = padded[index]
    x_plus = padded[index + 1]
    x_minus = padded[index - 1]
    y_plus = padded[index + width + 2]
    y_minus = padded[index - 2 - width]
    xy_plus = padded[index + width + 3]
    xy_minus = padded[index - width - 3]

    dx = 0.5 * (x_plus - x_minus)
    dy = 0.5 * (y_plus - y_minus)
    derivative = np.concatenate([dx, dy], axis=1).reshape(batch, num_kpts, 2, 1)

    dxx = x_plus - 2 * center_v + x_minus
    dyy = y_plus - 2 * center_v + y_minus
    dxy = 0.5 * (xy_plus - x_plus - y_plus + center_v + center_v - x_minus - y_minus + xy_minus)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(batch, num_kpts, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords = coords - np.einsum("ijmn,ijnk->ijmk", hessian, derivative).squeeze(-1)

    out = np.zeros_like(coords)
    for i in range(batch):
        scale = scales[i] * 200.0
        out[i, :, 0] = coords[i, :, 0] * scale[0] / (width - 1.0) + centers[i][0] - scale[0] * 0.5
        out[i, :, 1] = coords[i, :, 1] * scale[1] / (height - 1.0) + centers[i][1] - scale[1] * 0.5
    return out, scores


def _prior_pose(box) -> list[list[float]]:
    bx, by, bw, bh = [float(v) for v in box]
    return [[bx + u * bw, by + v * bh, 1.0] for u, v in _PRIOR]


def _resolve_image_path(sample: dict[str, Any]) -> str:
    image_path = str(sample.get("image_path", ""))
    if image_path and os.path.isfile(image_path):
        return image_path

    file_name = str(sample.get("file_name", ""))
    if image_path:
        candidates = [
            image_path.replace("/images/images/", "/images/"),
            image_path.replace("\\images\\images\\", "\\images\\"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

    if file_name:
        candidates = [
            file_name,
            file_name.removeprefix("images/"),
            file_name.removeprefix("images\\"),
        ]
        for candidate in candidates:
            candidate_path = os.path.join(HERE, candidate)
            if os.path.isfile(candidate_path):
                return candidate_path

    if image_path:
        return image_path
    raise FileNotFoundError("Could not resolve image path for sample.")


def predict_image(sample: dict[str, Any]) -> dict[str, Any]:
    if _MODEL is None:
        load_model()

    boxes = sample["boxes"]
    if not boxes:
        return {"keypoints": [], "scores": []}

    image = np.asarray(Image.open(_resolve_image_path(sample)).convert("RGB"), dtype=np.float32)

    keypoints: list[Any] = [None] * len(boxes)
    scores: list[float] = [0.0] * len(boxes)

    usable = [i for i, b in enumerate(boxes) if float(b[2]) > 0 and float(b[3]) > 0]
    for i in set(range(len(boxes))) - set(usable):
        keypoints[i] = _prior_pose(boxes[i])
        scores[i] = 0.01

    for start in range(0, len(usable), BATCH_SIZE):
        chunk = usable[start:start + BATCH_SIZE]
        tensors, centers, scales = [], [], []
        for i in chunk:
            tensor, center, scale = preprocess(image, boxes[i])
            tensors.append(tensor)
            centers.append(center)
            scales.append(scale)

        batch = torch.from_numpy(np.stack(tensors)).to(_DEVICE)
        heatmaps = _MODEL(batch).float().cpu().numpy()
        coords, joint_scores = decode_heatmaps(heatmaps, np.stack(centers), np.stack(scales))

        for j, i in enumerate(chunk):
            keypoints[i] = [[float(x), float(y), 1.0] for x, y in coords[j]]
            scores[i] = float(np.mean(joint_scores[j]))

    return {"keypoints": keypoints, "scores": scores}
