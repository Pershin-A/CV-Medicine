"""Run the trained image-to-geometry-to-quality DXA pipeline on a DICOM."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from dxa_project.augmentation.core import (
    hip_position_ok, hip_roi_margins_mm, hip_roi_ok, physical_axis_angle,
    spine_axis_angle,
    spine_position_ok,
)
from dxa_project.augmentation.generate import SCANNER_NOMINAL_SPACING_MM, _dicom_spacing
from dxa_project.geometry_ml.data import CRESTS, LANDMARKS, REGIONS, read_dicom_image
from dxa_project.geometry_ml.models import (
    Router, SpatialNet, artifact_detector, detector_images,
)
from labeler.geometry import empty_geometry
from dxa_project.augmentation.vertebral_axes import analyze_spine, placement_from_axes


def prepare_input(image: np.ndarray, size: int, flipped: bool = False,
                  stretch: bool = False):
    h, w = image.shape
    if flipped:
        image = np.fliplr(image)
    scale = min(size / w, size / h)
    dw, dh = round(w * scale), round(h * scale)
    left, top = (size - dw) // 2, (size - dh) // 2
    canvas = Image.new("L", (size, size))
    if stretch:
        canvas.paste(Image.fromarray((image * 255).astype(np.uint8)).resize(
            (size, size), Image.Resampling.BILINEAR))
    else:
        canvas.paste(Image.fromarray((image * 255).astype(np.uint8)).resize(
            (dw, dh), Image.Resampling.BILINEAR), (left, top))
    tensor = torch.from_numpy(np.repeat(np.asarray(canvas, np.float32)[None], 3, axis=0) / 255)
    tensor = ((tensor - torch.tensor([.485, .456, .406])[:, None, None]) /
              torch.tensor([.229, .224, .225])[:, None, None])
    return tensor, {"width": w, "height": h, "left": left, "top": top,
                    "scale": scale, "size": size, "flipped": flipped}


def _original_point(x: float, y: float, meta: dict):
    px = (x - meta["left"]) / meta["scale"]
    py = (y - meta["top"]) / meta["scale"]
    if meta["flipped"]:
        px = meta["width"] - 1 - px
    return [float(np.clip(px, 0, meta["width"] - 1)),
            float(np.clip(py, 0, meta["height"] - 1))]


def _heatmap_point(heatmap, meta):
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return _original_point(x, y, meta), float(heatmap[y, x])


def _spine_lines(probability: np.ndarray, meta: dict) -> list[dict]:
    """Peak rows followed by weighted straight-line fits in the line map."""
    left, top = meta["left"], meta["top"]
    dw = round(meta["width"] * meta["scale"])
    dh = round(meta["height"] * meta["scale"])
    core = probability[top:top + dh, left:left + dw]
    profile = gaussian_filter1d(core[:, dw // 4:3 * dw // 4].mean(axis=1), 1.2)
    prominence = max(0.08, float(profile.max()) * .2)
    peaks, properties = find_peaks(profile, prominence=prominence,
                                   distance=max(4, round(8 * meta["scale"])))
    order = np.argsort(properties["prominences"])[::-1][:7]
    lines = []
    for index in sorted(peaks[order]):
        xs = np.arange(dw, dtype=float)
        ys = []
        weights = []
        for x in range(dw):
            lo, hi = max(0, index - 7), min(dh, index + 8)
            column = core[lo:hi, x]
            local = int(np.argmax(column)) + lo
            ys.append(local)
            weights.append(float(core[local, x]))
        weights = np.asarray(weights)
        if (weights > .5).sum() < .2 * dw:
            continue
        slope, intercept = np.polyfit(xs, ys, 1, w=np.maximum(weights, 1e-3))
        a = _original_point(left, top + intercept, meta)
        b = _original_point(left + dw - 1, top + slope * (dw - 1) + intercept, meta)
        if abs(a[1] - b[1]) > .25 * meta["height"]:
            continue
        lines.append({"id": f"predicted_{len(lines)}", "points": [a, b]})
    return lines


@lru_cache(maxsize=8)
def _load_model(task: str, checkpoint_dir: Path, device):
    checkpoint = torch.load(checkpoint_dir / f"{task}.pt", map_location="cpu",
                            weights_only=False)
    model = (Router(False) if task == "router" else
             SpatialNet("SPINE" if task == "spine" else "HIP", False,
                        'resnet50' if checkpoint.get('architecture')=='heavy' else 'resnet18')
             if task in ("spine", "hip") else artifact_detector(False,heavy=checkpoint.get('architecture')=='heavy'))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, int(checkpoint["size"])


@torch.no_grad()
def predict_file(path: Path, checkpoint_dir: Path,
                 rotation_area_threshold_px2: float | None = None,
                 force_region: str | None = None,spacing_override_mm=None) -> dict:
    if rotation_area_threshold_px2 is None:
        calibration = checkpoint_dir / 'evaluation' / 'calibration.json'
        rotation_area_threshold_px2 = (float(json.loads(calibration.read_text(encoding='utf-8'))['rotation_threshold_px2'])
                                      if calibration.is_file() else 1.)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = pydicom.dcmread(str(path), force=True)
    raw = np.squeeze(np.asarray(ds.pixel_array))
    image = read_dicom_image(path)
    router, size = _load_model("router", checkpoint_dir, device)
    tensor, _ = prepare_input(image, size, stretch=True)
    logits = router(tensor[None].to(device))
    probs = torch.softmax(logits[0], 0).cpu().tolist()
    region = force_region or REGIONS[int(np.argmax(probs))]
    if region not in REGIONS:
        raise ValueError(region)
    result = {"source": str(path), "region": region,
              "router_probabilities": dict(zip(REGIONS, probs)),
              "quality_flags": {}, "geometry": None}
    result['quality_scores']={}
    sigmoid=lambda value: 1/(1+math.exp(-float(np.clip(value,-60,60))))
    h, w = image.shape
    geometry = empty_geometry(w, h)
    if region == "SPINE":
        model, spatial_size = _load_model("spine", checkpoint_dir, device)
        artifact, detector_size = _load_model("artifact", checkpoint_dir, device)
        x, meta = prepare_input(image, spatial_size)
        out = model(x[None].to(device))
        spatial = torch.sigmoid(out["spatial"][0]).cpu().numpy()
        geometry["spine"]["disc_lines"] = _spine_lines(spatial[0], meta)
        for k, key in enumerate(CRESTS):
            point, peak = _heatmap_point(spatial[k + 1], meta)
            present = torch.sigmoid(out["presence"][0, k]).item() >= .5 and peak >= .35
            geometry["spine"]["iliac_crests"][key] = point if present else None
        x, meta = prepare_input(image, detector_size)
        detections = artifact(detector_images(x[None].to(device)))[0]
        for i, (box, score) in enumerate(zip(detections["boxes"], detections["scores"])):
            if score.item() < .5:
                continue
            x1, y1, x2, y2 = box.cpu().tolist()
            a, b = _original_point(x1, y1, meta), _original_point(x2, y2, meta)
            geometry["spine"]["foreign_objects"].append(
                {"id": f"predicted_{i}", "kind": "other",
                 "bbox": [min(a[0], b[0]), min(a[1], b[1]),
                          max(a[0], b[0]), max(a[1], b[1])]})
        spacing, basis = _dicom_spacing(ds)
        if spacing_override_mm is not None:
            spacing,basis=tuple(spacing_override_mm),'external_augmentation_spacing'
        if spacing is None:
            spacing, basis = SCANNER_NOMINAL_SPACING_MM, "scanner_nominal_user_supplied"
        analysis = analyze_spine(raw,geometry,spacing,
                                 polarity="dark" if str(getattr(ds,"PhotometricInterpretation",""))=="MONOCHROME1" else "bright")
        angle = analysis["global_angle_deg"]
        placement = placement_from_axes(geometry,analysis)
        result["quality_flags"] = {
            "spine_position": None if placement is None else int(not placement),
            "spine_axis": None if angle is None else int(abs(angle) > 5),
            "spine_artifact": int(bool(geometry["spine"]["foreign_objects"]))}
        result["spine_axis_angle_deg"] = angle
        ratio=analysis['top_ratio']
        crest_presence=torch.sigmoid(out['presence'][0]).cpu().tolist()
        placement_score=max(1-min(crest_presence),sigmoid(max(.25-ratio,ratio-.75)/.1) if ratio is not None else .5)
        if len(geometry['spine']['disc_lines']) not in range(4,8): placement_score=1.
        result['quality_scores']={'spine_position':placement_score,
            'spine_axis':sigmoid(abs(angle)-5) if angle is not None else None,
            'spine_artifact':float(detections['scores'].max().item()) if len(detections['scores']) else 0.}
        result["spacing_basis"] = basis
        result["vertebral_axes"] = analysis
    else:
        side = region.removeprefix("LEG_")
        flipped = side == "RIGHT"
        model, spatial_size = _load_model("hip", checkpoint_dir, device)
        x, meta = prepare_input(image, spatial_size, flipped)
        out = model(x[None].to(device))
        spatial = torch.sigmoid(out["spatial"][0]).cpu().numpy()
        for k, key in enumerate(LANDMARKS):
            point, peak = _heatmap_point(spatial[k], meta)
            present = torch.sigmoid(out["presence"][0, k]).item() >= .5 and peak >= .35
            geometry["hip"]["landmarks"][key] = point if present else None
        y1, y2, lateral = (out["roi"][0].cpu().numpy() * spatial_size).tolist()
        top = _original_point(0, min(y1, y2), meta)[1]
        bottom = _original_point(0, max(y1, y2), meta)[1]
        x_lateral = _original_point(lateral, 0, meta)[0]
        geometry["hip"]["roi_box"] = ([0.0, top, x_lateral, bottom] if side == "LEFT"
                                       else [x_lateral, top, float(w - 1), bottom])
        mask = Image.fromarray((spatial[3] >= .5).astype(np.uint8) * 255)
        dw, dh = round(w * meta["scale"]), round(h * meta["scale"])
        mask = mask.crop((meta["left"], meta["top"],
                          meta["left"] + dw, meta["top"] + dh))
        mask = np.asarray(mask.resize((w, h), Image.Resampling.NEAREST)) > 0
        if flipped:
            mask = np.fliplr(mask)
        area = int(mask.sum())
        geometry["hip"]["lesser_trochanter_pixels"] = (
            [[int(x), int(y)] for y, x in np.argwhere(mask)])
        geometry["hip"]["lesser_trochanter_mask_ready"] = True
        spacing, basis = _dicom_spacing(ds)
        if spacing_override_mm is not None:
            spacing,basis=tuple(spacing_override_mm),'external_augmentation_spacing'
        if spacing is None:
            spacing, basis = SCANNER_NOMINAL_SPACING_MM, "scanner_nominal_user_supplied"
        margins = hip_roi_margins_mm(geometry, side, spacing)
        result["quality_flags"] = {
            "hip_position": int(not hip_position_ok(geometry)),
            "hip_roi": int(not hip_roi_ok(geometry, True, side, spacing)),
            "hip_rotation": int(area < rotation_area_threshold_px2)}
        result["roi_margins_mm"] = margins
        result["spacing_basis"] = basis
        result["lesser_trochanter_area_px2"] = area
        result['lesser_trochanter_area_mm2']=area*spacing[0]*spacing[1]
        presence=torch.sigmoid(out['presence'][0]).cpu().tolist()
        result['quality_scores']={'hip_position':1-min(presence),
            'hip_roi':sigmoid(max((30-margins['top'])/30,(30-margins['bottom'])/30,(20-margins['lateral'])/20)*10),
            'hip_rotation':sigmoid((rotation_area_threshold_px2-area)/max(1.,rotation_area_threshold_px2*.2))}
        result["rotation_area_threshold_px2"] = rotation_area_threshold_px2
    result["geometry"] = geometry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--region", choices=REGIONS)
    parser.add_argument("--rotation-area-threshold", type=float, default=None,
                        help='Override calibrated threshold; otherwise read checkpoints/evaluation/calibration.json')
    args = parser.parse_args()
    output = predict_file(args.image.resolve(), args.checkpoints.resolve(),
                          args.rotation_area_threshold, args.region)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "geometry"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
