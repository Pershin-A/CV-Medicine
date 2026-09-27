"""Pure geometry, image transforms and quality rules for DXA augmentation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter1d, median_filter
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from labeler.geometry import validate_geometry  # noqa: E402


@dataclass(frozen=True)
class Transform:
    """Source pixels -> output pixels; scale is isotropic and output size is fixed."""

    width: int
    height: int
    scale: float = 1.0
    angle_deg: float = 0.0  # clockwise in image coordinates
    source_center_x: float | None = None
    source_center_y: float | None = None

    @property
    def source_center(self) -> np.ndarray:
        return np.array([
            (self.width - 1) / 2 if self.source_center_x is None else self.source_center_x,
            (self.height - 1) / 2 if self.source_center_y is None else self.source_center_y,
        ], dtype=float)

    @property
    def output_center(self) -> np.ndarray:
        return np.array([(self.width - 1) / 2, (self.height - 1) / 2], dtype=float)

    @property
    def matrix(self) -> np.ndarray:
        a = math.radians(self.angle_deg)
        return self.scale * np.array([[math.cos(a), -math.sin(a)],
                                      [math.sin(a), math.cos(a)]])

    def point(self, p) -> list[float]:
        return (self.matrix @ (np.asarray(p, dtype=float) - self.source_center)
                + self.output_center).tolist()

    def covers_output(self, tolerance: float = 1e-6) -> bool:
        """Every output pixel must originate inside the real input image."""
        inverse = np.linalg.inv(self.matrix)
        for x in (0, self.width - 1):
            for y in (0, self.height - 1):
                source = inverse @ (np.array([x, y]) - self.output_center) + self.source_center
                if not (-tolerance <= source[0] <= self.width - 1 + tolerance and
                        -tolerance <= source[1] <= self.height - 1 + tolerance):
                    return False
        return True


def warp_image(image: np.ndarray, transform: Transform) -> np.ndarray:
    if image.shape != (transform.height, transform.width):
        raise ValueError("Image dimensions differ from transform")
    if not transform.covers_output():
        raise ValueError("Transform would synthesize anatomy outside the source image")
    inverse = np.linalg.inv(transform.matrix)
    offset_xy = transform.source_center - inverse @ transform.output_center
    matrix_yx = inverse[::-1, ::-1]
    offset_yx = offset_xy[::-1]
    return affine_transform(image, matrix_yx, offset_yx, output_shape=image.shape,
                            order=1, mode="nearest", prefilter=False)


def _inside(p, w, h, margin=0.0):
    return margin <= p[0] <= w - 1 - margin and margin <= p[1] <= h - 1 - margin


def _clip_segment(a, b, w, h):
    """Liang–Barsky clipping; return clipped endpoints and visible length fraction."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    delta = b - a
    lo, hi = 0.0, 1.0
    for p, q in ((-delta[0], a[0]), (delta[0], w - 1 - a[0]),
                 (-delta[1], a[1]), (delta[1], h - 1 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None, 0.0
        else:
            r = q / p
            if p < 0:
                lo = max(lo, r)
            else:
                hi = min(hi, r)
    if lo >= hi:
        return None, 0.0
    return [np.clip(a + lo * delta, [0, 0], [w - 1, h - 1]).tolist(),
            np.clip(a + hi * delta, [0, 0], [w - 1, h - 1]).tolist()], hi - lo


def _clip_polygon(points, w, h):
    polygon = [list(p) for p in points]
    for axis, boundary, keep_greater in ((0, 0, True), (0, w - 1, False),
                                          (1, 0, True), (1, h - 1, False)):
        if not polygon:
            break
        result = []
        prev = polygon[-1]
        prev_ok = prev[axis] >= boundary if keep_greater else prev[axis] <= boundary
        for curr in polygon:
            curr_ok = curr[axis] >= boundary if keep_greater else curr[axis] <= boundary
            if curr_ok != prev_ok:
                t = (boundary - prev[axis]) / (curr[axis] - prev[axis])
                result.append([prev[i] + t * (curr[i] - prev[i]) for i in (0, 1)])
            if curr_ok:
                result.append(curr)
            prev, prev_ok = curr, curr_ok
        polygon = result
    return [np.clip(p, [0, 0], [w - 1, h - 1]).tolist() for p in polygon]


def _area(points):
    return abs(sum(a[0] * b[1] - b[0] * a[1]
                   for a, b in zip(points, points[1:] + points[:1]))) / 2


def prepare_geometry(geometry: dict, region: str) -> dict:
    """Normalize annotation conventions without changing the saved source JSON.

    Spine line endpoints encode orientation, not length. For hips the
    image-left edge is arbitrary on the left leg, image-right on the right.
    """
    out = deepcopy(geometry)
    w, h = out["image_width"], out["image_height"]
    if region == "SPINE":
        for line in out["spine"]["disc_lines"]:
            a, b = [np.asarray(p, dtype=float) for p in line["points"]]
            dx = b[0] - a[0]
            if abs(dx) < 1e-9:
                # A vertical segment cannot encode a left/right disc boundary.
                raise ValueError("Vertical spine separation line cannot be extended horizontally")
            slope = (b[1] - a[1]) / dx
            left = [0.0, float(a[1] - slope * a[0])]
            right = [float(w - 1), float(a[1] + slope * (w - 1 - a[0]))]
            clipped, _ = _clip_segment(left, right, w, h)
            if clipped is None or math.dist(*clipped) < 1:
                raise ValueError("Extended spine line misses the image")
            line["points"] = clipped
    elif region in ("LEG_LEFT", "LEG_RIGHT"):
        box = out["hip"]["roi_box"]
        if box is not None:
            if region == "LEG_LEFT":
                box[0] = 0.0
            else:
                box[2] = float(w - 1)
    else:
        raise ValueError(f"Unknown region: {region}")
    if region in ("LEG_LEFT", "LEG_RIGHT"):
        mask = (_pixels_mask(out) if out["hip"].get("lesser_trochanter_mask_ready")
                else build_lesser_trochanter_mask(out))
        out["hip"]["lesser_trochanter_pixels"] = _mask_pixels(mask)
        out["hip"]["lesser_trochanter_mask_ready"] = True
        out["hip"]["lesser_trochanter_partial"] = bool(
            out["hip"].get("lesser_trochanter_partial") or
            (mask.any() and (mask[0].any() or mask[-1].any() or
                             mask[:, 0].any() or mask[:, -1].any())))
    return validate_geometry(out, w, h)


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _curve_intersections(first, second) -> list[list[float]]:
    """Distinct intersections of two freehand polylines in pixel coordinates."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    b0, bd = b[:-1], np.diff(b, axis=0)
    found = []
    for p, d in zip(a[:-1], np.diff(a, axis=0)):
        denominator = _cross2(d, bd)
        difference = b0 - p
        valid = np.abs(denominator) > 1e-9
        t = np.divide(_cross2(difference, bd), denominator,
                      out=np.zeros_like(denominator), where=valid)
        u = np.divide(_cross2(difference, d), denominator,
                      out=np.zeros_like(denominator), where=valid)
        for point in (p + t[j] * d for j in np.flatnonzero(
                valid & (t >= -1e-6) & (t <= 1 + 1e-6) &
                (u >= -1e-6) & (u <= 1 + 1e-6))):
            if not any(np.linalg.norm(point - old) < 0.75 for old in found):
                found.append(point)
    return [p.tolist() for p in found]


def _scanline_positions(strokes, height: int) -> list[list[float]]:
    """Subpixel x intersections of freehand segments with image pixel rows."""
    positions = [[] for _ in range(height)]
    for stroke in strokes:
        for first, second in zip(stroke["points"], stroke["points"][1:]):
            x1, y1 = first
            x2, y2 = second
            if abs(y2 - y1) < 1e-9:
                row = int(round((y1 + y2) / 2))
                if 0 <= row < height:
                    positions[row].append((x1 + x2) / 2)
                continue
            low = max(0, int(math.ceil(min(y1, y2) - 0.5)))
            high = min(height - 1, int(math.floor(max(y1, y2) - 0.5)))
            for row in range(low, high + 1):
                y = row + 0.5
                positions[row].append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
    return positions


def build_lesser_trochanter_mask(geometry: dict) -> np.ndarray:
    """Fill an interior widening bounded by narrow contour separations.

    The median x of each trace gives the horizontal gap on every common row.
    A visible tubercle is a smoothed interior peak at least 5 px wide, with
    a narrow (at most 2.5–5 px) shoulder on both sides, at least 10 rows long
    and 50 px² in integrated width. Choose the largest qualifying bulge.
    In particular, a U-shaped profile with wide ends has no interior peak.
    """
    h, w = geometry["image_height"], geometry["image_width"]
    mask = np.zeros((h, w), dtype=bool)
    layers = geometry["hip"]["lesser_trochanter_traces"]
    if not layers["trochanter"] or not layers["adjacent_bone"]:
        return mask
    first = _scanline_positions(layers["trochanter"], h)
    second = _scanline_positions(layers["adjacent_bone"], h)
    max_gap = 0.25 * min(w, h)
    samples = []
    for y, (a, b) in enumerate(zip(first, second)):
        if not a or not b:
            continue
        left_curve, right_curve = float(np.median(a)), float(np.median(b))
        gap = abs(right_curve - left_curve)
        if gap > max_gap:
            continue
        left = max(0, int(math.ceil(min(left_curve, right_curve))))
        right = min(w - 1, int(math.floor(max(left_curve, right_curve))))
        samples.append((y, left, right, gap))
    if len(samples) < 10:
        return mask
    widths = np.asarray([sample[3] for sample in samples], dtype=float)
    smooth = gaussian_filter1d(median_filter(widths, size=5), sigma=1.5)
    peaks, properties = find_peaks(smooth, prominence=2.0, distance=8)
    best = None
    for peak, prominence in zip(peaks, properties["prominences"]):
        height = float(smooth[peak])
        if height < 4 or prominence < 2:
            continue
        shoulder = min(5.0, max(2.5, 0.25 * height))
        left = np.flatnonzero(smooth[:peak] <= shoulder)
        right = np.flatnonzero(smooth[peak + 1:] <= shoulder)
        if not len(left) or not len(right):
            continue
        start, end = int(left[-1]), int(peak + 1 + right[0])
        length = samples[end][0] - samples[start][0] + 1
        area = float(widths[start:end + 1].sum())
        if length < 10 or area < 50:
            continue
        # A gap in either pencil trace cannot be filled by interpolation.
        if any(samples[i + 1][0] - samples[i][0] > 2
               for i in range(start, end)):
            continue
        if best is None or area > best[0]:
            best = (area, start, end)
    if best is None:
        return mask
    for y, left, right, _ in samples[best[1]:best[2] + 1]:
        if right >= left:
            mask[y, left:right + 1] = True
    return mask


def _mask_pixels(mask: np.ndarray) -> list[list[int]]:
    return [[int(x), int(y)] for y, x in np.argwhere(mask)]


def _pixels_mask(geometry: dict) -> np.ndarray:
    h, w = geometry["image_height"], geometry["image_width"]
    mask = np.zeros((h, w), dtype=bool)
    pixels = geometry["hip"].get("lesser_trochanter_pixels") or []
    if pixels:
        xy = np.asarray(pixels, dtype=int)
        mask[xy[:, 1], xy[:, 0]] = True
    elif not geometry["hip"].get("lesser_trochanter_mask_ready", False):
        mask = build_lesser_trochanter_mask(geometry)
    return mask


def lesser_trochanter_between_area(geometry: dict) -> tuple[int, int]:
    """Return pixel mask area and diagnostic count of curve crossings."""
    layers = geometry["hip"]["lesser_trochanter_traces"]
    intersections = []
    for first in layers["trochanter"]:
        for second in layers["adjacent_bone"]:
            for point in _curve_intersections(first["points"], second["points"]):
                if not any(math.dist(point, old) < 0.75 for old in intersections):
                    intersections.append(point)
    return int(_pixels_mask(geometry).sum()), len(intersections)


def transform_geometry(geometry: dict, transform: Transform,
                       min_visible: float = 0.5,
                       region: str | None = None) -> tuple[dict, dict]:
    """Transform annotations, removing mostly invisible objects, then validate."""
    w, h = transform.width, transform.height
    out = deepcopy(geometry)
    dropped = {"disc_lines": 0, "foreign_objects": 0, "landmarks": 0,
               "roi_box": 0, "trochanter": 0, "traces": 0,
               "trochanter_pixels": 0}
    lines = []
    for line in out["spine"]["disc_lines"]:
        endpoints, fraction = _clip_segment(*(transform.point(p) for p in line["points"]), w, h)
        if endpoints and fraction >= min_visible and math.dist(*endpoints) >= 1:
            lines.append({**line, "points": endpoints})
        else:
            dropped["disc_lines"] += 1
    out["spine"]["disc_lines"] = lines
    for key, p in out["spine"]["iliac_crests"].items():
        q = None if p is None else transform.point(p)
        out["spine"]["iliac_crests"][key] = q if q is not None and _inside(q, w, h) else None
    objects = []
    for obj in out["spine"]["foreign_objects"]:
        x1, y1, x2, y2 = obj["bbox"]
        poly = [transform.point(p) for p in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
        clipped = _clip_polygon(poly, w, h)
        if clipped and _area(clipped) / max(_area(poly), 1e-9) >= min_visible:
            xs, ys = [p[0] for p in clipped], [p[1] for p in clipped]
            if max(xs) - min(xs) >= 1 and max(ys) - min(ys) >= 1:
                objects.append({**obj, "bbox": [min(xs), min(ys), max(xs), max(ys)]})
                continue
        dropped["foreign_objects"] += 1
    out["spine"]["foreign_objects"] = objects
    for key, p in out["hip"]["landmarks"].items():
        q = None if p is None else transform.point(p)
        if q is not None and not _inside(q, w, h):
            dropped["landmarks"] += 1
        out["hip"]["landmarks"][key] = q if q is not None and _inside(q, w, h) else None
    box = out["hip"]["roi_box"]
    roi_fully_visible = False
    if box is not None:
        x1, y1, x2, y2 = box
        poly = [transform.point(p) for p in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
        if region == "LEG_LEFT":
            roi_fully_visible = all(_inside(p, w, h) for p in (poly[1], poly[2]))
        elif region == "LEG_RIGHT":
            roi_fully_visible = all(_inside(p, w, h) for p in (poly[0], poly[3]))
        else:
            roi_fully_visible = all(_inside(p, w, h) for p in poly)
        clipped = _clip_polygon(poly, w, h)
        if clipped and _area(clipped) >= 1 and (
                region in ("LEG_LEFT", "LEG_RIGHT") or
                _area(clipped) / max(_area(poly), 1e-9) >= min_visible):
            xs, ys = [p[0] for p in clipped], [p[1] for p in clipped]
            out["hip"]["roi_box"] = [min(xs), min(ys), max(xs), max(ys)]
            if region == "LEG_LEFT":
                out["hip"]["roi_box"][0] = 0.0
            elif region == "LEG_RIGHT":
                out["hip"]["roi_box"][2] = float(w - 1)
        else:
            out["hip"]["roi_box"] = None
            dropped["roi_box"] += 1
    poly = out["hip"]["lesser_trochanter"]
    if poly:
        moved = [transform.point(p) for p in poly]
        clipped = _clip_polygon(moved, w, h)
        if len(clipped) >= 3 and _area(clipped) >= 1 and _area(clipped) / _area(moved) >= min_visible:
            out["hip"]["lesser_trochanter"] = clipped
        else:
            out["hip"]["lesser_trochanter"] = None
            dropped["trochanter"] += 1
    for layer in ("trochanter", "adjacent_bone"):
        strokes = []
        for stroke in out["hip"]["lesser_trochanter_traces"][layer]:
            moved = [transform.point(p) for p in stroke["points"]]
            segments = [_clip_segment(a, b, w, h) for a, b in zip(moved, moved[1:])]
            runs, run = [], []
            for seg, _ in segments:
                if seg:
                    if run and math.dist(run[-1], seg[0]) > 1e-4:
                        runs.append(run)
                        run = []
                    if not run:
                        run.append(seg[0])
                    run.append(seg[1])
                elif run:
                    runs.append(run)
                    run = []
            if run:
                runs.append(run)
            for index, visible_run in enumerate(runs):
                if sum(math.dist(a, b) for a, b in zip(visible_run, visible_run[1:])) >= 1:
                    item = {**stroke, "points": visible_run}
                    if index:
                        item["id"] = f"{stroke['id'][:56]}_{index}"
                    strokes.append(item)
            if not runs:
                dropped["traces"] += 1
        out["hip"]["lesser_trochanter_traces"][layer] = strokes
    source_mask = _pixels_mask(geometry)
    source_partial = bool(geometry["hip"].get("lesser_trochanter_partial", False))
    if source_mask.any():
        yx = np.argwhere(source_mask)
        xy = yx[:, ::-1].astype(float)
        mapped = (transform.matrix @ (xy - transform.source_center).T).T + transform.output_center
        newly_clipped = bool(np.any((mapped[:, 0] < 0) | (mapped[:, 0] > w - 1) |
                                    (mapped[:, 1] < 0) | (mapped[:, 1] > h - 1)))
        inverse = np.linalg.inv(transform.matrix)
        offset_xy = transform.source_center - inverse @ transform.output_center
        moved_mask = affine_transform(source_mask, inverse[::-1, ::-1],
                                      offset_xy[::-1], output_shape=(h, w),
                                      order=0, mode="constant", cval=0,
                                      prefilter=False)
        out["hip"]["lesser_trochanter_pixels"] = _mask_pixels(moved_mask)
        estimated_total = int(round(int(source_mask.sum()) * transform.scale ** 2))
        dropped["trochanter_pixels"] = max(0, estimated_total - int(moved_mask.sum()))
        out["hip"]["lesser_trochanter_partial"] = source_partial or newly_clipped
    else:
        out["hip"]["lesser_trochanter_pixels"] = []
        out["hip"]["lesser_trochanter_partial"] = source_partial
    out["hip"]["lesser_trochanter_mask_ready"] = True
    # A clipped ROI must never be mistaken for a valid physical ROI.
    meta = {"dropped": dropped, "roi_fully_visible": roi_fully_visible}
    out = validate_geometry(out, w, h)
    return out, meta


def spine_axis_angle(image: np.ndarray, geometry: dict) -> float | None:
    """Fit a brightness-symmetry axis inside each vertebra; angle from vertical."""
    lines = sorted(geometry["spine"]["disc_lines"],
                   key=lambda l: sum(p[1] for p in l["points"]) / 2)
    if len(lines) < 2:
        return None
    h, w = image.shape
    fitted_axes = []
    for upper, lower in zip(lines, lines[1:]):
        top = np.mean(upper["points"], axis=0)
        bottom = np.mean(lower["points"], axis=0)
        half_width = max(4, int(min(np.linalg.norm(np.subtract(*upper["points"])),
                                    np.linalg.norm(np.subtract(*lower["points"]))) * 0.20))
        samples = []
        for fraction in (0.2, 0.5, 0.8):
            y_mid = float(top[1] + fraction * (bottom[1] - top[1]))
            y = int(round(y_mid))
            if y < 2 or y >= h - 2:
                continue
            x_guess = top[0] + fraction * (bottom[0] - top[0])
            candidates = range(max(half_width + 1, int(x_guess - half_width)),
                               min(w - half_width - 1, int(x_guess + half_width)) + 1)
            if not candidates:
                continue
            row = image[max(0, y - 2):min(h, y + 3)].astype(float).mean(axis=0)
            offsets = np.arange(1, half_width + 1)
            scores = [np.mean(np.abs(row[x - offsets] - row[x + offsets]))
                      + 0.05 * abs(x - x_guess) for x in candidates]
            samples.append((y_mid, float(candidates[int(np.argmin(scores))])))
        if len(samples) < 2:
            return None
        slope, intercept = np.polyfit([p[0] for p in samples], [p[1] for p in samples], 1)
        fitted_axes.append((slope, intercept))
    top_y = float(np.mean(lines[0]["points"], axis=0)[1])
    bottom_y = float(np.mean(lines[-1]["points"], axis=0)[1])
    first = np.array([fitted_axes[0][0] * top_y + fitted_axes[0][1], top_y])
    second = np.array([fitted_axes[-1][0] * bottom_y + fitted_axes[-1][1], bottom_y])
    dy = second[1] - first[1]
    return math.degrees(math.atan2(second[0] - first[0], dy)) if dy > 0 else None


def spine_position_ok(geometry: dict, top_ratio_range=(0.25, 0.75),
                      crest_margin_fraction=0.025) -> bool:
    lines = sorted(geometry["spine"]["disc_lines"],
                   key=lambda l: sum(p[1] for p in l["points"]) / 2)
    if len(lines) not in (4, 5, 6, 7):
        return False
    mids = [np.mean(line["points"], axis=0) for line in lines]
    gaps = np.diff([m[1] for m in mids])
    if np.any(gaps <= 1):
        return False
    ratio = mids[0][1] / float(np.median(gaps))
    if not top_ratio_range[0] <= ratio <= top_ratio_range[1]:
        return False
    w, h = geometry["image_width"], geometry["image_height"]
    margin = crest_margin_fraction * min(w, h)
    return all(p is not None and _inside(p, w, h, margin)
               for p in geometry["spine"]["iliac_crests"].values())


def hip_position_ok(geometry: dict, margin_fraction=0.025) -> bool:
    w, h = geometry["image_width"], geometry["image_height"]
    margin = margin_fraction * min(w, h)
    return all(p is not None and _inside(p, w, h, margin)
               for p in geometry["hip"]["landmarks"].values())


def hip_roi_ok(geometry: dict, fully_visible: bool, side: str,
               spacing_mm: tuple[float, float] | None,
               lateral_edge: dict[str, str] | None = None) -> bool | None:
    box = geometry["hip"]["roi_box"]
    if box is None or not fully_visible:
        return False
    if spacing_mm is None:
        return None
    margins = hip_roi_margins_mm(geometry, side, spacing_mm, lateral_edge)
    return (margins["top"] >= 30 and margins["bottom"] >= 30 and
            margins["lateral"] >= 20)


def hip_roi_margins_mm(geometry: dict, side: str,
                      spacing_mm: tuple[float, float],
                      lateral_edge: dict[str, str] | None = None) -> dict[str, float]:
    """Distances from the three meaningful ROI sides to the image frame."""
    box = geometry["hip"]["roi_box"]
    if box is None:
        raise ValueError("ROI is missing")
    lateral_edge = lateral_edge or {"LEFT": "right", "RIGHT": "left"}
    edge = lateral_edge[side]
    x1, y1, x2, y2 = box
    w, h = geometry["image_width"], geometry["image_height"]
    row_mm, col_mm = spacing_mm
    top = y1 * row_mm
    bottom = (h - 1 - y2) * row_mm
    lateral = (w - 1 - x2 if edge == "right" else x1) * col_mm
    return {"top": float(top), "bottom": float(bottom),
            "lateral": float(lateral)}


def transformed_spacing(spacing_mm, transform: Transform):
    return None if spacing_mm is None else tuple(float(v) / transform.scale for v in spacing_mm)
