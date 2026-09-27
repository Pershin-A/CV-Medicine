"""Merge two DXA labeler output folders by explicit 1-based index ranges.

Selected scans come wholly from --first; every other scan comes wholly from
--second. Both inputs remain untouched. The destination must not exist.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile


RANGE_PATTERN = re.compile(r"^(\d+)(?:\s*[-:]\s*(\d+))?$")


def parse_ranges(values: list[str]) -> list[tuple[int, int]]:
    """Accept repeated --range 1-10,20:25 or --range 30."""
    ranges = []
    for value in values:
        for token in value.split(","):
            match = RANGE_PATTERN.fullmatch(token.strip())
            if not match:
                raise ValueError(f"Invalid range {token!r}; use 1-10 or 15")
            start = int(match.group(1))
            end = int(match.group(2) or match.group(1))
            if start < 1 or end < start:
                raise ValueError(f"Invalid range {token!r}: indices start at 1 and must increase")
            ranges.append((start, end))
    if not ranges:
        raise ValueError("At least one --range is required")
    return ranges


def safe_relative(value: str, field: str) -> Path:
    """Normalize CSV paths without allowing absolute paths or parent traversal."""
    normalized = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if (not normalized or normalized.startswith("/") or
            re.match(r"^[A-Za-z]:", normalized) or
            any(part in (".", "..", "") for part in normalized.split("/"))):
        raise ValueError(f"Unsafe {field} path: {value!r}")
    return Path(*path.parts)


def _safe_source_file(root: Path, relative: Path, field: str) -> Path:
    source = (root / relative).resolve()
    if not source.is_relative_to(root):
        raise ValueError(f"{field} escapes its source folder: {relative}")
    if not source.is_file():
        raise FileNotFoundError(f"Missing {field}: {source}")
    return source


def read_registry(root: Path) -> tuple[list[dict[str, str]], list[str]]:
    csv_path = root / "labels.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "relative_path" not in fields or "output_path" not in fields:
            raise ValueError(f"{csv_path} needs relative_path and output_path columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty registry: {csv_path}")
    keys = set()
    for row in rows:
        if None in row:
            raise ValueError(f"Malformed CSV row in {csv_path}: too many columns")
        relative = safe_relative(row["relative_path"], "relative_path")
        key = relative.as_posix()
        if key in keys:
            raise ValueError(f"Duplicate relative_path in {csv_path}: {key}")
        keys.add(key)
    return rows, fields


def _selection(first_rows: list[dict[str, str]], ranges: list[tuple[int, int]],
               unit: str) -> tuple[set[str], dict[str, int]]:
    if unit == "image":
        ordered_units = [safe_relative(row["relative_path"], "relative_path").as_posix()
                         for row in first_rows]
        unit_for_row = ordered_units
    elif unit == "study":
        unit_for_row = [safe_relative(row["relative_path"], "relative_path").parts[0]
                        for row in first_rows]
        ordered_units = list(dict.fromkeys(unit_for_row))
    else:
        raise ValueError(f"Unknown unit: {unit}")
    total = len(ordered_units)
    for start, end in ranges:
        if end > total:
            raise ValueError(f"Range {start}-{end} exceeds {total} {unit} units in first labels.csv")
    selected_units = {ordered_units[index - 1]
                      for start, end in ranges for index in range(start, end + 1)}
    selected_keys = {
        safe_relative(row["relative_path"], "relative_path").as_posix()
        for row, group in zip(first_rows, unit_for_row) if group in selected_units
    }
    return selected_keys, {value: i + 1 for i, value in enumerate(ordered_units)}


def build_plan(first: Path, second: Path, ranges: list[tuple[int, int]],
               unit: str = "image") -> tuple[list[dict], list[str], dict]:
    first, second = first.resolve(), second.resolve()
    if first == second:
        raise ValueError("--first and --second must be different folders")
    first_rows, first_fields = read_registry(first)
    second_rows, second_fields = read_registry(second)
    first_by_key = {safe_relative(r["relative_path"], "relative_path").as_posix(): r
                    for r in first_rows}
    second_by_key = {safe_relative(r["relative_path"], "relative_path").as_posix(): r
                     for r in second_rows}
    first_keys, index_map = _selection(first_rows, ranges, unit)
    missing_second = (set(first_by_key) - first_keys) - set(second_by_key)
    if missing_second:
        examples = ", ".join(sorted(missing_second)[:3])
        raise ValueError(f"Second folder lacks {len(missing_second)} unselected scans: {examples}")
    # Preserve the first registry's order. Additional second-only scans follow it.
    ordered_keys = list(first_by_key) + [k for k in second_by_key if k not in first_by_key]
    fields = list(dict.fromkeys(first_fields + second_fields))
    plan = []
    occupied_paths = set()
    for key in ordered_keys:
        use_first = key in first_keys
        root = first if use_first else second
        row = dict(first_by_key[key] if use_first else second_by_key[key])
        output_rel = safe_relative(row.get("output_path") or key, "output_path")
        source_dicom = _safe_source_file(root, output_rel, "annotated DICOM")
        if output_rel.as_posix() in occupied_paths:
            raise ValueError(f"Output path collision: {output_rel}")
        occupied_paths.add(output_rel.as_posix())
        geometry_rel = None
        source_geometry = None
        if row.get("geometry_path", "").strip():
            geometry_rel = safe_relative(row["geometry_path"], "geometry_path")
            source_geometry = _safe_source_file(root, geometry_rel, "geometry JSON")
            try:
                raw = json.loads(source_geometry.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("geometry must be an object")
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid geometry JSON: {source_geometry}") from exc
            if geometry_rel.as_posix() in occupied_paths:
                raise ValueError(f"Output path collision: {geometry_rel}")
            occupied_paths.add(geometry_rel.as_posix())
        if (row.get("geometry_spine_complete") == "1" or
                row.get("geometry_hip_complete") == "1") and source_geometry is None:
            raise ValueError(f"Completed geometry has no JSON sidecar: {key}")
        row["relative_path"] = key
        row["output_path"] = output_rel.as_posix()
        if geometry_rel is not None:
            row["geometry_path"] = geometry_rel.as_posix()
        plan.append({"relative_path": key, "origin": "first" if use_first else "second",
                     "root": root, "row": row, "dicom_relative": output_rel,
                     "dicom_source": source_dicom, "geometry_relative": geometry_rel,
                     "geometry_source": source_geometry})
    report = {"first": str(first), "second": str(second),
              "range_unit": unit, "ranges": [f"{a}-{b}" for a, b in ranges],
              "first_units": len(index_map), "from_first": sum(p["origin"] == "first" for p in plan),
              "from_second": sum(p["origin"] == "second" for p in plan),
              "total_scans": len(plan),
              "geometry_sidecars": sum(p["geometry_source"] is not None for p in plan)}
    return plan, fields, report


def merge(first: Path, second: Path, output: Path, ranges: list[tuple[int, int]],
          unit: str = "image", dry_run: bool = False) -> dict:
    if output.is_symlink():
        raise ValueError("Destination must not be a symbolic link")
    first, second, output = first.resolve(), second.resolve(), output.resolve()
    if output == first or output == second or output.is_relative_to(first) or output.is_relative_to(second):
        raise ValueError("Output must be a new folder outside both source folders")
    if first.is_relative_to(output) or second.is_relative_to(output):
        raise ValueError("Output cannot be an ancestor of a source folder")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Destination is not an empty folder: {output}")
    plan, fields, report = build_plan(first, second, ranges, unit)
    report["output"] = str(output)
    if dry_run:
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".merge_annotations_", dir=output.parent) as temporary:
        stage = Path(temporary)
        for item in plan:
            dicom_dest = stage / item["dicom_relative"]
            dicom_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item["dicom_source"], dicom_dest)
            if item["geometry_source"] is not None:
                geometry_dest = stage / item["geometry_relative"]
                geometry_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item["geometry_source"], geometry_dest)
        with (stage / "labels.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(item["row"] for item in plan)
        with (stage / "merge_sources.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["relative_path", "origin"])
            writer.writeheader()
            writer.writerows({"relative_path": item["relative_path"],
                              "origin": item["origin"]} for item in plan)
        (stage / "merge_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if output.exists():
            # Only remove an explicitly empty destination, after all files are staged.
            output.rmdir()
        os.replace(stage, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path, help="First annotation folder")
    parser.add_argument("--second", required=True, type=Path, help="Second annotation folder")
    parser.add_argument("--output", required=True, type=Path, help="New destination folder")
    parser.add_argument("--range", dest="ranges", required=True, action="append",
                        help="Inclusive 1-based range from first folder, e.g. 1-40 or 1-40,81-100")
    parser.add_argument("--unit", choices=("image", "study"), default="image",
                        help="Range unit: labels.csv row (default) or study folder order")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without copying")
    args = parser.parse_args()
    try:
        report = merge(args.first, args.second, args.output,
                       parse_ranges(args.ranges), args.unit, args.dry_run)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
