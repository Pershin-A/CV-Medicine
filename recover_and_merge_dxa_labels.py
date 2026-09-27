from __future__ import annotations

from pathlib import Path, PurePosixPath
from collections import defaultdict
from itertools import combinations
from datetime import datetime
import hashlib
import json

import pandas as pd
import pydicom


# ============================================================
# НАСТРОЙКИ
# ============================================================

ROOT = Path(r"C:\Users\Андрей\Desktop\Хакатон")

ORIGINAL_ROOT = ROOT / "Исследования"

ANNOTATION_FOLDERS = {
    "Размеченные": ROOT / "Размеченные",
    "Размеченные_244_Илья": ROOT / "Размеченные_244_Илья",
    "Размеченные_499_Илья": ROOT / "Размеченные_499_Илья",
    "Размеченные_old": ROOT / "Размеченные_old",
}

REPORT_DIR = ROOT / "labels_recovery_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_ORIGINAL_COUNT = 499

PRIVATE_GROUP = 0x0011
PRIVATE_CREATOR = "DXA_MANUAL_LABELER"

OFFSET_LABEL = 0x01
OFFSET_NOTES = 0x02
OFFSET_ANNOTATOR = 0x03
OFFSET_DATETIME = 0x04

KNOWN_LABELS = {"LEG", "SPINE"}
UNKNOWN_LABELS = {"UNKNOWN", ""}


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def normalize_path(value) -> str:
    value = str(value or "").strip().replace("\\", "/")

    while "//" in value:
        value = value.replace("//", "/")

    return value.lstrip("./")


def safe_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def read_dicom_header(path: Path):
    """
    Читаем только DICOM-header, без PixelData.
    """
    return pydicom.dcmread(
        path,
        stop_before_pixels=True,
        force=True,
    )


def get_private_block(ds, create=False):
    try:
        return ds.private_block(
            PRIVATE_GROUP,
            PRIVATE_CREATOR,
            create=create,
        )
    except Exception:
        return None


def get_private_value(ds, offset, default=""):
    block = get_private_block(
        ds,
        create=False,
    )

    if block is None:
        return default

    try:
        tag = block.get_tag(offset)
    except Exception:
        return default

    if tag not in ds:
        return default

    value = ds[tag].value

    if value is None:
        return default

    # Private DICOM tags могут читаться как bytes/UN.
    if isinstance(value, bytes):
        try:
            value = value.decode(
                "utf-8",
                errors="replace",
            )
        except Exception:
            return default

    # DICOM-строки могут иметь padding пробелом или NULL.
    return (
        str(value)
        .replace("\x00", "")
        .strip()
    )


def extract_private_annotation(ds):
    return {
        "label": get_private_value(
            ds,
            OFFSET_LABEL,
            "",
        ).upper(),

        "notes": get_private_value(
            ds,
            OFFSET_NOTES,
            "",
        ),

        "annotator": get_private_value(
            ds,
            OFFSET_ANNOTATOR,
            "",
        ),

        "annotated_at_dicom": get_private_value(
            ds,
            OFFSET_DATETIME,
            "",
        ),
    }


def file_sha1(path: Path, chunk_size=1024 * 1024):
    """
    Используется только при необходимости диагностики.
    Для основного merge идентификатором служит SOPInstanceUID.
    """
    h = hashlib.sha1()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# 1. ИНДЕКС ОРИГИНАЛЬНОГО ДАТАСЕТА
# ============================================================

def build_original_index(root: Path):
    files = sorted(
        root.rglob("*.dcm")
    )

    rows = []
    uid_to_paths = defaultdict(list)
    exact_path_map = {}
    suffix_map = defaultdict(list)
    basename_map = defaultdict(list)

    print("=" * 100)
    print("СКАНИРОВАНИЕ ИСХОДНЫХ DICOM")
    print("=" * 100)

    print(f"Папка: {root}")
    print(f"Найдено *.dcm: {len(files)}")

    for i, path in enumerate(files, start=1):
        print(
            f"\rЧтение оригиналов: {i}/{len(files)}",
            end=""
        )

        try:
            ds = read_dicom_header(path)

            sop_uid = safe_str(
                getattr(
                    ds,
                    "SOPInstanceUID",
                    "",
                )
            )

            study_uid = safe_str(
                getattr(
                    ds,
                    "StudyInstanceUID",
                    "",
                )
            )

            series_uid = safe_str(
                getattr(
                    ds,
                    "SeriesInstanceUID",
                    "",
                )
            )

        except Exception as e:
            sop_uid = ""
            study_uid = ""
            series_uid = ""

            print(
                f"\nОшибка DICOM: {path}\n{e}"
            )

        relative_path = normalize_path(
            path.relative_to(root)
        )

        rows.append({
            "sop_uid": sop_uid,
            "study_uid": study_uid,
            "series_uid": series_uid,
            "relative_path": relative_path,
            "original_path": str(path),
            "filename": path.name,
            "size_bytes": path.stat().st_size,
        })

        if sop_uid:
            uid_to_paths[sop_uid].append(path)

        exact_path_map[
            relative_path.casefold()
        ] = path

        parts = PurePosixPath(
            relative_path
        ).parts

        # Несколько suffix-вариантов нужны,
        # потому что старые labels.csv могли содержать
        # различное число ведущих каталогов.
        for n in range(
            2,
            min(8, len(parts)) + 1
        ):
            key = "/".join(
                parts[-n:]
            ).casefold()

            suffix_map[key].append(path)

        basename_map[
            path.name.casefold()
        ].append(path)

    print()

    df = pd.DataFrame(rows)

    duplicate_uids = (
        df[
            df["sop_uid"] != ""
        ]
        .groupby("sop_uid")
        .size()
        .loc[lambda s: s > 1]
    )

    duplicate_uid_df = (
        df[
            df["sop_uid"].isin(
                duplicate_uids.index
            )
        ]
        .sort_values([
            "sop_uid",
            "relative_path",
        ])
        if len(duplicate_uids)
        else pd.DataFrame(
            columns=df.columns
        )
    )

    no_uid_df = df[
        df["sop_uid"] == ""
    ].copy()

    return {
        "files": files,
        "df": df,
        "uid_to_paths": uid_to_paths,
        "exact_path_map": exact_path_map,
        "suffix_map": suffix_map,
        "basename_map": basename_map,
        "duplicate_uid_df": duplicate_uid_df,
        "no_uid_df": no_uid_df,
    }


ORIGINAL = build_original_index(
    ORIGINAL_ROOT
)

original_df = ORIGINAL["df"]

original_df.to_csv(
    REPORT_DIR / "00_original_manifest.csv",
    index=False,
    encoding="utf-8-sig",
)

ORIGINAL[
    "duplicate_uid_df"
].to_csv(
    REPORT_DIR / "00_duplicate_original_uids.csv",
    index=False,
    encoding="utf-8-sig",
)

ORIGINAL[
    "no_uid_df"
].to_csv(
    REPORT_DIR / "00_originals_without_uid.csv",
    index=False,
    encoding="utf-8-sig",
)


print(
    f"\nОригинальных DICOM: {len(original_df)}"
)

print(
    "Уникальных SOPInstanceUID:",
    original_df[
        "sop_uid"
    ][
        original_df["sop_uid"] != ""
    ].nunique()
)

if EXPECTED_ORIGINAL_COUNT:
    if len(original_df) == EXPECTED_ORIGINAL_COUNT:
        print(
            f"OK: ожидаемое число "
            f"{EXPECTED_ORIGINAL_COUNT}"
        )
    else:
        print(
            f"ВНИМАНИЕ: ожидалось "
            f"{EXPECTED_ORIGINAL_COUNT}, "
            f"найдено {len(original_df)}"
        )


# ============================================================
# 2. СОПОСТАВЛЕНИЕ PATH ИЗ CSV С ОРИГИНАЛОМ
# ============================================================

def path_candidates(value):
    raw = normalize_path(value)

    if not raw:
        return []

    parts = list(
        PurePosixPath(raw).parts
    )

    folded = [
        str(p).casefold()
        for p
        in parts
    ]

    candidates = [raw]

    # Отбрасываем типичные корневые части.
    for anchor in (
        "исследования",
        "data",
    ):
        if anchor in folded:
            i = folded.index(anchor)

            if i + 1 < len(parts):
                candidates.append(
                    "/".join(
                        parts[i + 1:]
                    )
                )

    # suffix-варианты
    for n in range(
        2,
        min(8, len(parts)) + 1
    ):
        candidates.append(
            "/".join(
                parts[-n:]
            )
        )

    result = []
    seen = set()

    for candidate in candidates:
        key = candidate.casefold()

        if key not in seen:
            seen.add(key)
            result.append(candidate)

    return result


def resolve_original_from_relative_path(value):
    """
    Возвращает:
        (Path | None, method)
    """

    for candidate in path_candidates(
        value
    ):
        key = candidate.casefold()

        # 1. exact
        if key in ORIGINAL[
            "exact_path_map"
        ]:
            return (
                ORIGINAL[
                    "exact_path_map"
                ][key],
                "exact_path",
            )

        # 2. unique suffix
        matches = ORIGINAL[
            "suffix_map"
        ].get(
            key,
            []
        )

        if len(matches) == 1:
            return (
                matches[0],
                "unique_suffix",
            )

    # 3. basename только если уникален
    name = PurePosixPath(
        normalize_path(value)
    ).name.casefold()

    matches = ORIGINAL[
        "basename_map"
    ].get(
        name,
        []
    )

    if len(matches) == 1:
        return (
            matches[0],
            "unique_basename",
        )

    return None, "not_found"


def original_info_from_path(
    path: Path | None
):
    if path is None:
        return {
            "sop_uid": "",
            "canonical_relative_path": "",
            "canonical_original_path": "",
        }

    rel = normalize_path(
        path.relative_to(
            ORIGINAL_ROOT
        )
    )

    row = original_df[
        original_df[
            "relative_path"
        ]
        == rel
    ]

    if row.empty:
        return {
            "sop_uid": "",
            "canonical_relative_path": rel,
            "canonical_original_path": str(path),
        }

    row = row.iloc[0]

    return {
        "sop_uid": row["sop_uid"],
        "canonical_relative_path": row[
            "relative_path"
        ],
        "canonical_original_path": row[
            "original_path"
        ],
    }


# ============================================================
# 3. ФИЗИЧЕСКИЕ DICOM В ПАПКАХ РАЗМЕТКИ
# ============================================================

def scan_annotation_dicoms(
    source_name: str,
    folder: Path
):
    rows = []

    dicoms = sorted(
        folder.rglob("*.dcm")
    )

    print(
        f"\n{source_name}: "
        f"физических *.dcm = "
        f"{len(dicoms)}"
    )

    for i, path in enumerate(
        dicoms,
        start=1
    ):
        print(
            f"\r  DICOM: {i}/{len(dicoms)}",
            end=""
        )

        try:
            ds = read_dicom_header(path)

            sop_uid = safe_str(
                getattr(
                    ds,
                    "SOPInstanceUID",
                    "",
                )
            )

            private = (
                extract_private_annotation(
                    ds
                )
            )

            canonical = ""

            if (
                sop_uid
                and sop_uid
                in ORIGINAL[
                    "uid_to_paths"
                ]
                and len(
                    ORIGINAL[
                        "uid_to_paths"
                    ][sop_uid]
                )
                == 1
            ):
                original_path = (
                    ORIGINAL[
                        "uid_to_paths"
                    ][sop_uid][0]
                )

                canonical = normalize_path(
                    original_path.relative_to(
                        ORIGINAL_ROOT
                    )
                )

                match_status = (
                    "uid_match"
                )

            elif sop_uid:
                match_status = (
                    "uid_not_unique_or_missing"
                )
            else:
                match_status = (
                    "no_uid"
                )

            rows.append({
                "source": source_name,
                "annotation_dicom_path": str(path),
                "annotation_relative_path": normalize_path(
                    path.relative_to(folder)
                ),
                "sop_uid": sop_uid,
                "canonical_relative_path": canonical,
                "dicom_private_label": private[
                    "label"
                ],
                "dicom_private_notes": private[
                    "notes"
                ],
                "dicom_private_annotator": private[
                    "annotator"
                ],
                "dicom_private_datetime": private[
                    "annotated_at_dicom"
                ],
                "dicom_match_status": match_status,
            })

        except Exception as e:
            rows.append({
                "source": source_name,
                "annotation_dicom_path": str(path),
                "annotation_relative_path": normalize_path(
                    path.relative_to(folder)
                ),
                "sop_uid": "",
                "canonical_relative_path": "",
                "dicom_private_label": "",
                "dicom_private_notes": "",
                "dicom_private_annotator": "",
                "dicom_private_datetime": "",
                "dicom_match_status":
                    f"read_error: {e}",
            })

    print()

    return pd.DataFrame(
        rows
    )


physical_frames = []

for source, folder in (
    ANNOTATION_FOLDERS.items()
):
    if not folder.exists():
        print(
            f"\nПапка отсутствует: "
            f"{folder}"
        )
        continue

    physical_frames.append(
        scan_annotation_dicoms(
            source,
            folder
        )
    )


physical_df = (
    pd.concat(
        physical_frames,
        ignore_index=True
    )
    if physical_frames
    else pd.DataFrame()
)

physical_df.to_csv(
    REPORT_DIR
    / "01_physical_annotation_dicoms.csv",
    index=False,
    encoding="utf-8-sig",
)


# Индекс физических размеченных DICOM
# source -> SOP UID -> строки
physical_by_source_uid = (
    defaultdict(
        lambda: defaultdict(list)
    )
)

if not physical_df.empty:
    for _, row in physical_df.iterrows():
        uid = safe_str(
            row[
                "sop_uid"
            ]
        )

        if uid:
            physical_by_source_uid[
                row[
                    "source"
                ]
            ][uid].append(row)


# ============================================================
# 4. ЧТЕНИЕ labels.csv
# ============================================================

def try_parse_datetime(
    value
):
    value = safe_str(value)

    if not value:
        return pd.NaT

    return pd.to_datetime(
        value,
        errors="coerce",
    )


def resolve_csv_row(
    source_name: str,
    folder: Path,
    row: pd.Series,
):
    """
    Порядок:

    1. relative_path -> оригинал
    2. output_path -> физический размеченный DICOM -> SOP UID
    3. relative_path -> физический DICOM -> SOP UID
    """

    relative_path = normalize_path(
        row.get(
            "relative_path",
            ""
        )
    )

    output_path_value = normalize_path(
        row.get(
            "output_path",
            ""
        )
    )

    # --------------------------------------------------------
    # A. Прямо по relative_path к оригиналу
    # --------------------------------------------------------

    original_path, method = (
        resolve_original_from_relative_path(
            relative_path
        )
    )

    if original_path is not None:
        info = original_info_from_path(
            original_path
        )

        return {
            **info,
            "resolve_method":
                method,
            "resolved_via":
                "csv_relative_path",
        }

    # --------------------------------------------------------
    # B. output_path из CSV
    # --------------------------------------------------------

    possible_paths = []

    if output_path_value:
        out = Path(
            output_path_value
        )

        if out.is_absolute():
            possible_paths.append(
                out
            )

        possible_paths.append(
            folder
            / Path(
                *PurePosixPath(
                    output_path_value
                ).parts
            )
        )

    # --------------------------------------------------------
    # C. relative_path внутри annotation folder
    # --------------------------------------------------------

    if relative_path:
        possible_paths.append(
            folder
            / Path(
                *PurePosixPath(
                    relative_path
                ).parts
            )
        )

    seen_paths = set()

    for candidate in possible_paths:
        key = str(
            candidate
        ).casefold()

        if key in seen_paths:
            continue

        seen_paths.add(key)

        if not candidate.exists():
            continue

        try:
            ds = read_dicom_header(
                candidate
            )

            uid = safe_str(
                getattr(
                    ds,
                    "SOPInstanceUID",
                    "",
                )
            )

        except Exception:
            continue

        if (
            uid
            and uid
            in ORIGINAL[
                "uid_to_paths"
            ]
            and len(
                ORIGINAL[
                    "uid_to_paths"
                ][uid]
            )
            == 1
        ):
            original_path = (
                ORIGINAL[
                    "uid_to_paths"
                ][uid][0]
            )

            info = original_info_from_path(
                original_path
            )

            return {
                **info,
                "resolve_method":
                    "uid_from_annotation_dicom",
                "resolved_via":
                    str(candidate),
            }

    return {
        "sop_uid": "",
        "canonical_relative_path": "",
        "canonical_original_path": "",
        "resolve_method": "not_found",
        "resolved_via": "",
    }


def read_labels_csv(
    source_name: str,
    folder: Path
):
    csv_path = folder / "labels.csv"

    if not csv_path.exists():
        print(
            f"{source_name}: "
            f"labels.csv отсутствует"
        )

        return pd.DataFrame()

    df = pd.read_csv(
        csv_path,
        dtype=str,
    ).fillna("")

    if (
        "relative_path"
        not in df.columns
    ):
        raise ValueError(
            f"{csv_path}: "
            f"нет relative_path"
        )

    if "label" not in df.columns:
        raise ValueError(
            f"{csv_path}: "
            f"нет label"
        )

    rows = []

    for i, row in df.iterrows():
        resolved = resolve_csv_row(
            source_name,
            folder,
            row,
        )

        rows.append({
            "source": source_name,
            "csv_path": str(csv_path),
            "csv_row_number": i + 2,
            "csv_relative_path": normalize_path(
                row.get(
                    "relative_path",
                    ""
                )
            ),
            "csv_output_path": normalize_path(
                row.get(
                    "output_path",
                    ""
                )
            ),
            "csv_label": safe_str(
                row.get(
                    "label",
                    ""
                )
            ).upper(),
            "csv_notes": safe_str(
                row.get(
                    "notes",
                    ""
                )
            ),
            "csv_annotator": safe_str(
                row.get(
                    "annotator",
                    ""
                )
            ),
            "csv_annotated_at": safe_str(
                row.get(
                    "annotated_at",
                    ""
                )
            ),
            **resolved,
        })

    return pd.DataFrame(
        rows
    )


csv_frames = []

for source, folder in (
    ANNOTATION_FOLDERS.items()
):
    if not folder.exists():
        continue

    frame = read_labels_csv(
        source,
        folder,
    )

    if not frame.empty:
        csv_frames.append(
            frame
        )


csv_df = (
    pd.concat(
        csv_frames,
        ignore_index=True
    )
    if csv_frames
    else pd.DataFrame()
)

csv_df.to_csv(
    REPORT_DIR
    / "02_labels_csv_resolved.csv",
    index=False,
    encoding="utf-8-sig",
)


unresolved_csv_df = (
    csv_df[
        csv_df[
            "resolve_method"
        ]
        == "not_found"
    ].copy()
    if not csv_df.empty
    else pd.DataFrame()
)

unresolved_csv_df.to_csv(
    REPORT_DIR
    / "03_unresolved_csv_rows.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 5. СОБИРАЕМ ЕДИНУЮ РАЗМЕТКУ ВНУТРИ КАЖДОГО SOURCE
# ============================================================

def choose_csv_row_for_uid(group):
    """
    Если в одном labels.csv один UID встречается несколько раз,
    стараемся взять наиболее позднюю запись.
    """

    group = group.copy()

    group[
        "_dt"
    ] = group[
        "csv_annotated_at"
    ].map(
        try_parse_datetime
    )

    if group[
        "_dt"
    ].notna().any():

        return (
            group.sort_values(
                [
                    "_dt",
                    "csv_row_number",
                ]
            )
            .iloc[-1]
        )

    return (
        group.sort_values(
            "csv_row_number"
        )
        .iloc[-1]
    )


def build_source_annotations(
    source_name
):
    """
    Возвращает 1 строку на SOP UID для одного source.

    Приоритет:
    1. labels.csv
    2. private DICOM label как fallback

    Если CSV и private tag расходятся,
    это явно отмечается.
    """

    rows = []

    csv_source = (
        csv_df[
            (csv_df["source"] == source_name)
            & (csv_df["sop_uid"] != "")
        ].copy()
        if not csv_df.empty
        else pd.DataFrame()
    )

    physical_source = (
        physical_df[
            (physical_df["source"] == source_name)
            & (physical_df["sop_uid"] != "")
        ].copy()
        if not physical_df.empty
        else pd.DataFrame()
    )

    all_uids = set()

    if not csv_source.empty:
        all_uids.update(
            csv_source[
                "sop_uid"
            ].unique()
        )

    if not physical_source.empty:
        all_uids.update(
            physical_source[
                "sop_uid"
            ].unique()
        )

    for uid in sorted(
        all_uids
    ):
        csv_group = (
            csv_source[
                csv_source[
                    "sop_uid"
                ]
                == uid
            ]
        )

        dcm_group = (
            physical_source[
                physical_source[
                    "sop_uid"
                ]
                == uid
            ]
        )

        csv_label = ""
        csv_notes = ""
        csv_annotator = ""
        csv_annotated_at = ""

        canonical_relative_path = ""

        if not csv_group.empty:
            selected_csv = (
                choose_csv_row_for_uid(
                    csv_group
                )
            )

            csv_label = safe_str(
                selected_csv[
                    "csv_label"
                ]
            ).upper()

            csv_notes = safe_str(
                selected_csv[
                    "csv_notes"
                ]
            )

            csv_annotator = safe_str(
                selected_csv[
                    "csv_annotator"
                ]
            )

            csv_annotated_at = safe_str(
                selected_csv[
                    "csv_annotated_at"
                ]
            )

            canonical_relative_path = (
                selected_csv[
                    "canonical_relative_path"
                ]
            )

        private_labels = set()

        private_notes = []
        private_annotators = []
        private_datetimes = []

        for _, dcm_row in (
            dcm_group.iterrows()
        ):
            label = safe_str(
                dcm_row[
                    "dicom_private_label"
                ]
            ).upper()

            if label:
                private_labels.add(
                    label
                )

            note = safe_str(
                dcm_row[
                    "dicom_private_notes"
                ]
            )

            if note:
                private_notes.append(
                    note
                )

            ann = safe_str(
                dcm_row[
                    "dicom_private_annotator"
                ]
            )

            if ann:
                private_annotators.append(
                    ann
                )

            dt = safe_str(
                dcm_row[
                    "dicom_private_datetime"
                ]
            )

            if dt:
                private_datetimes.append(
                    dt
                )

            if (
                not canonical_relative_path
                and safe_str(
                    dcm_row[
                        "canonical_relative_path"
                    ]
                )
            ):
                canonical_relative_path = (
                    dcm_row[
                        "canonical_relative_path"
                    ]
                )

        # Внутри DICOM одного source тоже может быть конфликт.
        if len(private_labels) == 1:
            private_label = next(
                iter(
                    private_labels
                )
            )
        elif len(private_labels) > 1:
            private_label = (
                "INTERNAL_DICOM_CONFLICT"
            )
        else:
            private_label = ""

        # Итоговая метка source.
        if csv_label:
            final_label = csv_label
            label_origin = "labels.csv"
        else:
            final_label = private_label
            label_origin = (
                "private_dicom"
                if private_label
                else "none"
            )

        internal_conflict = False

        if (
            csv_label
            and private_label
            and private_label
            != "INTERNAL_DICOM_CONFLICT"
            and csv_label
            != private_label
        ):
            internal_conflict = True

        if (
            private_label
            == "INTERNAL_DICOM_CONFLICT"
        ):
            internal_conflict = True

        original_match = (
            original_df[
                original_df[
                    "sop_uid"
                ]
                == uid
            ]
        )

        if (
            not original_match.empty
            and not canonical_relative_path
        ):
            canonical_relative_path = (
                original_match.iloc[0][
                    "relative_path"
                ]
            )

        rows.append({
            "source": source_name,
            "sop_uid": uid,
            "canonical_relative_path":
                canonical_relative_path,
            "label": final_label,
            "label_origin": label_origin,
            "csv_label": csv_label,
            "private_label": private_label,
            "internal_conflict":
                internal_conflict,
            "notes": (
                csv_notes
                or " | ".join(
                    sorted(
                        set(
                            private_notes
                        )
                    )
                )
            ),
            "annotator": (
                csv_annotator
                or " | ".join(
                    sorted(
                        set(
                            private_annotators
                        )
                    )
                )
            ),
            "annotated_at": (
                csv_annotated_at
                or (
                    sorted(
                        private_datetimes
                    )[-1]
                    if private_datetimes
                    else ""
                )
            ),
            "n_csv_rows":
                len(csv_group),
            "n_physical_dicom_copies":
                len(dcm_group),
        })

    return pd.DataFrame(
        rows
    )


source_annotation_frames = []

for source in ANNOTATION_FOLDERS:
    frame = build_source_annotations(
        source
    )

    if not frame.empty:
        source_annotation_frames.append(
            frame
        )


source_annotations_df = (
    pd.concat(
        source_annotation_frames,
        ignore_index=True
    )
    if source_annotation_frames
    else pd.DataFrame()
)

source_annotations_df.to_csv(
    REPORT_DIR
    / "04_annotations_by_source_uid.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 6. СТАТИСТИКА ПО ПАПКАМ
# ============================================================

summary_rows = []

for source, folder in (
    ANNOTATION_FOLDERS.items()
):
    source_ann = (
        source_annotations_df[
            source_annotations_df[
                "source"
            ]
            == source
        ]
        if not source_annotations_df.empty
        else pd.DataFrame()
    )

    source_csv = (
        csv_df[
            csv_df[
                "source"
            ]
            == source
        ]
        if not csv_df.empty
        else pd.DataFrame()
    )

    source_physical = (
        physical_df[
            physical_df[
                "source"
            ]
            == source
        ]
        if not physical_df.empty
        else pd.DataFrame()
    )

    label_counts = (
        source_ann[
            "label"
        ]
        .value_counts()
        if not source_ann.empty
        else pd.Series(
            dtype=int
        )
    )

    summary_rows.append({
        "source": source,
        "folder_exists": folder.exists(),
        "physical_dicom_count":
            len(source_physical),
        "labels_csv_rows":
            len(source_csv),
        "csv_rows_resolved":
            (
                source_csv[
                    "sop_uid"
                ]
                .ne("")
                .sum()
                if not source_csv.empty
                else 0
            ),
        "csv_rows_unresolved":
            (
                source_csv[
                    "sop_uid"
                ]
                .eq("")
                .sum()
                if not source_csv.empty
                else 0
            ),
        "unique_annotated_uids":
            source_ann[
                "sop_uid"
            ].nunique()
            if not source_ann.empty
            else 0,
        "LEG":
            int(
                label_counts.get(
                    "LEG",
                    0
                )
            ),
        "SPINE":
            int(
                label_counts.get(
                    "SPINE",
                    0
                )
            ),
        "UNKNOWN":
            int(
                label_counts.get(
                    "UNKNOWN",
                    0
                )
            ),
        "internal_conflicts":
            int(
                source_ann[
                    "internal_conflict"
                ].sum()
            )
            if not source_ann.empty
            else 0,
    })


source_summary_df = pd.DataFrame(
    summary_rows
)

source_summary_df.to_csv(
    REPORT_DIR
    / "05_source_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


print("\n")
print("=" * 100)
print("СТАТИСТИКА ПО ИСТОЧНИКАМ")
print("=" * 100)

print(
    source_summary_df.to_string(
        index=False
    )
)


# ============================================================
# 7. ПЕРЕСЕЧЕНИЯ МЕЖДУ SOURCE
# ============================================================

uid_sets = {}

for source in (
    ANNOTATION_FOLDERS
):
    if source_annotations_df.empty:
        uid_sets[source] = set()
        continue

    uid_sets[source] = set(
        source_annotations_df.loc[
            source_annotations_df[
                "source"
            ]
            == source,
            "sop_uid",
        ]
    )


pairwise_rows = []

for a, b in combinations(
    ANNOTATION_FOLDERS.keys(),
    2
):
    set_a = uid_sets[a]
    set_b = uid_sets[b]

    intersection = (
        set_a
        & set_b
    )

    union = (
        set_a
        | set_b
    )

    pairwise_rows.append({
        "source_1": a,
        "source_2": b,
        "n_1": len(set_a),
        "n_2": len(set_b),
        "intersection":
            len(intersection),
        "only_1":
            len(set_a - set_b),
        "only_2":
            len(set_b - set_a),
        "union":
            len(union),
        "jaccard":
            (
                len(intersection)
                / len(union)
                if union
                else 0
            ),
    })


pairwise_df = pd.DataFrame(
    pairwise_rows
)

pairwise_df.to_csv(
    REPORT_DIR
    / "06_pairwise_overlap.csv",
    index=False,
    encoding="utf-8-sig",
)


print("\n")
print("=" * 100)
print("ПОПАРНЫЕ ПЕРЕСЕЧЕНИЯ")
print("=" * 100)

print(
    pairwise_df.to_string(
        index=False
    )
)


non_empty_uid_sets = [
    s
    for s
    in uid_sets.values()
    if s
]

if non_empty_uid_sets:
    union_all = set.union(
        *non_empty_uid_sets
    )

    intersection_all = set.intersection(
        *non_empty_uid_sets
    )
else:
    union_all = set()
    intersection_all = set()


print("\n")
print(
    "Всего уникальных DICOM с какой-либо разметкой:",
    len(union_all)
)

print(
    "DICOM, размеченных во всех непустых источниках:",
    len(intersection_all)
)


# ============================================================
# 8. МЕЖИСТОЧНИКОВЫЕ КОНФЛИКТЫ
# ============================================================

comparison_rows = []

for uid in sorted(
    union_all
):
    group = (
        source_annotations_df[
            source_annotations_df[
                "sop_uid"
            ]
            == uid
        ]
    )

    labels_by_source = {
        row["source"]:
            safe_str(
                row["label"]
            ).upper()
        for _, row
        in group.iterrows()
    }

    nonempty_labels = {
        label
        for label
        in labels_by_source.values()
        if label != ""
    }

    known = (
        nonempty_labels
        & KNOWN_LABELS
    )

    unknown_present = (
        "UNKNOWN"
        in nonempty_labels
    )

    if (
        "LEG" in known
        and "SPINE" in known
    ):
        status = "HARD_CONFLICT"

    elif len(known) == 1:
        if unknown_present:
            status = (
                "KNOWN_VS_UNKNOWN"
            )
        elif len(group) > 1:
            status = "AGREE"
        else:
            status = "SINGLE"

    elif (
        not known
        and unknown_present
    ):
        status = (
            "UNKNOWN_ONLY"
        )

    else:
        status = (
            "OTHER"
        )

    original_match = (
        original_df[
            original_df[
                "sop_uid"
            ]
            == uid
        ]
    )

    canonical_path = (
        original_match.iloc[0][
            "relative_path"
        ]
        if not original_match.empty
        else ""
    )

    row = {
        "sop_uid": uid,
        "canonical_relative_path":
            canonical_path,
        "n_sources": len(group),
        "status": status,
        "labels_all":
            " | ".join(
                sorted(
                    nonempty_labels
                )
            ),
    }

    for source in (
        ANNOTATION_FOLDERS
    ):
        row[source] = (
            labels_by_source.get(
                source,
                ""
            )
        )

    comparison_rows.append(
        row
    )


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    REPORT_DIR
    / "07_comparison_all_sources.csv",
    index=False,
    encoding="utf-8-sig",
)


hard_conflicts_df = (
    comparison_df[
        comparison_df[
            "status"
        ]
        == "HARD_CONFLICT"
    ].copy()
)

hard_conflicts_df.to_csv(
    REPORT_DIR
    / "08_hard_conflicts_LEG_vs_SPINE.csv",
    index=False,
    encoding="utf-8-sig",
)


print("\n")
print("=" * 100)
print("СОГЛАСОВАННОСТЬ")
print("=" * 100)

if not comparison_df.empty:
    print(
        comparison_df[
            "status"
        ]
        .value_counts()
        .to_string()
    )

print(
    "\nLEG ↔ SPINE конфликтов:",
    len(
        hard_conflicts_df
    )
)


# ============================================================
# 9. MASTER LABELS
# ============================================================

master_rows = []

for _, row in (
    comparison_df.iterrows()
):
    labels = {
        safe_str(
            row[source]
        ).upper()
        for source
        in ANNOTATION_FOLDERS
        if safe_str(
            row[source]
        )
    }

    known = (
        labels
        & KNOWN_LABELS
    )

    if (
        "LEG" in known
        and "SPINE" in known
    ):
        final_label = (
            "CONFLICT"
        )

        merge_status = (
            "manual_review"
        )

    elif len(known) == 1:
        final_label = next(
            iter(
                known
            )
        )

        if (
            "UNKNOWN"
            in labels
        ):
            merge_status = (
                "resolved_known_over_unknown"
            )
        elif row["n_sources"] > 1:
            merge_status = (
                "agreement"
            )
        else:
            merge_status = (
                "single_source"
            )

    elif (
        not known
        and "UNKNOWN"
        in labels
    ):
        final_label = (
            "UNKNOWN"
        )

        merge_status = (
            "unknown_only"
        )

    else:
        final_label = (
            "UNRESOLVED"
        )

        merge_status = (
            "manual_review"
        )

    master_rows.append({
        "sop_uid":
            row[
                "sop_uid"
            ],

        "relative_path":
            row[
                "canonical_relative_path"
            ],

        "label":
            final_label,

        "merge_status":
            merge_status,

        "n_sources":
            row[
                "n_sources"
            ],

        "labels_all":
            row[
                "labels_all"
            ],
    })


master_df = pd.DataFrame(
    master_rows
)

master_df.to_csv(
    REPORT_DIR
    / "09_labels_master_all.csv",
    index=False,
    encoding="utf-8-sig",
)


usable_master_df = (
    master_df[
        master_df[
            "label"
        ].isin(
            [
                "LEG",
                "SPINE",
            ]
        )
    ].copy()
)


usable_master_df.to_csv(
    REPORT_DIR
    / "10_labels_master_LEG_SPINE.csv",
    index=False,
    encoding="utf-8-sig",
)


print("\n")
print("=" * 100)
print("MASTER LABELS")
print("=" * 100)

if not master_df.empty:
    print(
        master_df[
            "label"
        ]
        .value_counts()
        .to_string()
    )

print(
    "\nВсего уникальных размеченных SOP UID:",
    len(
        master_df
    )
)

print(
    "Готово для LEG/SPINE обучения:",
    len(
        usable_master_df
    )
)

print(
    "Требуют ручного пересмотра:",
    int(
        master_df[
            "label"
        ].isin(
            [
                "CONFLICT",
                "UNRESOLVED",
            ]
        ).sum()
    )
)


# ============================================================
# 10. КРАТКИЙ TXT-ОТЧЁТ
# ============================================================

report_lines = []

report_lines.append(
    f"Original DICOM count: "
    f"{len(original_df)}"
)

report_lines.append(
    f"Original unique SOPInstanceUID: "
    f"{original_df['sop_uid'][original_df['sop_uid'] != ''].nunique()}"
)

report_lines.append("")

for _, row in (
    source_summary_df.iterrows()
):
    report_lines.append(
        f"{row['source']}: "
        f"physical={row['physical_dicom_count']}, "
        f"csv_rows={row['labels_csv_rows']}, "
        f"resolved_csv={row['csv_rows_resolved']}, "
        f"unresolved_csv={row['csv_rows_unresolved']}, "
        f"unique_uid={row['unique_annotated_uids']}, "
        f"LEG={row['LEG']}, "
        f"SPINE={row['SPINE']}, "
        f"UNKNOWN={row['UNKNOWN']}"
    )

report_lines.append("")

report_lines.append(
    f"Union annotated UID: "
    f"{len(union_all)}"
)

report_lines.append(
    f"Intersection all non-empty sources: "
    f"{len(intersection_all)}"
)

report_lines.append(
    f"Hard LEG/SPINE conflicts: "
    f"{len(hard_conflicts_df)}"
)

report_lines.append(
    f"Usable master LEG/SPINE: "
    f"{len(usable_master_df)}"
)

(REPORT_DIR / "SUMMARY.txt").write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


print("\n")
print("=" * 100)
print("ГОТОВО")
print("=" * 100)

print(
    "Отчёты сохранены в:"
)

print(
    REPORT_DIR
)

print(
    "\nГлавный файл для прототипа:"
)

print(
    REPORT_DIR
    / "10_labels_master_LEG_SPINE.csv"
)

print(
    "\nКонфликты для ручной проверки:"
)

print(
    REPORT_DIR
    / "08_hard_conflicts_LEG_vs_SPINE.csv"
)
