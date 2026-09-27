from __future__ import annotations

from pathlib import Path, PurePosixPath
from datetime import datetime
import argparse
import shutil
import sys

import pandas as pd
import pydicom


# ============================================================
# НАСТРОЙКИ
# ============================================================

ROOT = Path(r"C:\Users\Андрей\Desktop\Хакатон")

ORIGINAL_ROOT = ROOT / "Исследования"
MASTER_CSV = ROOT / "labels_recovery_report" / "10_labels_master_LEG_SPINE.csv"
CONFLICTS_CSV = ROOT / "labels_recovery_report" / "08_hard_conflicts_LEG_vs_SPINE.csv"

ACTIVE_OUTPUT = ROOT / "Размеченные"

OLD_ANNOTATION_FOLDERS = [
    ROOT / "Размеченные",
    ROOT / "Размеченные_244_Илья",
    ROOT / "Размеченные_499_Илья",
    ROOT / "Размеченные_old",
]

EXPECTED_COUNT = 499
ALLOWED_LABELS = {"LEG", "SPINE"}

PRIVATE_GROUP = 0x0011
PRIVATE_CREATOR = "DXA_MANUAL_LABELER"
OFFSET_LABEL = 0x01

CSV_COLUMNS = [
    "relative_path",
    "label",
    "notes",
    "annotator",
    "annotated_at",
    "output_path",
]


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def normalize_path(value: str) -> str:
    s = str(value or "").strip().replace("\\", "/")

    while "//" in s:
        s = s.replace("//", "/")

    return s.lstrip("./")


def relative_to_local_path(relative_path: str) -> Path:
    return Path(
        *PurePosixPath(
            normalize_path(relative_path)
        ).parts
    )


def read_header(path: Path):
    return pydicom.dcmread(
        path,
        stop_before_pixels=True,
        force=True,
    )


def read_full(path: Path):
    return pydicom.dcmread(
        path,
        force=True,
    )


def set_private_label(ds, label: str):
    """
    Записываем только итоговую master-метку.
    """
    block = ds.private_block(
        PRIVATE_GROUP,
        PRIVATE_CREATOR,
        create=True,
    )

    block.add_new(
        OFFSET_LABEL,
        "LO",
        str(label).strip(),
    )


def decode_private_value(value):
    if value is None:
        return ""

    if isinstance(value, bytes):
        value = value.decode(
            "utf-8",
            errors="replace",
        )

    return (
        str(value)
        .replace("\x00", "")
        .strip()
    )


def get_private_label(ds):
    try:
        block = ds.private_block(
            PRIVATE_GROUP,
            PRIVATE_CREATOR,
            create=False,
        )
    except Exception:
        return ""

    if block is None:
        return ""

    try:
        tag = block.get_tag(
            OFFSET_LABEL
        )
    except Exception:
        return ""

    if tag not in ds:
        return ""

    return decode_private_value(
        ds[tag].value
    ).upper()


# ============================================================
# 1. ВАЛИДАЦИЯ MASTER CSV
# ============================================================

def validate_master():
    print("=" * 100)
    print("1. ПРОВЕРКА ИСХОДНЫХ DICOM И MASTER-РАЗМЕТКИ")
    print("=" * 100)

    if not ORIGINAL_ROOT.exists():
        raise RuntimeError(
            f"Нет папки исходных DICOM: {ORIGINAL_ROOT}"
        )

    if not MASTER_CSV.exists():
        raise RuntimeError(
            f"Нет master labels: {MASTER_CSV}"
        )

    original_files = sorted(
        ORIGINAL_ROOT.rglob("*.dcm")
    )

    print("Исходных *.dcm:", len(original_files))

    if len(original_files) != EXPECTED_COUNT:
        raise RuntimeError(
            f"Ожидалось {EXPECTED_COUNT} исходных DICOM, "
            f"найдено {len(original_files)}."
        )

    master = pd.read_csv(
        MASTER_CSV,
        dtype=str,
    ).fillna("")

    required = {"relative_path", "label"}
    missing_columns = required - set(master.columns)

    if missing_columns:
        raise RuntimeError(
            f"В master CSV нет колонок: {sorted(missing_columns)}"
        )

    master["relative_path"] = (
        master["relative_path"]
        .map(normalize_path)
    )

    master["label"] = (
        master["label"]
        .str.strip()
        .str.upper()
    )

    print("Строк в master CSV:", len(master))
    print("\nРаспределение классов:")
    print(master["label"].value_counts().to_string())

    if len(master) != EXPECTED_COUNT:
        raise RuntimeError(
            f"В master CSV должно быть {EXPECTED_COUNT} строк, "
            f"сейчас {len(master)}."
        )

    duplicate_paths = master[
        master["relative_path"].duplicated(
            keep=False
        )
    ]

    if len(duplicate_paths):
        duplicate_paths.to_csv(
            ROOT / "FINALIZE_duplicate_relative_paths.csv",
            index=False,
            encoding="utf-8-sig",
        )

        raise RuntimeError(
            "В master CSV есть повторяющиеся relative_path. "
            "См. FINALIZE_duplicate_relative_paths.csv"
        )

    bad_labels = master[
        ~master["label"].isin(ALLOWED_LABELS)
    ]

    if len(bad_labels):
        bad_labels.to_csv(
            ROOT / "FINALIZE_bad_labels.csv",
            index=False,
            encoding="utf-8-sig",
        )

        raise RuntimeError(
            "В master CSV присутствуют метки кроме LEG/SPINE. "
            "См. FINALIZE_bad_labels.csv"
        )

    if CONFLICTS_CSV.exists():
        conflicts = pd.read_csv(
            CONFLICTS_CSV,
            dtype=str,
        )

        if len(conflicts):
            raise RuntimeError(
                f"Найдены LEG/SPINE конфликты: {len(conflicts)}. "
                f"Сначала разберите {CONFLICTS_CSV}"
            )

        print("LEG/SPINE конфликтов: 0")

    original_rel_set = {
        normalize_path(
            p.relative_to(ORIGINAL_ROOT)
        )
        for p in original_files
    }

    master_rel_set = set(master["relative_path"])

    missing_in_master = original_rel_set - master_rel_set
    extra_in_master = master_rel_set - original_rel_set

    if missing_in_master:
        pd.DataFrame({
            "relative_path": sorted(missing_in_master)
        }).to_csv(
            ROOT / "FINALIZE_missing_in_master.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if extra_in_master:
        pd.DataFrame({
            "relative_path": sorted(extra_in_master)
        }).to_csv(
            ROOT / "FINALIZE_extra_in_master.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if missing_in_master or extra_in_master:
        raise RuntimeError(
            "Master CSV и Исследования содержат разные relative_path. "
            "См. FINALIZE_missing_in_master.csv / FINALIZE_extra_in_master.csv"
        )

    uid_errors = []

    for _, row in master.iterrows():
        relative_path = row["relative_path"]

        original_path = (
            ORIGINAL_ROOT
            / relative_to_local_path(relative_path)
        )

        if not original_path.exists():
            uid_errors.append({
                "relative_path": relative_path,
                "error": "original file not found",
            })
            continue

        if (
            "sop_uid" in master.columns
            and str(row.get("sop_uid", "")).strip()
        ):
            ds = read_header(original_path)

            actual_uid = str(
                getattr(ds, "SOPInstanceUID", "")
            ).strip()

            expected_uid = str(
                row["sop_uid"]
            ).strip()

            if actual_uid != expected_uid:
                uid_errors.append({
                    "relative_path": relative_path,
                    "master_sop_uid": expected_uid,
                    "actual_sop_uid": actual_uid,
                    "error": "SOPInstanceUID mismatch",
                })

    if uid_errors:
        pd.DataFrame(uid_errors).to_csv(
            ROOT / "FINALIZE_uid_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )

        raise RuntimeError(
            "Есть несоответствия SOPInstanceUID. "
            "См. FINALIZE_uid_errors.csv"
        )

    print("\nOK:")
    print(f"- {EXPECTED_COUNT} исходных DICOM")
    print(f"- {EXPECTED_COUNT} уникальных relative_path")
    print("- только LEG/SPINE")
    print("- master полностью совпадает с исходным набором")

    return master


# ============================================================
# 2. СОЗДАНИЕ ЧИСТОЙ ПАПКИ
# ============================================================

def build_clean_folder(master: pd.DataFrame, temp_root: Path):
    print("\n")
    print("=" * 100)
    print("2. СОЗДАНИЕ ЧИСТОЙ КОНСОЛИДИРОВАННОЙ ПАПКИ")
    print("=" * 100)

    if temp_root.exists():
        shutil.rmtree(temp_root)

    temp_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    labels_rows = []

    for idx, row in master.iterrows():
        relative_path = normalize_path(
            row["relative_path"]
        )

        label = str(
            row["label"]
        ).strip().upper()

        src = (
            ORIGINAL_ROOT
            / relative_to_local_path(relative_path)
        )

        dst = (
            temp_root
            / relative_to_local_path(relative_path)
        )

        dst.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Берём чистый оригинал и добавляем только итоговую метку.
        ds = read_full(src)
        set_private_label(ds, label)

        ds.save_as(
            dst,
            write_like_original=False,
        )

        labels_rows.append({
            "relative_path": relative_path,
            "label": label,
            "notes": "",
            "annotator": "",
            "annotated_at": "",
            "output_path": relative_path,
        })

        if (
            (idx + 1) % 25 == 0
            or idx + 1 == len(master)
        ):
            print(
                f"\rСоздано: {idx + 1}/{len(master)}",
                end="",
            )

    print()

    labels_path = temp_root / "labels.csv"

    pd.DataFrame(
        labels_rows,
        columns=CSV_COLUMNS,
    ).to_csv(
        labels_path,
        index=False,
        encoding="utf-8-sig",
    )

    return labels_path


# ============================================================
# 3. ПРОВЕРКА СОЗДАННОЙ ПАПКИ
# ============================================================

def validate_clean_folder(
    master: pd.DataFrame,
    temp_root: Path,
):
    print("\n")
    print("=" * 100)
    print("3. ПРОВЕРКА НОВОЙ ПАПКИ")
    print("=" * 100)

    files = sorted(
        temp_root.rglob("*.dcm")
    )

    labels_path = temp_root / "labels.csv"

    if not labels_path.exists():
        raise RuntimeError(
            "В новой папке нет labels.csv"
        )

    new_labels = pd.read_csv(
        labels_path,
        dtype=str,
    ).fillna("")

    print("DICOM:", len(files))
    print("labels.csv:", len(new_labels))

    if len(files) != EXPECTED_COUNT:
        raise RuntimeError(
            f"В новой папке {len(files)} DICOM вместо {EXPECTED_COUNT}"
        )

    if len(new_labels) != EXPECTED_COUNT:
        raise RuntimeError(
            f"В новом labels.csv {len(new_labels)} строк "
            f"вместо {EXPECTED_COUNT}"
        )

    label_errors = []
    uid_errors = []

    master_by_path = master.set_index("relative_path")

    for idx, path in enumerate(files, start=1):
        rel = normalize_path(
            path.relative_to(temp_root)
        )

        ds = read_header(path)

        actual_label = get_private_label(ds)

        expected_label = (
            master_by_path
            .loc[
                rel,
                "label"
            ]
        )

        if actual_label != expected_label:
            label_errors.append({
                "relative_path": rel,
                "expected_label": expected_label,
                "actual_private_label": actual_label,
            })

        original_path = (
            ORIGINAL_ROOT
            / relative_to_local_path(rel)
        )

        original_ds = read_header(original_path)

        new_uid = str(
            getattr(ds, "SOPInstanceUID", "")
        ).strip()

        original_uid = str(
            getattr(original_ds, "SOPInstanceUID", "")
        ).strip()

        if new_uid != original_uid:
            uid_errors.append({
                "relative_path": rel,
                "original_sop_uid": original_uid,
                "new_sop_uid": new_uid,
            })

        if (
            idx % 50 == 0
            or idx == len(files)
        ):
            print(
                f"\rПроверено: {idx}/{len(files)}",
                end="",
            )

    print()

    if label_errors:
        pd.DataFrame(
            label_errors
        ).to_csv(
            ROOT / "FINALIZE_private_label_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )

        raise RuntimeError(
            "Private labels новой папки не совпадают с master. "
            "См. FINALIZE_private_label_errors.csv"
        )

    if uid_errors:
        pd.DataFrame(
            uid_errors
        ).to_csv(
            ROOT / "FINALIZE_new_uid_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )

        raise RuntimeError(
            "SOPInstanceUID изменился при создании копий. "
            "См. FINALIZE_new_uid_errors.csv"
        )

    new_labels["relative_path"] = (
        new_labels["relative_path"]
        .map(normalize_path)
    )

    new_labels["label"] = (
        new_labels["label"]
        .str.strip()
        .str.upper()
    )

    master_simple = (
        master[
            [
                "relative_path",
                "label"
            ]
        ]
        .sort_values("relative_path")
        .reset_index(drop=True)
    )

    new_simple = (
        new_labels[
            [
                "relative_path",
                "label"
            ]
        ]
        .sort_values("relative_path")
        .reset_index(drop=True)
    )

    if not master_simple.equals(new_simple):
        raise RuntimeError(
            "Новый labels.csv не совпадает с master."
        )

    print("OK:")
    print("- 499 DICOM")
    print("- 499 строк labels.csv")
    print("- private label каждого DICOM совпадает с labels.csv")
    print("- SOPInstanceUID сохранены")
    print("- структура путей совпадает с Исследования")


# ============================================================
# 4. АКТИВАЦИЯ И АРХИВАЦИЯ СТАРЫХ ПАПОК
# ============================================================

def activate_clean_folder(temp_root: Path):
    print("\n")
    print("=" * 100)
    print("4. АКТИВАЦИЯ НОВОЙ РАЗМЕТКИ")
    print("=" * 100)

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    archive_root = (
        ROOT
        / f"Разметка_архив_{stamp}"
    )

    archive_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    moved = []

    for folder in OLD_ANNOTATION_FOLDERS:
        if not folder.exists():
            continue

        destination = archive_root / folder.name

        if destination.exists():
            suffix = 1

            while (
                archive_root
                / f"{folder.name}_{suffix}"
            ).exists():
                suffix += 1

            destination = (
                archive_root
                / f"{folder.name}_{suffix}"
            )

        print(
            f"Архивирую:\n"
            f"  {folder}\n"
            f"  -> {destination}"
        )

        shutil.move(
            str(folder),
            str(destination),
        )

        moved.append(
            (
                folder,
                destination
            )
        )

    try:
        print(
            f"\nАктивирую:\n"
            f"  {temp_root}\n"
            f"  -> {ACTIVE_OUTPUT}"
        )

        shutil.move(
            str(temp_root),
            str(ACTIVE_OUTPUT),
        )

    except Exception:
        print("\nОШИБКА ПРИ АКТИВАЦИИ.")
        print("Пытаюсь вернуть старые папки из архива...")

        for original, archived in reversed(moved):
            if archived.exists() and not original.exists():
                shutil.move(
                    str(archived),
                    str(original),
                )

        raise

    print("\nГотово.")
    print("Активная папка:")
    print(ACTIVE_OUTPUT)
    print("\nАрхив предыдущих вариантов:")
    print(archive_root)

    return archive_root


# ============================================================
# 5. ФИНАЛЬНАЯ ПРОВЕРКА
# ============================================================

def final_check():
    print("\n")
    print("=" * 100)
    print("5. ФИНАЛЬНАЯ ПРОВЕРКА")
    print("=" * 100)

    files = sorted(
        ACTIVE_OUTPUT.rglob("*.dcm")
    )

    labels_path = ACTIVE_OUTPUT / "labels.csv"

    labels = pd.read_csv(
        labels_path,
        dtype=str,
    ).fillna("")

    label_counts = labels["label"].value_counts()

    print("Активная папка:", ACTIVE_OUTPUT)
    print("DICOM:", len(files))
    print("labels.csv:", len(labels))
    print("\nКлассы:")
    print(label_counts.to_string())

    other_active_annotation_dirs = []

    for folder in ROOT.iterdir():
        if (
            folder.is_dir()
            and folder.name.startswith("Размеченные")
            and folder != ACTIVE_OUTPUT
        ):
            other_active_annotation_dirs.append(folder)

    print(
        "\nДругих активных папок 'Размеченные*' в корне:",
        len(other_active_annotation_dirs)
    )

    for folder in other_active_annotation_dirs:
        print(" -", folder)

    if (
        len(files) != EXPECTED_COUNT
        or len(labels) != EXPECTED_COUNT
        or len(other_active_annotation_dirs) != 0
    ):
        raise RuntimeError(
            "Финальная проверка не пройдена."
        )

    print("\nФИНАЛЬНОЕ СОСТОЯНИЕ:")
    print(f"- одна активная папка: {ACTIVE_OUTPUT.name}")
    print(f"- {EXPECTED_COUNT} DICOM")
    print(f"- один labels.csv, {EXPECTED_COUNT} строк")
    print(f"- LEG: {int(label_counts.get('LEG', 0))}")
    print(f"- SPINE: {int(label_counts.get('SPINE', 0))}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Проверить master-разметку и собрать один "
            "чистый каталог 'Размеченные' с одним labels.csv."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "После полной проверки активировать новую папку "
            "и переместить старые варианты в архив. "
            "Без этого флага выполняется только проверка."
        ),
    )

    args = parser.parse_args()

    master = validate_master()

    if not args.apply:
        print("\n")
        print("=" * 100)
        print("DRY RUN УСПЕШНО ЗАВЕРШЁН")
        print("=" * 100)

        print("Разметка прошла базовую проверку.")
        print("Файлы НЕ изменялись.")

        print("\nДля создания единой папки выполните:")
        print("python finalize_dxa_labels.py --apply")

        return

    temp_root = ROOT / "_Размеченные_clean_tmp"

    build_clean_folder(
        master,
        temp_root,
    )

    validate_clean_folder(
        master,
        temp_root,
    )

    activate_clean_folder(
        temp_root,
    )

    final_check()


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print("\n")
        print("=" * 100)
        print("ОШИБКА")
        print("=" * 100)

        print(
            type(e).__name__ + ":",
            e,
        )

        sys.exit(1)
