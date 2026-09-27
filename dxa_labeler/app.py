from pathlib import Path
from datetime import datetime
import hashlib
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pydicom
import streamlit as st
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut


DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data"))
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", "/output"))

EXPECTED_DICOM_COUNT = int(os.getenv("EXPECTED_DICOM_COUNT", "0") or 0)
EXPECTED_DATASET_FINGERPRINT = os.getenv(
    "EXPECTED_DATASET_FINGERPRINT", ""
).strip().lower()
STRICT_DATASET_CHECK = os.getenv(
    "STRICT_DATASET_CHECK", "1"
).strip().lower() not in {"0", "false", "no"}

PRIVATE_GROUP = 0x0011
PRIVATE_CREATOR = "DXA_MANUAL_LABELER"

OFFSET_LABEL = 0x01
OFFSET_NOTES = 0x02
OFFSET_ANNOTATOR = 0x03
OFFSET_DATETIME = 0x04

LABELS = {
    "Не определено": "UNKNOWN",
    "Позвоночник": "SPINE",
    "Нога": "LEG",
}
LABELS_REVERSE = {v: k for k, v in LABELS.items()}


st.set_page_config(
    page_title="DXA DICOM Labeler",
    page_icon="🩻",
    layout="wide",
)


# ============================================================
# Пути, список файлов, fingerprint
# ============================================================

def normalize_relpath(value) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def find_dicoms(root: Path):
    # Кэш намеренно НЕ используем.
    # Если данные были докопированы после запуска контейнера,
    # приложение должно сразу увидеть новый список.
    return sorted(
        (
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() == ".dcm"
        ),
        key=lambda p: normalize_relpath(p.relative_to(root)).casefold(),
    )


def relative_path_string(source_path: Path) -> str:
    return normalize_relpath(source_path.relative_to(DATA_ROOT))


def dataset_manifest(files):
    rows = []
    for path in files:
        try:
            size = path.stat().st_size
        except OSError:
            size = -1

        rows.append({
            "relative_path": relative_path_string(path),
            "size_bytes": size,
        })

    return pd.DataFrame(rows)


def dataset_fingerprint(manifest_df: pd.DataFrame) -> str:
    h = hashlib.sha256()

    for row in manifest_df.itertuples(index=False):
        line = f"{row.relative_path}|{row.size_bytes}\n"
        h.update(line.encode("utf-8", errors="replace"))

    return h.hexdigest()


def output_path_for(source_path: Path) -> Path:
    return OUTPUT_ROOT / source_path.relative_to(DATA_ROOT)


def effective_path_for(source_path: Path) -> Path:
    annotated = output_path_for(source_path)
    return annotated if annotated.exists() else source_path


# ============================================================
# CSV разметки: labels.csv = основной реестр
# ============================================================

def labels_csv_path() -> Path:
    return OUTPUT_ROOT / "labels.csv"


def history_csv_path() -> Path:
    return OUTPUT_ROOT / "labels_history.csv"


def read_labels_df() -> pd.DataFrame:
    path = labels_csv_path()

    if not path.exists():
        return pd.DataFrame(columns=[
            "relative_path",
            "label",
            "notes",
            "annotator",
            "annotated_at",
            "output_path",
        ])

    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:
        st.error(
            f"Не удалось прочитать существующий labels.csv: {e}\n\n"
            "Файл не перезаписывается автоматически. "
            "Проверьте labels.csv и labels.csv.bak в папке результатов."
        )
        st.stop()

    if "relative_path" not in df.columns:
        st.error(
            "В существующем labels.csv нет колонки relative_path. "
            "Чтобы не повредить уже сделанную разметку, сохранение остановлено."
        )
        st.stop()

    df["relative_path"] = df["relative_path"].map(normalize_relpath)

    # Если по одному пути по какой-то причине несколько строк,
    # последняя строка считается текущим состоянием.
    df = df.drop_duplicates(
        subset=["relative_path"],
        keep="last",
    )

    return df.reset_index(drop=True)


def labels_lookup(df: pd.DataFrame):
    if df.empty:
        return {}

    return {
        normalize_relpath(row["relative_path"]): row
        for _, row in df.iterrows()
    }


def atomic_write_csv(df: pd.DataFrame, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=destination.stem + "_",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(fd)

    tmp_path = Path(tmp_name)

    try:
        df.to_csv(
            tmp_path,
            index=False,
            encoding="utf-8-sig",
        )
        os.replace(tmp_path, destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def append_history_row(row: dict):
    path = history_csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([row])

    if path.exists():
        try:
            old = pd.read_csv(path, dtype=str).fillna("")
            out = pd.concat([old, new_df], ignore_index=True)
        except Exception:
            # Не трогаем повреждённую историю; создаём отдельный recovery-файл.
            recovery = OUTPUT_ROOT / (
                "labels_history_recovery_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
                + ".csv"
            )
            new_df.to_csv(
                recovery,
                index=False,
                encoding="utf-8-sig",
            )
            return
    else:
        out = new_df

    atomic_write_csv(out, path)


def update_labels_csv(
    relative_path,
    label,
    notes,
    annotator,
    annotated_at,
    output_path,
):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    csv_path = labels_csv_path()
    backup_path = OUTPUT_ROOT / "labels.csv.bak"

    df = read_labels_df()

    # Перед каждым обновлением сохраняем предыдущую рабочую версию.
    if csv_path.exists():
        shutil.copy2(csv_path, backup_path)

    relative_path = normalize_relpath(relative_path)

    new_row = {
        "relative_path": relative_path,
        "label": label,
        "notes": notes,
        "annotator": annotator,
        "annotated_at": annotated_at,
        "output_path": normalize_relpath(
            Path(output_path).relative_to(OUTPUT_ROOT)
        ),
    }

    if not df.empty:
        df = df[
            df["relative_path"].map(normalize_relpath)
            != relative_path
        ]

    df = pd.concat(
        [df, pd.DataFrame([new_row])],
        ignore_index=True,
    )

    df["relative_path"] = df["relative_path"].map(normalize_relpath)
    df = df.sort_values(
        "relative_path",
        key=lambda s: s.str.casefold(),
    )

    atomic_write_csv(df, csv_path)
    append_history_row(new_row)


# ============================================================
# Private DICOM tags
# ============================================================

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
    block = get_private_block(ds, create=False)

    if block is None:
        return default

    try:
        tag = block.get_tag(offset)
    except Exception:
        return default

    if tag in ds:
        value = ds[tag].value
        return "" if value is None else str(value)

    return default


def annotation_from_dicom(source_path: Path):
    path = effective_path_for(source_path)

    try:
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=True,
        )
    except Exception:
        return {
            "label": "UNKNOWN",
            "notes": "",
            "annotator": "",
            "datetime": "",
        }

    return {
        "label": get_private_value(
            ds, OFFSET_LABEL, "UNKNOWN"
        ),
        "notes": get_private_value(
            ds, OFFSET_NOTES, ""
        ),
        "annotator": get_private_value(
            ds, OFFSET_ANNOTATOR, ""
        ),
        "datetime": get_private_value(
            ds, OFFSET_DATETIME, ""
        ),
    }


def read_existing_annotation(
    source_path: Path,
    lookup: dict,
):
    """
    labels.csv считается главным источником состояния.
    Это важно для уже размеченных данных:
    даже если размеченная DICOM-копия отсутствует/перемещена,
    существующая строка labels.csv не теряется.
    """
    rel = relative_path_string(source_path)

    if rel in lookup:
        row = lookup[rel]

        return {
            "label": str(row.get("label", "UNKNOWN")) or "UNKNOWN",
            "notes": str(row.get("notes", "")),
            "annotator": str(row.get("annotator", "")),
            "datetime": str(row.get("annotated_at", "")),
            "source": "labels.csv",
        }

    result = annotation_from_dicom(source_path)
    result["source"] = "DICOM"
    return result


# ============================================================
# DICOM image / metadata
# ============================================================

def dicom_to_image(ds):
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    try:
        arr = apply_voi_lut(arr, ds).astype(np.float32)
    except Exception:
        pass

    finite = np.isfinite(arr)

    if not finite.any():
        raise ValueError(
            "Изображение не содержит конечных значений."
        )

    lo = float(np.nanpercentile(arr[finite], 1))
    hi = float(np.nanpercentile(arr[finite], 99))

    if hi <= lo:
        lo = float(np.nanmin(arr[finite]))
        hi = float(np.nanmax(arr[finite]))

    if hi <= lo:
        image = np.zeros(arr.shape, dtype=np.uint8)
    else:
        image = np.clip(
            (arr - lo) / (hi - lo),
            0,
            1,
        )
        image = (image * 255).astype(np.uint8)

    if (
        str(getattr(ds, "PhotometricInterpretation", ""))
        == "MONOCHROME1"
    ):
        image = 255 - image

    return Image.fromarray(image)


def safe_value(element):
    try:
        value = element.value
    except Exception as e:
        return f"<error: {e}>"

    if isinstance(value, bytes):
        return f"<binary: {len(value)} bytes>"

    text = str(value)

    if len(text) > 1000:
        return (
            text[:1000]
            + f" ... <truncated; length={len(text)}>"
        )

    return text


def metadata_dataframe(ds):
    rows = []

    if getattr(ds, "file_meta", None):
        for element in ds.file_meta:
            rows.append({
                "Section": "FileMeta",
                "Tag": str(element.tag),
                "Keyword": element.keyword or "",
                "Name": element.name,
                "VR": element.VR,
                "Value": safe_value(element),
            })

    for element in ds.iterall():
        if element.tag == (0x7FE0, 0x0010):
            value = "<Pixel Data omitted>"
        else:
            value = safe_value(element)

        rows.append({
            "Section": "Dataset",
            "Tag": str(element.tag),
            "Keyword": element.keyword or "",
            "Name": element.name,
            "VR": element.VR,
            "Value": value,
        })

    return pd.DataFrame(rows)


# ============================================================
# Безопасное сохранение DICOM
# ============================================================

def atomic_save_dicom(ds, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = out_path.with_name(
        out_path.name + ".tmp"
    )

    try:
        ds.save_as(
            temp_path,
            enforce_file_format=True,
        )
        os.replace(temp_path, out_path)
    finally:
        temp_path.unlink(missing_ok=True)


def save_annotation(
    source_path: Path,
    label: str,
    notes: str,
    annotator: str,
):
    read_path = effective_path_for(source_path)

    ds = pydicom.dcmread(
        read_path,
        force=True,
    )

    block = get_private_block(
        ds,
        create=True,
    )

    if block is None:
        raise RuntimeError(
            "Не удалось создать private DICOM block "
            "для разметки."
        )

    label_tag = block.get_tag(OFFSET_LABEL)
    notes_tag = block.get_tag(OFFSET_NOTES)
    annotator_tag = block.get_tag(OFFSET_ANNOTATOR)
    datetime_tag = block.get_tag(OFFSET_DATETIME)

    now = datetime.now()
    dicom_dt = now.strftime("%Y%m%d%H%M%S.%f")
    iso_dt = now.isoformat(timespec="seconds")

    ds.add_new(
        label_tag,
        "CS",
        label,
    )
    ds.add_new(
        notes_tag,
        "LT",
        notes or "",
    )
    ds.add_new(
        annotator_tag,
        "LO",
        annotator or "",
    )
    ds.add_new(
        datetime_tag,
        "DT",
        dicom_dt,
    )

    out_path = output_path_for(source_path)

    # 1. Сначала безопасно пишем размеченную DICOM-копию.
    atomic_save_dicom(ds, out_path)

    # 2. Затем atomically обновляем labels.csv + backup + history.
    relative_path = relative_path_string(source_path)

    update_labels_csv(
        relative_path=relative_path,
        label=label,
        notes=notes or "",
        annotator=annotator or "",
        annotated_at=iso_dt,
        output_path=out_path,
    )

    return out_path


# ============================================================
# Навигация
# ============================================================

def move_to(index: int, files):
    index = max(
        0,
        min(index, len(files) - 1),
    )

    st.session_state.current_idx = index

    # Значение selectbox изменяем на следующем rerun,
    # до создания самого widget.
    st.session_state.pending_file_select = (
        relative_path_string(files[index])
    )


# ============================================================
# Startup / dataset validation
# ============================================================

st.title("🩻 DXA DICOM manual labeler")
st.caption(
    "Исходные DICOM читаются только из /data. "
    "Разметка хранится в /output. "
    "labels.csv является основным реестром разметки."
)

if not DATA_ROOT.exists():
    st.error(
        f"Каталог с данными не найден: {DATA_ROOT}\n\n"
        "Проверьте DXA_DATA_DIR в файле .env."
    )
    st.stop()

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

# КЭША НЕТ — список пересчитывается на каждом rerun.
files = find_dicoms(DATA_ROOT)
manifest_df = dataset_manifest(files)
fingerprint = dataset_fingerprint(manifest_df)

if not files:
    st.error(
        f"В {DATA_ROOT} не найдено файлов *.dcm"
    )
    st.stop()

relative_options = manifest_df[
    "relative_path"
].tolist()

labels_df = read_labels_df()
lookup = labels_lookup(labels_df)

current_paths = set(relative_options)

annotated_current = (
    set(labels_df["relative_path"].map(normalize_relpath))
    & current_paths
    if not labels_df.empty
    else set()
)

n_annotated = len(annotated_current)

count_ok = (
    EXPECTED_DICOM_COUNT <= 0
    or len(files) == EXPECTED_DICOM_COUNT
)

fingerprint_ok = (
    not EXPECTED_DATASET_FINGERPRINT
    or fingerprint.lower()
    == EXPECTED_DATASET_FINGERPRINT
)

dataset_ok = count_ok and fingerprint_ok


# ============================================================
# Session state before widgets
# ============================================================

if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0

st.session_state.current_idx = min(
    st.session_state.current_idx,
    len(files) - 1,
)

if "file_select" not in st.session_state:
    st.session_state.file_select = relative_options[
        st.session_state.current_idx
    ]

if "pending_file_select" in st.session_state:
    pending_value = st.session_state.pop(
        "pending_file_select"
    )

    if pending_value in relative_options:
        st.session_state.file_select = pending_value
        st.session_state.current_idx = (
            relative_options.index(pending_value)
        )

# Если список данных изменился и старое значение selectbox
# больше не существует, аккуратно возвращаемся к текущему индексу.
if st.session_state.file_select not in relative_options:
    st.session_state.current_idx = min(
        st.session_state.current_idx,
        len(relative_options) - 1,
    )
    st.session_state.file_select = relative_options[
        st.session_state.current_idx
    ]


def on_file_selected():
    selected = st.session_state.file_select

    try:
        st.session_state.current_idx = (
            relative_options.index(selected)
        )
    except ValueError:
        st.session_state.current_idx = 0


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Навигация")

    st.selectbox(
        "Выберите DICOM",
        options=relative_options,
        key="file_select",
        on_change=on_file_selected,
    )

    idx = st.session_state.current_idx

    st.write(
        f"Файл **{idx + 1} / {len(files)}**"
    )

    prev_col, next_col = st.columns(2)

    with prev_col:
        if st.button(
            "← Пред.",
            width="stretch",
            disabled=(idx == 0),
        ):
            move_to(
                idx - 1,
                files,
            )
            st.rerun()

    with next_col:
        if st.button(
            "След. →",
            width="stretch",
            disabled=(idx >= len(files) - 1),
        ):
            move_to(
                idx + 1,
                files,
            )
            st.rerun()

    st.divider()

    st.metric(
        "Размечено в текущем наборе",
        f"{n_annotated} / {len(files)}",
    )

    st.progress(
        min(
            n_annotated / len(files),
            1.0,
        )
    )

    if len(labels_df) != n_annotated:
        st.caption(
            f"Всего строк в labels.csv: {len(labels_df)}. "
            "Некоторые строки могут относиться к файлам, "
            "которых нет в текущем /data."
        )

    st.divider()

    st.subheader("Проверка набора")

    if count_ok:
        st.success(
            f"DICOM в контейнере: {len(files)}"
        )
    else:
        st.error(
            f"DICOM в контейнере: {len(files)}, "
            f"ожидалось: {EXPECTED_DICOM_COUNT}"
        )

    st.code(
        fingerprint[:16],
        language=None,
    )
    st.caption(
        "Fingerprint должен совпадать у всех разметчиков, "
        "если набор файлов один и тот же."
    )

    st.download_button(
        "Скачать manifest.csv",
        data=manifest_df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="dataset_manifest.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander(
        "Технические пути"
    ):
        st.write(
            f"DATA_ROOT: `{DATA_ROOT}`"
        )
        st.write(
            f"OUTPUT_ROOT: `{OUTPUT_ROOT}`"
        )
        st.write(
            f"labels.csv: `{labels_csv_path()}`"
        )


# ============================================================
# Stop new labeling if dataset check failed
# ============================================================

if not dataset_ok:
    st.error(
        "Проверка набора данных не пройдена. "
        "Разметка остановлена, чтобы не продолжать работу "
        "по неполному или другому списку файлов.\n\n"
        "Существующие labels.csv и размеченные DICOM НЕ изменены."
    )

    if STRICT_DATASET_CHECK:
        st.stop()
    else:
        st.warning(
            "STRICT_DATASET_CHECK=0: работа разрешена "
            "несмотря на несовпадение."
        )


# ============================================================
# Current file
# ============================================================

current_idx = st.session_state.current_idx
source_path = files[current_idx]
relative_path = relative_path_string(source_path)
read_path = effective_path_for(source_path)

try:
    ds = pydicom.dcmread(
        read_path,
        force=True,
    )
except Exception as e:
    st.error(
        f"Не удалось прочитать DICOM:\n\n{e}"
    )
    st.stop()

existing = read_existing_annotation(
    source_path,
    lookup,
)

top_left, top_right = st.columns(
    [1.05, 0.95],
    gap="large",
)

with top_left:
    st.subheader("Изображение")
    st.code(
        relative_path,
        language=None,
    )

    try:
        image = dicom_to_image(ds)

        st.image(
            image,
            caption=relative_path,
            width="stretch",
        )
    except Exception as e:
        st.error(
            "Не удалось декодировать PixelData.\n\n"
            f"{e}\n\n"
            "Если файл сжат, проверьте pylibjpeg."
        )


with top_right:
    st.subheader("Разметка")

    if relative_path in annotated_current:
        st.success(
            "Файл уже размечен. "
            f"Источник текущей метки: {existing['source']}."
        )
    elif output_path_for(source_path).exists():
        st.info(
            "Есть размеченная DICOM-копия, "
            "но записи для неё нет в текущем labels.csv."
        )

    initial_label = LABELS_REVERSE.get(
        existing["label"],
        "Не определено",
    )

    # Ключ теперь зависит от пути, а не от номера в списке.
    # Если список файлов изменился, значения widget не "переедут"
    # на другой DICOM.
    widget_suffix = hashlib.sha1(
        relative_path.encode(
            "utf-8",
            errors="replace",
        )
    ).hexdigest()[:12]

    label_ui = st.radio(
        "Класс",
        options=list(LABELS.keys()),
        index=list(LABELS.keys()).index(
            initial_label
        ),
        key=f"label_{widget_suffix}",
        horizontal=True,
    )

    annotator = st.text_input(
        "Разметчик",
        value=existing["annotator"],
        key=f"annotator_{widget_suffix}",
        placeholder="Например: Андрей",
    )

    notes = st.text_area(
        "Комментарий",
        value=existing["notes"],
        key=f"notes_{widget_suffix}",
        placeholder="Необязательно",
        height=120,
    )

    current_label = LABELS[
        label_ui
    ]

    st.write(
        "Будет записано в private DICOM metadata:"
    )

    st.code(
        "\n".join([
            f"Private Creator: {PRIVATE_CREATOR}",
            f"ManualLabel: {current_label}",
            f"Notes: {notes or '<empty>'}",
            f"Annotator: {annotator or '<empty>'}",
        ]),
        language=None,
    )

    save_col, save_next_col = st.columns(2)

    with save_col:
        if st.button(
            "💾 Сохранить",
            type="primary",
            width="stretch",
        ):
            try:
                out_path = save_annotation(
                    source_path=source_path,
                    label=current_label,
                    notes=notes,
                    annotator=annotator,
                )
                st.success(
                    f"Сохранено:\n{out_path}"
                )
                st.rerun()
            except Exception as e:
                st.exception(e)

    with save_next_col:
        if st.button(
            "💾 Сохранить и следующий →",
            width="stretch",
        ):
            try:
                save_annotation(
                    source_path=source_path,
                    label=current_label,
                    notes=notes,
                    annotator=annotator,
                )

                if current_idx < len(files) - 1:
                    move_to(
                        current_idx + 1,
                        files,
                    )
                    st.rerun()
                else:
                    st.success(
                        "Последний файл сохранён."
                    )
            except Exception as e:
                st.exception(e)


# ============================================================
# Metadata table
# ============================================================

st.divider()
st.subheader("DICOM metadata")

metadata_df = metadata_dataframe(ds)

metadata_query = st.text_input(
    "Поиск по metadata",
    placeholder=(
        "Например: Patient Orientation, "
        "0020, Image Comments..."
    ),
)

if metadata_query.strip():
    q = metadata_query.strip().lower()

    mask = metadata_df.astype(str).apply(
        lambda col: col.str.lower().str.contains(
            q,
            regex=False,
            na=False,
        )
    ).any(axis=1)

    display_metadata = metadata_df[
        mask
    ]
else:
    display_metadata = metadata_df

st.dataframe(
    display_metadata,
    width="stretch",
    hide_index=True,
    height=600,
)

with st.expander(
    "Техническая информация текущего DICOM"
):
    st.write(
        f"Источник: `{source_path}`"
    )
    st.write(
        f"Читается сейчас: `{read_path}`"
    )
    st.write(
        f"Размеченная копия: "
        f"`{output_path_for(source_path)}`"
    )
    st.write(
        f"Relative path: `{relative_path}`"
    )
    st.write(
        f"SOP Instance UID: "
        f"`{getattr(ds, 'SOPInstanceUID', 'N/A')}`"
    )
    st.write(
        f"Modality: "
        f"`{getattr(ds, 'Modality', 'N/A')}`"
    )
    st.write(
        f"Rows × Columns: "
        f"`{getattr(ds, 'Rows', '?')} × "
        f"{getattr(ds, 'Columns', '?')}`"
    )
