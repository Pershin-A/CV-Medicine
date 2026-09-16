from pathlib import Path
from datetime import datetime
import csv
import os

import numpy as np
import pandas as pd
import pydicom
import streamlit as st
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut


DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data"))
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", "/output"))

PRIVATE_GROUP = 0x0011
PRIVATE_CREATOR = "DXA_MANUAL_LABELER"

# Offsets inside our private block
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


def find_dicoms(root: Path):
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".dcm"
    )


def output_path_for(source_path: Path) -> Path:
    relative = source_path.relative_to(DATA_ROOT)
    return OUTPUT_ROOT / relative


def effective_path_for(source_path: Path) -> Path:
    annotated = output_path_for(source_path)
    return annotated if annotated.exists() else source_path


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


def read_existing_annotation(source_path: Path):
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
        "label": get_private_value(ds, OFFSET_LABEL, "UNKNOWN"),
        "notes": get_private_value(ds, OFFSET_NOTES, ""),
        "annotator": get_private_value(ds, OFFSET_ANNOTATOR, ""),
        "datetime": get_private_value(ds, OFFSET_DATETIME, ""),
    }


def dicom_to_image(ds):
    arr = ds.pixel_array.astype(np.float32)

    # Apply Modality LUT / rescale if present.
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    # Apply Window Center / Width or VOI LUT where possible.
    try:
        arr = apply_voi_lut(arr, ds).astype(np.float32)
    except Exception:
        pass

    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("Изображение не содержит конечных значений.")

    lo = float(np.nanpercentile(arr[finite], 1))
    hi = float(np.nanpercentile(arr[finite], 99))

    if hi <= lo:
        lo = float(np.nanmin(arr[finite]))
        hi = float(np.nanmax(arr[finite]))

    if hi <= lo:
        image = np.zeros(arr.shape, dtype=np.uint8)
    else:
        image = np.clip((arr - lo) / (hi - lo), 0, 1)
        image = (image * 255).astype(np.uint8)

    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        image = 255 - image

    return Image.fromarray(image)


def metadata_dataframe(ds):
    rows = []

    # File meta
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

    # Full dataset, including nested sequence elements.
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


def safe_value(element):
    try:
        value = element.value
    except Exception as e:
        return f"<error: {e}>"

    if isinstance(value, bytes):
        return f"<binary: {len(value)} bytes>"

    text = str(value)
    if len(text) > 1000:
        return text[:1000] + f" ... <truncated; length={len(text)}>"

    return text


def update_labels_csv(
    relative_path,
    label,
    notes,
    annotator,
    annotated_at,
    output_path,
):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "labels.csv"

    new_row = {
        "relative_path": relative_path,
        "label": label,
        "notes": notes,
        "annotator": annotator,
        "annotated_at": annotated_at,
        "output_path": str(output_path),
    }

    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, dtype=str).fillna("")
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if not df.empty and "relative_path" in df.columns:
        df = df[df["relative_path"] != relative_path]

    df = pd.concat(
        [df, pd.DataFrame([new_row])],
        ignore_index=True,
    )

    df = df.sort_values("relative_path")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def save_annotation(
    source_path: Path,
    label: str,
    notes: str,
    annotator: str,
):
    # If the file has already been annotated, read the copy so edits are preserved.
    read_path = effective_path_for(source_path)

    ds = pydicom.dcmread(
        read_path,
        force=True,
    )

    block = get_private_block(ds, create=True)
    if block is None:
        raise RuntimeError(
            "Не удалось создать private DICOM block для разметки."
        )

    label_tag = block.get_tag(OFFSET_LABEL)
    notes_tag = block.get_tag(OFFSET_NOTES)
    annotator_tag = block.get_tag(OFFSET_ANNOTATOR)
    datetime_tag = block.get_tag(OFFSET_DATETIME)

    now = datetime.now()
    dicom_dt = now.strftime("%Y%m%d%H%M%S.%f")
    iso_dt = now.isoformat(timespec="seconds")

    # CS should be short and upper-case.
    ds.add_new(label_tag, "CS", label)
    ds.add_new(notes_tag, "LT", notes or "")
    ds.add_new(annotator_tag, "LO", annotator or "")
    ds.add_new(datetime_tag, "DT", dicom_dt)

    out_path = output_path_for(source_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve file format / transfer syntax.
    ds.save_as(out_path, enforce_file_format=True)

    relative_path = str(source_path.relative_to(DATA_ROOT))

    update_labels_csv(
        relative_path=relative_path,
        label=label,
        notes=notes or "",
        annotator=annotator or "",
        annotated_at=iso_dt,
        output_path=out_path,
    )

    return out_path


def move_to(index: int, files):
    index = max(0, min(index, len(files) - 1))
    st.session_state.current_idx = index
    st.session_state.file_select = str(
        files[index].relative_to(DATA_ROOT)
    )


@st.cache_data(show_spinner=False)
def cached_file_list(data_root_string):
    root = Path(data_root_string)
    return [str(p) for p in find_dicoms(root)]


st.title("🩻 DXA DICOM manual labeler")
st.caption(
    "Исходные DICOM читаются только из /data. "
    "Размеченные копии сохраняются в /output с сохранением структуры папок."
)

if not DATA_ROOT.exists():
    st.error(
        f"Каталог с данными не найден: {DATA_ROOT}\n\n"
        "Проверьте volume в docker-compose.yml."
    )
    st.stop()

file_strings = cached_file_list(str(DATA_ROOT))
files = [Path(p) for p in file_strings]

if not files:
    st.error(f"В {DATA_ROOT} не найдено файлов *.dcm")
    st.stop()

relative_options = [
    str(p.relative_to(DATA_ROOT))
    for p in files
]

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


def on_file_selected():
    selected = st.session_state.file_select
    try:
        st.session_state.current_idx = relative_options.index(selected)
    except ValueError:
        st.session_state.current_idx = 0


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
            use_container_width=True,
            disabled=(idx == 0),
        ):
            move_to(idx - 1, files)
            st.rerun()

    with next_col:
        if st.button(
            "След. →",
            use_container_width=True,
            disabled=(idx >= len(files) - 1),
        ):
            move_to(idx + 1, files)
            st.rerun()

    st.divider()

    labels_csv = OUTPUT_ROOT / "labels.csv"

    if labels_csv.exists():
        try:
            labels_df = pd.read_csv(labels_csv, dtype=str).fillna("")
            annotated_paths = set(labels_df["relative_path"])
            n_annotated = len(annotated_paths)
        except Exception:
            n_annotated = 0
    else:
        n_annotated = 0

    st.metric(
        "Размечено",
        f"{n_annotated} / {len(files)}",
    )

    st.progress(
        min(n_annotated / len(files), 1.0)
    )


current_idx = st.session_state.current_idx
source_path = files[current_idx]
relative_path = source_path.relative_to(DATA_ROOT)
read_path = effective_path_for(source_path)

try:
    ds = pydicom.dcmread(
        read_path,
        force=True,
    )
except Exception as e:
    st.error(f"Не удалось прочитать DICOM:\n\n{e}")
    st.stop()

existing = read_existing_annotation(source_path)

top_left, top_right = st.columns([1.05, 0.95], gap="large")

with top_left:
    st.subheader("Изображение")
    st.code(str(relative_path), language=None)

    try:
        image = dicom_to_image(ds)
        st.image(
            image,
            caption=str(relative_path),
            use_container_width=True,
        )
    except Exception as e:
        st.error(
            "Не удалось декодировать PixelData.\n\n"
            f"{e}\n\n"
            "Если файл сжат, проверьте установку pylibjpeg."
        )

with top_right:
    st.subheader("Разметка")

    if output_path_for(source_path).exists():
        st.success(
            "Для этого файла уже существует размеченная копия."
        )

    initial_label = LABELS_REVERSE.get(
        existing["label"],
        "Не определено",
    )

    # Widget keys are file-specific, so values don't leak between files.
    widget_suffix = str(current_idx)

    label_ui = st.radio(
        "Класс",
        options=list(LABELS.keys()),
        index=list(LABELS.keys()).index(initial_label),
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

    current_label = LABELS[label_ui]

    st.write("Будет записано в private DICOM metadata:")
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
            use_container_width=True,
        ):
            try:
                out_path = save_annotation(
                    source_path=source_path,
                    label=current_label,
                    notes=notes,
                    annotator=annotator,
                )
                st.success(f"Сохранено:\n{out_path}")
            except Exception as e:
                st.exception(e)

    with save_next_col:
        if st.button(
            "💾 Сохранить и следующий →",
            use_container_width=True,
        ):
            try:
                save_annotation(
                    source_path=source_path,
                    label=current_label,
                    notes=notes,
                    annotator=annotator,
                )

                if current_idx < len(files) - 1:
                    move_to(current_idx + 1, files)
                    st.rerun()
                else:
                    st.success("Последний файл сохранён.")
            except Exception as e:
                st.exception(e)


st.divider()

st.subheader("DICOM metadata")

metadata_df = metadata_dataframe(ds)

metadata_query = st.text_input(
    "Поиск по metadata",
    placeholder="Например: Patient Orientation, 0020, Image Comments...",
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

    display_metadata = metadata_df[mask]
else:
    display_metadata = metadata_df

st.dataframe(
    display_metadata,
    use_container_width=True,
    hide_index=True,
    height=600,
)

with st.expander("Техническая информация"):
    st.write(f"Источник: `{source_path}`")
    st.write(f"Читается сейчас: `{read_path}`")
    st.write(f"Размеченная копия: `{output_path_for(source_path)}`")
    st.write(
        f"SOP Instance UID: "
        f"`{getattr(ds, 'SOPInstanceUID', 'N/A')}`"
    )
    st.write(
        f"Modality: `{getattr(ds, 'Modality', 'N/A')}`"
    )
    st.write(
        f"Rows × Columns: "
        f"`{getattr(ds, 'Rows', '?')} × {getattr(ds, 'Columns', '?')}`"
    )
