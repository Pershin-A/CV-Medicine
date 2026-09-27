from pathlib import Path
import argparse

import pandas as pd


def normalize_path(value):
    return str(value).replace("\\", "/").lstrip("./")


def load_one_labels(path: Path, source_name: str):
    df = pd.read_csv(
        path,
        dtype=str,
    ).fillna("")

    required = {
        "relative_path",
        "label",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"{path}: нужны колонки {sorted(required)}"
        )

    df["relative_path"] = df[
        "relative_path"
    ].map(normalize_path)

    if "annotated_at" not in df.columns:
        df["annotated_at"] = ""

    if "notes" not in df.columns:
        df["notes"] = ""

    if "annotator" not in df.columns:
        df["annotator"] = ""

    # Внутри одного файла оставляем последнее состояние пути.
    # labels.csv из приложения и так уникален,
    # но это делает merge устойчивым к старым версиям.
    df = df.drop_duplicates(
        subset=["relative_path"],
        keep="last",
    )

    df["source"] = source_name

    return df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Объединение labels.csv нескольких разметчиков "
            "по relative_path, а не по номеру файла."
        )
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Пути к labels.csv, например "
            "Размеченные_1/labels.csv Размеченные_2/labels.csv"
        ),
    )

    parser.add_argument(
        "--out-dir",
        default="merged_labels",
    )

    args = parser.parse_args()

    frames = []

    for input_name in args.inputs:
        path = Path(input_name)

        if not path.exists():
            raise FileNotFoundError(path)

        frames.append(
            load_one_labels(
                path,
                source_name=path.parent.name,
            )
        )

    all_rows = pd.concat(
        frames,
        ignore_index=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Аудит всех исходных строк
    # --------------------------------------------------------
    all_rows.to_csv(
        out_dir / "labels_all_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    merged_rows = []
    conflict_rows = []

    for relative_path, group in all_rows.groupby(
        "relative_path",
        sort=True,
    ):
        labels = sorted(
            set(
                x
                for x in group["label"].astype(str)
                if x
            )
        )

        # Один файл размечал один человек
        # или все разметчики дали одинаковую метку.
        if len(labels) <= 1:
            chosen = group.copy()

            # Если есть несколько одинаковых меток,
            # оставляем наиболее позднюю запись при наличии даты.
            if "annotated_at" in chosen.columns:
                chosen = chosen.sort_values(
                    "annotated_at"
                )

            row = chosen.iloc[-1].copy()
            row["n_annotations"] = len(group)
            row["agreement"] = "single" if len(group) == 1 else "agree"
            row["sources_all"] = " | ".join(
                sorted(set(group["source"].astype(str)))
            )
            merged_rows.append(row)

        # Разные классы для одного relative_path:
        # ничего автоматически не выбираем.
        else:
            for _, row in group.iterrows():
                item = row.copy()
                item["labels_all"] = " | ".join(labels)
                conflict_rows.append(item)

    merged_df = pd.DataFrame(merged_rows)
    conflicts_df = pd.DataFrame(conflict_rows)

    if not merged_df.empty:
        merged_df = merged_df.sort_values(
            "relative_path"
        )

    merged_df.to_csv(
        out_dir / "labels_merged.csv",
        index=False,
        encoding="utf-8-sig",
    )

    conflicts_df.to_csv(
        out_dir / "labels_conflicts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Всего уникальных путей: "
        f"{all_rows['relative_path'].nunique()}"
    )
    print(
        f"Без конфликтов: {len(merged_df)}"
    )
    print(
        f"Путей с конфликтами: "
        f"{conflicts_df['relative_path'].nunique() if not conflicts_df.empty else 0}"
    )

    print("\nРезультаты:")
    print(out_dir / "labels_merged.csv")
    print(out_dir / "labels_conflicts.csv")
    print(out_dir / "labels_all_rows.csv")


if __name__ == "__main__":
    main()
