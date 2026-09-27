from pathlib import Path
import argparse
import hashlib

import pandas as pd


def normalize(value):
    return str(value).replace("\\", "/").lstrip("./")


def find_dicoms(root: Path):
    return sorted(
        (
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() == ".dcm"
        ),
        key=lambda p: normalize(p.relative_to(root)).casefold(),
    )


def make_manifest(root: Path):
    rows = []

    for path in find_dicoms(root):
        rows.append({
            "relative_path": normalize(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
        })

    return pd.DataFrame(rows)


def fingerprint(df):
    h = hashlib.sha256()

    for row in df.itertuples(index=False):
        h.update(
            f"{row.relative_path}|{row.size_bytes}\n".encode(
                "utf-8",
                errors="replace",
            )
        )

    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument(
        "--reference",
        default="",
        help="Эталонный dataset_manifest.csv",
    )
    parser.add_argument(
        "--out",
        default="dataset_manifest.csv",
    )

    args = parser.parse_args()

    root = Path(args.data_root)
    current = make_manifest(root)

    current.to_csv(
        args.out,
        index=False,
        encoding="utf-8-sig",
    )

    print("DICOM:", len(current))
    print("Fingerprint:", fingerprint(current))
    print("Manifest:", args.out)

    if args.reference:
        ref = pd.read_csv(
            args.reference,
            dtype={"relative_path": str},
        )

        ref["relative_path"] = ref[
            "relative_path"
        ].map(normalize)

        current["relative_path"] = current[
            "relative_path"
        ].map(normalize)

        ref_paths = set(ref["relative_path"])
        current_paths = set(current["relative_path"])

        missing = sorted(
            ref_paths - current_paths
        )
        extra = sorted(
            current_paths - ref_paths
        )

        pd.DataFrame(
            {"relative_path": missing}
        ).to_csv(
            "missing_from_current.csv",
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame(
            {"relative_path": extra}
        ).to_csv(
            "extra_in_current.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("Missing:", len(missing))
        print("Extra:", len(extra))

        if missing:
            print("\nПервые отсутствующие:")
            for x in missing[:20]:
                print(" -", x)

        if extra:
            print("\nПервые лишние:")
            for x in extra[:20]:
                print(" +", x)


if __name__ == "__main__":
    main()
