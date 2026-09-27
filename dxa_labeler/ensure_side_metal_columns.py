from pathlib import Path
import pandas as pd

root = Path(r"C:\Users\Андрей\Desktop\Хакатон")
path = root / "Размеченные" / "labels.csv"
df = pd.read_csv(path, dtype=str).fillna("")
for col in ("side", "metal"):
    if col not in df.columns:
        df[col] = ""
df.to_csv(path, index=False, encoding="utf-8-sig")
print("Колонки labels.csv:", list(df.columns))
print("LEG:", (df["label"].str.upper() == "LEG").sum())
print("side размечено:", df["side"].str.upper().isin(["LEFT", "RIGHT"]).sum())
print("metal размечено:", df["metal"].astype(str).isin(["0", "1"]).sum())