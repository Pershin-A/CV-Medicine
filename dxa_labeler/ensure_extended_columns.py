from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\Андрей\Desktop\Хакатон")
path = ROOT / "Размеченные" / "labels.csv"
df = pd.read_csv(path, dtype=str).fillna("")
for col in ["side", "metal", "fracture", "spine_issue"]:
    if col not in df.columns:
        df[col] = ""
df.to_csv(path, index=False, encoding="utf-8-sig")
print("Колонки labels.csv:", list(df.columns))
print("Строк:", len(df))
