from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\Андрей\Desktop\Хакатон")
path = ROOT / "Размеченные" / "labels.csv"

df = pd.read_csv(path, dtype=str).fillna("")
if "side" not in df.columns:
    df["side"] = ""
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("Добавлена колонка side.")
else:
    print("Колонка side уже существует.")

print(df["label"].value_counts())
print("Ног со стороной:", df.get("side", pd.Series(dtype=str)).isin(["LEFT", "RIGHT"]).sum())