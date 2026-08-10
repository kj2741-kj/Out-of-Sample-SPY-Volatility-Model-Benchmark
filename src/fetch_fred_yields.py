"""Pull real FRED Treasury constant-maturity yields for the Vasicek fit (Phase 2).
Public fredgraph.csv endpoint, no API key needed. 3mo is the primary series the
Vasicek short-rate model is fit to; 1yr/2yr are stored alongside for a possible
future term-structure extension, not used by single-factor Vasicek itself.
"""
import pandas as pd

OUT = r"C:\Users\Kartavya\Desktop\NYU\Summer\Options Model Validation\data"
SERIES = {"DGS3MO": "y3mo", "DGS1": "y1yr", "DGS2": "y2yr", "DGS3": "y3yr"}

frames = []
for series_id, col in SERIES.items():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=2010-01-01&coed=2025-12-31"
    df = pd.read_csv(url)
    df.columns = ["date", col]
    df["date"] = pd.to_datetime(df["date"])
    df[col] = pd.to_numeric(df[col], errors="coerce")  # FRED uses "." for holidays/missing
    frames.append(df.set_index("date"))
    print(series_id, "->", col, len(df), "rows,", df[col].isna().sum(), "missing")

merged = pd.concat(frames, axis=1).reset_index()
merged.to_parquet(f"{OUT}\\fred_treasury_yields.parquet", index=False)
print(merged.describe())
print(merged.head())
print(merged.tail())
