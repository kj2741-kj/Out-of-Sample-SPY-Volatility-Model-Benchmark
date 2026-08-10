"""
Pull small reference series needed to independently replicate vendor-supplied
IV/Greeks: SPY underlying daily closes and a risk-free rate proxy (13-week
T-bill, ^IRX). Neither is in the options parquet (no spot price column there),
and both are tiny (~4000 rows) compared to the 23.5M-row options dataset, so
pulling live is fine even though the options data itself is locally sourced.
"""
import pandas as pd
import yfinance as yf

OUT = r"C:\Users\Kartavya\Desktop\NYU\Summer\Options Model Validation\data"

spy = yf.download("SPY", start="2010-01-01", end="2026-01-01", auto_adjust=False, progress=False)
spy = spy[["Close"]].reset_index()
spy.columns = ["date", "spy_close"]
spy.to_parquet(f"{OUT}\\spy_underlying_close.parquet", index=False)
print("SPY close rows:", len(spy), spy["date"].min(), spy["date"].max())

irx = yf.download("^IRX", start="2010-01-01", end="2026-01-01", auto_adjust=False, progress=False)
irx = irx[["Close"]].reset_index()
irx.columns = ["date", "irx_yield_pct"]
irx.to_parquet(f"{OUT}\\riskfree_irx.parquet", index=False)
print("IRX rows:", len(irx), irx["date"].min(), irx["date"].max())
