import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------
output_dir = Path(r"C: 'Filepath' ") #Add your filepath
output_dir.mkdir(parents=True, exist_ok=True)

detail_file = output_dir / "price_elasticity_detail_union.csv"
summary_file = output_dir / "price_elasticity_summary_union.csv"

# ------------------------------------------------------------
# Sample input data
# ------------------------------------------------------------
data = [
    ["A", "2024-01", 100, 1000],
    ["A", "2024-02", 110, 950],
    ["A", "2024-03", 103, 970],
    ["A", "2024-04", 105, 960],
    ["A", "2024-05", 108, 940],
    ["A", "2024-06", 102, 980],
    ["A", "2024-07", 98, 1020],
    ["A", "2024-08", 95, 1050],
    ["A", "2024-09", 97, 1030],
    ["A", "2024-10", 101, 990],
    ["A", "2024-11", 106, 955],
    ["A", "2024-12", 112, 920],
    ["B", "2024-01", 60, 1500],
    ["B", "2024-02", 65, 1480],
    ["B", "2024-03", 63, 1490],
    ["B", "2024-04", 64, 1475],
    ["B", "2024-05", 66, 1460],
    ["B", "2024-06", 62, 1505],
    ["B", "2024-07", 59, 1530],
    ["B", "2024-08", 57, 1550],
    ["B", "2024-09", 58, 1540],
    ["B", "2024-10", 61, 1510],
    ["B", "2024-11", 64, 1490],
    ["B", "2024-12", 68, 1450],
]

df = pd.DataFrame(data, columns=["Product", "Period", "Price", "Volume"])

df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

# ------------------------------------------------------------
# Price Elasticity Preparation
# ------------------------------------------------------------
def process_product(product_df):
    product_df = product_df.copy()

    product_df = product_df.sort_values("Period").reset_index(drop=True)

    product_df["Prev_Price"] = product_df["Price"].shift(1)
    product_df["Prev_Volume"] = product_df["Volume"].shift(1)

    product_df["Pct_Chg_Price"] = (
        (product_df["Price"] - product_df["Prev_Price"]) / product_df["Prev_Price"]
    )

    product_df["Pct_Chg_Volume"] = (
        (product_df["Volume"] - product_df["Prev_Volume"]) / product_df["Prev_Volume"]
    )

    product_df["Elasticity"] = np.where(
        product_df["Pct_Chg_Price"] != 0,
        product_df["Pct_Chg_Volume"] / product_df["Pct_Chg_Price"],
        np.nan
    )

    product_df["Log_Price"] = np.log(product_df["Price"])
    product_df["Log_Volume"] = np.log(product_df["Volume"])

    detail_filtered = product_df[
        product_df["Prev_Price"].notna()
        & product_df["Prev_Volume"].notna()
        & (product_df["Prev_Price"] != 0)
        & (product_df["Prev_Volume"] != 0)
        & (product_df["Pct_Chg_Price"] != 0)
    ].copy()

    detail_filtered["Elasticity_Weighted"] = (
        detail_filtered["Elasticity"] * detail_filtered["Volume"]
    )

    summary = pd.DataFrame({
        "Product": [detail_filtered["Product"].iloc[0]],
        "Avg_Elasticity": [detail_filtered["Elasticity"].mean()],
        "Total_Elasticity_Weighted": [detail_filtered["Elasticity_Weighted"].sum()],
        "Total_Volume": [detail_filtered["Volume"].sum()]
    })

    summary["weighted elasticity by product"] = (
        summary["Total_Elasticity_Weighted"] / summary["Total_Volume"]
    )

    return detail_filtered, summary


# ------------------------------------------------------------
# Loop Based on Products
# ------------------------------------------------------------
all_detail = []
all_summary = []

for product_value, group in df.groupby("Product", sort=False):
    detail_out, summary_out = process_product(group)
    all_detail.append(detail_out)
    all_summary.append(summary_out)

detail_union = pd.concat(all_detail, ignore_index=True)
summary_union = pd.concat(all_summary, ignore_index=True)

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------
detail_union.to_csv(detail_file, index=False)
summary_union.to_csv(summary_file, index=False)

print("Detailed output written to:", detail_file)
print("Summary output written to:", summary_file)