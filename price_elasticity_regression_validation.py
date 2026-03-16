import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# ------------------------------------------------------------
# Output directory and files
# ------------------------------------------------------------
output_dir = Path(r"C:filepath")#Add filepath
output_dir.mkdir(parents=True, exist_ok=True)

union_output_file = output_dir / "regression_union_output.csv"
results_output_file = output_dir / "regression_results.csv"
validation_output_file = output_dir / "regression_validation.csv"

# ------------------------------------------------------------
# Sample Data
# ------------------------------------------------------------
train_data = [
    ["A", "1/1/2020", 94, 940],
    ["A", "2/1/2020", 95, 935],
    ["A", "3/1/2020", 96, 940],
    ["A", "4/1/2020", 98, 950],
    ["A", "5/1/2020", 100, 960],
    ["A", "6/1/2020", 102, 970],
    ["A", "7/1/2020", 104, 980],
    ["A", "8/1/2020", 99, 985],
    ["A", "9/1/2020", 96, 990],
    ["A", "10/1/2020", 93, 1000],
    ["A", "11/1/2020", 91, 1010],
    ["A", "12/1/2020", 90, 1025],
    ["A", "1/1/2021", 96, 930],
    ["A", "2/1/2021", 97, 925],
    ["A", "3/1/2021", 99, 930],
    ["A", "4/1/2021", 101, 940],
    ["A", "5/1/2021", 103, 950],
    ["A", "6/1/2021", 105, 960],
    ["A", "7/1/2021", 107, 970],
    ["A", "8/1/2021", 102, 975],
    ["A", "9/1/2021", 99, 980],
    ["A", "10/1/2021", 96, 990],
    ["A", "11/1/2021", 94, 1000],
    ["A", "12/1/2021", 92, 1015],
    ["A", "1/1/2022", 98, 920],
    ["A", "2/1/2022", 100, 910],
    ["A", "3/1/2022", 101, 915],
    ["A", "4/1/2022", 103, 925],
    ["A", "5/1/2022", 105, 935],
    ["A", "6/1/2022", 107, 945],
    ["A", "7/1/2022", 109, 955],
    ["A", "8/1/2022", 104, 960],
    ["A", "9/1/2022", 100, 965],
    ["A", "10/1/2022", 97, 975],
    ["A", "11/1/2022", 96, 985],
    ["A", "12/1/2022", 94, 1000],
    ["A", "1/1/2023", 100, 910],
    ["A", "2/1/2023", 102, 900],
    ["A", "3/1/2023", 103, 905],
    ["A", "4/1/2023", 105, 915],
    ["A", "5/1/2023", 108, 925],
    ["A", "6/1/2023", 110, 935],
    ["A", "7/1/2023", 112, 945],
    ["A", "8/1/2023", 106, 950],
    ["A", "9/1/2023", 101, 955],
    ["A", "10/1/2023", 98, 965],
    ["A", "11/1/2023", 97, 975],
    ["A", "12/1/2023", 95, 990],
    ["A", "1/1/2024", 107, 1000],
    ["A", "2/1/2024", 106, 950],
    ["A", "3/1/2024", 110, 970],
    ["A", "4/1/2024", 115, 960],
    ["A", "5/1/2024", 114, 940],
    ["A", "6/1/2024", 116, 980],
    ["A", "7/1/2024", 118, 1020],
    ["A", "8/1/2024", 112, 1050],
    ["A", "9/1/2024", 108, 1030],
    ["A", "10/1/2024", 104, 990],
    ["A", "11/1/2024", 102, 955],
    ["A", "12/1/2024", 101, 920]
]

train_df = pd.DataFrame(train_data, columns=["Product", "Period", "Price", "Volume"])

# ------------------------------------------------------------
# Sample Scoring Data to predict volume
# ------------------------------------------------------------
score_data = [
    ["A", "1/1/2025", 99.96, np.nan],
    ["A", "2/1/2025", 99.75, np.nan],
    ["A", "3/1/2025", 103.02, np.nan],
    ["A", "4/1/2025", 107.10, np.nan],
    ["A", "5/1/2025", 108.12, np.nan],
    ["A", "6/1/2025", 111.10, np.nan],
    ["A", "7/1/2025", 117.60, np.nan],
    ["A", "8/1/2025", 111.24, np.nan],
    ["A", "9/1/2025", 108.15, np.nan],
    ["A", "10/1/2025", 105.04, np.nan],
    ["A", "11/1/2025", 103.02, np.nan],
    ["A", "12/1/2025", 99.91, np.nan],
]

score_df = pd.DataFrame(score_data, columns=["Product", "Period", "Price", "Volume"])

# ------------------------------------------------------------
# Preparation Steps
# ------------------------------------------------------------
def prepare_features(df: pd.DataFrame, include_log_volume: bool = True) -> pd.DataFrame:
    df = df.copy()

    # Format Date field
    df["Date"] = pd.to_datetime(df["Period"], format="%m/%d/%Y")

    # Sort Date
    df = df.sort_values("Date").reset_index(drop=True)

    # Add Log_Price and Log_Volume
    df["Log_Price"] = np.log(df["Price"])

    if include_log_volume:
        df["Log_Volume"] = np.log(df["Volume"])
    else:
        df["Log_Volume"] = np.nan

    df["Month"] = df["Date"].dt.month.astype(str)
    df["Year"] = df["Date"].dt.year

    # Create month dummies dummy variables for seasonality
    for m in range(1, 13):
        df[str(m)] = np.where(df["Date"].dt.month == m, 1.0, 0.0)

    return df

# Prepare train and score data
train_prepped = prepare_features(train_df, include_log_volume=True)
score_prepped = prepare_features(score_df, include_log_volume=False)

# ------------------------------------------------------------
# Regression model
# Y Var = Log_Volume
# X Vars = Log_Price,1,2,3,4,5,6,7,8,9,10,11
# Month 12 is the reference month
# ------------------------------------------------------------
x_cols = ["Log_Price"] + [str(m) for m in range(1, 12)]

X_train = train_prepped[x_cols]
X_train = sm.add_constant(X_train)

y_train = train_prepped["Log_Volume"]

model = sm.OLS(y_train, X_train).fit()

# ------------------------------------------------------------
# Validation model (time-based holdout)
# Train on 2020-2023, test on 2024
# ------------------------------------------------------------
train_set = train_prepped[train_prepped["Year"] <= 2023].copy()
test_set = train_prepped[train_prepped["Year"] == 2024].copy()

X_train_holdout = sm.add_constant(train_set[x_cols], has_constant="add")
y_train_holdout = train_set["Log_Volume"]

X_test_holdout = sm.add_constant(test_set[x_cols], has_constant="add")
y_test_holdout = test_set["Log_Volume"]

validation_model = sm.OLS(y_train_holdout, X_train_holdout).fit()

train_pred_holdout = validation_model.predict(X_train_holdout)
test_pred_holdout = validation_model.predict(X_test_holdout)

train_rmse = np.sqrt(np.mean((y_train_holdout - train_pred_holdout) ** 2))
test_rmse = np.sqrt(np.mean((y_test_holdout - test_pred_holdout) ** 2))
rmse_ratio = test_rmse / train_rmse if train_rmse != 0 else np.nan

validation_results = pd.DataFrame({
    "Metric": [
        "Training RMSE",
        "Test RMSE",
        "RMSE Ratio",
        "Training Observations",
        "Test Observations"
    ],
    "Value": [
        train_rmse,
        test_rmse,
        rmse_ratio,
        len(train_set),
        len(test_set)
    ]
})

# ------------------------------------------------------------
# Score the 2025 scenario
# Convert the Log_Volume backt to Volume
# ------------------------------------------------------------
X_score = score_prepped[x_cols]
X_score = sm.add_constant(X_score, has_constant="add")

score_prepped["Volume__Predict_"] = model.predict(X_score)
score_prepped["Volume"] = np.exp(score_prepped["Volume__Predict_"])

# ------------------------------------------------------------
# Clean up Sample Data
# ------------------------------------------------------------
actual_output = train_prepped[["Product", "Date", "Price", "Volume"]].copy()
actual_output["Act"] = "Actual"
actual_output["Revenue"] = actual_output["Price"] * actual_output["Volume"]

# ------------------------------------------------------------
# Clean up Score Data
# ------------------------------------------------------------
estimate_output = score_prepped[["Product", "Date", "Price", "Volume"]].copy()
estimate_output["Act"] = "Estimate"
estimate_output["Revenue"] = estimate_output["Price"] * estimate_output["Volume"]

# ------------------------------------------------------------
# Union Sample and Score Data
# ------------------------------------------------------------
union_output = pd.concat([actual_output, estimate_output], ignore_index=True)

union_output = union_output.sort_values(["Date", "Act"]).reset_index(drop=True)

# ------------------------------------------------------------
# Regression results output
# Build a coefficient table plus model metrics
# ------------------------------------------------------------
coef_table = pd.DataFrame({
    "Term": model.params.index,
    "Coefficient": model.params.values,
    "Std_Error": model.bse.values,
    "t_Statistic": model.tvalues.values,
    "P_Value": model.pvalues.values,
})

model_stats = pd.DataFrame({
    "Metric": [
        "R-squared",
        "Adj. R-squared",
        "No. Observations",
        "AIC",
        "BIC",
        "RMSE"
    ],
    "Value": [
        model.rsquared,
        model.rsquared_adj,
        int(model.nobs),
        model.aic,
        model.bic,
        np.sqrt(model.mse_resid)
    ]
})

# Put both sections into CSV table
blank_row = pd.DataFrame([{
    "Section": "",
    "Metric": "",
    "Value": "",
    "Term": "",
    "Coefficient": "",
    "Std_Error": "",
    "t_Statistic": "",
    "P_Value": ""
}])

model_stats_export = model_stats.copy()
model_stats_export["Section"] = "Model Statistics"
model_stats_export["Term"] = ""
model_stats_export["Coefficient"] = ""
model_stats_export["Std_Error"] = ""
model_stats_export["t_Statistic"] = ""
model_stats_export["P_Value"] = ""

coef_export = coef_table.copy()
coef_export["Section"] = "Coefficients"
coef_export["Metric"] = ""
coef_export["Value"] = ""

regression_results = pd.concat(
    [
        model_stats_export[
            ["Section", "Metric", "Value", "Term", "Coefficient", "Std_Error", "t_Statistic", "P_Value"]
        ],
        blank_row,
        coef_export[
            ["Section", "Metric", "Value", "Term", "Coefficient", "Std_Error", "t_Statistic", "P_Value"]
        ],
    ],
    ignore_index=True
)

# ------------------------------------------------------------
# Write outputs
# ------------------------------------------------------------
union_output.to_csv(union_output_file, index=False)
regression_results.to_csv(results_output_file, index=False)
validation_results.to_csv(validation_output_file, index=False)

# ------------------------------------------------------------
# Console output
# ------------------------------------------------------------
print("Union output written to:", union_output_file)
print("Regression results written to:", results_output_file)
print("Validation results written to:", validation_output_file)

print("\n=== Union Output Preview ===")
print(union_output.head())

print("\n=== Regression Coefficients ===")
print(coef_table)

print("\n=== Model Statistics ===")
print(model_stats)

print("\n=== Validation Results ===")
print(validation_results)