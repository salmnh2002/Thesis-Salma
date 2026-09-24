"""
Machine-learning framework for the thesis:
"A Machine Learning Framework for Identifying Climate-Adaptive Envelope
Retrofit Strategies to Improve Thermal Comfort and Reduce Overheating in
Free-Floating School Buildings"

Main objective
--------------
Predict degree-hours above 26 C during school-occupied hours and use the
best-performing regression model to rank simulated retrofit combinations:
    1. by climate zone;
    2. by climate zone and floor;
    3. by climate zone and floor among cases with no simulated overheating.

The workflow uses a climate-and-floor-stratified 70/15/15 split:
    1. 70% training data for fitting candidate models;
    2. 15% validation data for model selection;
    3. 15% hidden test data for one final unbiased evaluation.

It also diagnoses the largest untouched-test-set errors and checks whether
identical predictor combinations have conflicting target values.

Expected input file: "Master Data.xlsx"
The script can also read a .csv file if DATA_FILE is changed below.

Install the required packages once in Google Colab if necessary:
    !pip install -q pandas numpy matplotlib seaborn scikit-learn openpyxl \
        xgboost catboost joblib
"""

# %% SECTION 1 - IMPORT LIBRARIES

from pathlib import Path
import re
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "XGBoost is not installed. Run: pip install xgboost"
    ) from exc

try:
    from catboost import CatBoostRegressor
except ImportError as exc:
    raise ImportError(
        "CatBoost is not installed. Run: pip install catboost"
    ) from exc

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")


# %% SECTION 2 - USER SETTINGS

DATA_FILE = "Master Data.xlsx"
SHEET_NAME = 0
OUTPUT_DIR = Path("ML_Results")
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
N_LARGEST_ERRORS_TO_REPORT = 15
RESIDUAL_IQR_MULTIPLIER = 1.5
DUPLICATE_TARGET_TOLERANCE = 1e-6
# Degree-hours are expected to be non-negative. This small tolerance prevents
# floating-point noise extremely close to zero from being labelled overheating.
NO_OVERHEATING_TOLERANCE = 1e-9

TARGET = "degree_hours_above_26_degrees_school_occupied_hours"
CLIMATE_GROUP = "location"

# These are simulation outputs or metadata. They are retained for reporting,
# but they are NEVER included in X (the model predictors).
METADATA_COLUMNS = ["case_id"]
SUPPORTING_OUTPUTS = [
    "tmean_school_occupied_hours",
    "tmax_school_occupied_hours",
    "tmin_school_occupied_hours",
    "hours_above_26_degrees_school_occupied_hours",
    "hours_exceeding_the_category_i_upper_comfort_limit_with_ventilation",
    "hours_exceeding_the_category_ii_upper_comfort_limit_with_ventilation",
    "hours_exceeding_the_category_iii_upper_comfort_limit_with_ventilation",
    "overheating",
    "hours_exceeding_the_category_i_upper_comfort_limit_without_ventilation",
    "hours_exceeding_the_category_ii_upper_comfort_limit_without_ventilation",
    "hours_exceeding_the_category_iii_upper_comfort_limit_without_ventilation",
    "overheating_without_ventilation",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %% SECTION 3 - HELPER FUNCTIONS

def normalise_column_name(name):
    """Convert Excel headings to consistent snake_case column names."""
    name = str(name).strip().lower()
    name = name.replace("°", "").replace("%", "percent")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def find_input_file(requested_file):
    """Find the input locally or request an upload when running in Colab."""
    path = Path(requested_file)
    if path.exists():
        return path

    allowed_extensions = {".xlsx", ".xls", ".csv"}
    candidates = sorted(
        p for p in Path.cwd().iterdir()
        if p.suffix.lower() in allowed_extensions
        and not p.name.startswith("ML_Results")
    )
    if len(candidates) == 1:
        return candidates[0]

    try:
        from google.colab import files
        print(f"Upload {requested_file} (or your equivalent master-data file).")
        uploaded = files.upload()
        valid = [Path(name) for name in uploaded if Path(name).suffix.lower() in allowed_extensions]
        if not valid:
            raise FileNotFoundError("No Excel or CSV file was uploaded.")
        return valid[0]
    except ImportError as exc:
        if candidates:
            names = ", ".join(p.name for p in candidates)
            raise FileNotFoundError(
                f"Could not find '{requested_file}'. Multiple data files exist: {names}. "
                "Set DATA_FILE to the correct filename."
            ) from exc
        raise FileNotFoundError(
            f"Could not find '{requested_file}' in {Path.cwd()}."
        ) from exc


def read_dataset(path):
    """Read an Excel or CSV dataset."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=SHEET_NAME)


def convert_numeric(series):
    """Convert numbers saved as text, including decimal-comma values."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def normalise_overheating(series):
    """Convert common TRUE/FALSE or 1/0 representations to nullable Boolean."""
    mapping = {
        "true": True, "false": False,
        "yes": True, "no": False,
        "1": True, "0": False,
        "y": True, "n": False,
    }
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    return series.astype("string").str.strip().str.lower().map(mapping).astype("boolean")


def normalise_orientation(series):
    """Map orientation degrees to validated cardinal-direction categories."""
    degree_to_direction = {
        0.0: "South",
        90.0: "East",
        180.0: "North",
        270.0: "West",
    }
    text_to_direction = {
        "north": "North", "n": "North",
        "east": "East", "e": "East",
        "south": "South", "s": "South",
        "west": "West", "w": "West",
    }

    raw = series.astype("string").str.strip()
    numeric = convert_numeric(series)
    result = pd.Series(pd.NA, index=series.index, dtype="string")

    numeric_mask = numeric.notna()
    result.loc[numeric_mask] = numeric.loc[numeric_mask].map(degree_to_direction)

    text_mask = ~numeric_mask & raw.notna() & raw.ne("")
    result.loc[text_mask] = raw.loc[text_mask].str.lower().map(text_to_direction)

    invalid_mask = raw.notna() & raw.ne("") & result.isna()
    if invalid_mask.any():
        invalid_values = sorted(series.loc[invalid_mask].astype(str).unique())
        raise ValueError(
            "Unexpected building_orientation value(s): "
            + ", ".join(invalid_values)
            + ". Allowed values are 0/North, 90/East, 180/South, and 270/West."
        )

    return result


def regression_metrics(y_true, y_pred):
    """Return the three regression metrics used in the thesis."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_by_group(result_frame, group_columns):
    """Calculate test-set performance for each climate/floor group."""
    rows = []
    group_key = group_columns[0] if len(group_columns) == 1 else group_columns
    for group_values, group in result_frame.groupby(group_key, observed=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        metrics = regression_metrics(group["actual"], group["predicted"])
        row = dict(zip(group_columns, group_values))
        row.update({"n_test_cases": len(group), **metrics})
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_repeated_inputs(frame, feature_columns, target_column):
    """Find repeated X combinations and quantify disagreement in their Y values."""
    aggregation = {
        "n_cases": (target_column, "size"),
        "n_unique_targets": (target_column, lambda values: values.nunique(dropna=True)),
        "target_min": (target_column, "min"),
        "target_max": (target_column, "max"),
        "target_mean": (target_column, "mean"),
        "target_sd": (target_column, "std"),
    }
    if "case_id" in frame.columns:
        aggregation["case_ids"] = (
            "case_id",
            lambda values: ", ".join(values.astype("string").fillna("missing")),
        )

    summary = (
        frame.groupby(feature_columns, observed=True, dropna=False)
        .agg(**aggregation)
        .reset_index()
    )
    summary["target_range"] = summary["target_max"] - summary["target_min"]
    return (
        summary.loc[summary["n_cases"].gt(1)]
        .sort_values(["target_range", "n_cases"], ascending=[False, False])
        .reset_index(drop=True)
    )


def rank_best_cases(frame, group_columns, eligibility_mask=None):
    """Select the minimum predicted degree-hours within each requested group."""
    candidates = frame.copy()
    if eligibility_mask is not None:
        candidates = candidates.loc[eligibility_mask].copy()
    if candidates.empty:
        return candidates

    best_indices = candidates.groupby(group_columns, observed=True)[
        "predicted_degree_hours_above_26"
    ].idxmin()
    return (
        candidates.loc[best_indices]
        .sort_values(group_columns)
        .reset_index(drop=True)
    )


def save_figure(filename):
    """Apply consistent export settings to all thesis figures."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# %% SECTION 4 - LOAD AND CLEAN THE MASTER DATA

data_path = find_input_file(DATA_FILE)
df_raw = read_dataset(data_path)

df = df_raw.copy()
df.columns = [normalise_column_name(column) for column in df.columns]
df = df.loc[:, ~df.columns.str.startswith("unnamed")]

# Remove completely empty rows and exact duplicate rows.
df = df.dropna(how="all").drop_duplicates().reset_index(drop=True)

print(f"Input file: {data_path.name}")
print(f"Dataset shape after basic cleaning: {df.shape[0]} rows x {df.shape[1]} columns")
print("Columns found:")
print(df.columns.tolist())


# %% SECTION 5 - DEFINE X AND Y WITHOUT DATA LEAKAGE

# Categorical contextual inputs. Building orientation is categorical because
# the four degree values identify directions rather than a linear quantity.
CATEGORICAL_FEATURES = [
    "location",
    "floor",
    "building_orientation",
]

NUMERIC_FEATURES = [
    "ventilation_rate_l_s_person",
    "window_to_floor_ratio",
    "wall_thermal_transmittance",
    "roof_thermal_transmittance",
    "window_thermal_transmittance",
    "shgc",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

required_columns = FEATURES + [TARGET]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise ValueError(
        "The following required columns are missing from the dataset: "
        + ", ".join(missing_columns)
    )

# Convert physical parameters and the target to numeric.
for column in NUMERIC_FEATURES + [TARGET]:
    df[column] = convert_numeric(df[column])

df["building_orientation"] = normalise_orientation(df["building_orientation"])

for column in ["location", "floor"]:
    df[column] = df[column].astype("string").str.strip()

if "overheating" in df.columns:
    # Preserve the criterion reported by the source spreadsheet. It may use a
    # comfort-standard rule that is different from degree-hours above 26 C.
    df["overheating"] = normalise_overheating(df["overheating"])
    df["overheating_reported"] = df["overheating"].copy()
else:
    df["overheating"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df["overheating_reported"] = df["overheating"].copy()

if "overheating_without_ventilation" in df.columns:
    df["overheating_without_ventilation"] = normalise_overheating(
        df["overheating_without_ventilation"]
    )

for column in SUPPORTING_OUTPUTS:
    if column in df.columns and column not in {
        "overheating", "overheating_without_ventilation"
    }:
        df[column] = convert_numeric(df[column])

# The target cannot be imputed. Rows without a target are removed and reported.
rows_before_target_filter = len(df)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
print(f"Rows removed because the target was missing: {rows_before_target_filter - len(df)}")

# This is a separate, explicit definition based only on the regression target.
# It must not silently overwrite the source spreadsheet's overheating criterion.
df["overheating_from_degree_hours"] = (
    df[TARGET].gt(NO_OVERHEATING_TOLERANCE).astype("boolean")
)
reported_flag_available = df["overheating_reported"].notna()
flag_mismatch_mask = (
    reported_flag_available
    & df["overheating_reported"].ne(df["overheating_from_degree_hours"]).fillna(False)
)
df["overheating_flag_matches_degree_hours"] = pd.Series(
    pd.NA, index=df.index, dtype="boolean"
)
df.loc[
    reported_flag_available, "overheating_flag_matches_degree_hours"
] = ~flag_mismatch_mask.loc[reported_flag_available]

overheating_flag_check = pd.DataFrame({
    "check": [
        "total_cases_with_target",
        "cases_with_reported_flag",
        "reported_true",
        "reported_false",
        "derived_true_degree_hours_gt_tolerance",
        "derived_false_degree_hours_at_or_below_tolerance",
        "reported_vs_derived_mismatches",
        "unrecognised_or_missing_reported_flags",
    ],
    "count": [
        len(df),
        int(reported_flag_available.sum()),
        int(df["overheating_reported"].eq(True).sum()),
        int(df["overheating_reported"].eq(False).sum()),
        int(df["overheating_from_degree_hours"].eq(True).sum()),
        int(df["overheating_from_degree_hours"].eq(False).sum()),
        int(flag_mismatch_mask.sum()),
        int(df["overheating_reported"].isna().sum()),
    ],
})

flag_diagnostic_columns = [
    column for column in [
        "case_id", "location", "floor", TARGET,
        "hours_above_26_degrees_school_occupied_hours",
        "overheating_reported", "overheating_from_degree_hours",
        "overheating_flag_matches_degree_hours",
    ]
    if column in df.columns
]
overheating_flag_mismatches = (
    df.loc[flag_mismatch_mask, flag_diagnostic_columns]
    .sort_values([TARGET, "case_id"] if "case_id" in df.columns else [TARGET],
                 ascending=False)
    .reset_index(drop=True)
)

overheating_flag_check.to_csv(
    OUTPUT_DIR / "17_Overheating_Flag_Check.csv", index=False
)
overheating_flag_mismatches.to_csv(
    OUTPUT_DIR / "18_Overheating_Flag_Mismatches.csv", index=False
)

print("\nOverheating-flag validation:")
print(overheating_flag_check.to_string(index=False))
if flag_mismatch_mask.any():
    print(
        "WARNING: The reported overheating flag disagrees with degree-hours > "
        f"{NO_OVERHEATING_TOLERANCE:g} in {int(flag_mismatch_mask.sum())} case(s). "
        "See 18_Overheating_Flag_Mismatches.csv."
    )

# Reject an entirely empty predictor because median/mode imputation cannot repair it.
empty_features = [column for column in FEATURES if df[column].isna().all()]
if empty_features:
    raise ValueError("These predictors are completely empty: " + ", ".join(empty_features))

# Explicit leakage check.
forbidden_predictors = set([TARGET] + SUPPORTING_OUTPUTS + METADATA_COLUMNS)
leaked_columns = sorted(set(FEATURES).intersection(forbidden_predictors))
if leaked_columns:
    raise RuntimeError("Output leakage detected in X: " + ", ".join(leaked_columns))

X = df[FEATURES].copy()
y = df[TARGET].copy()

print("\nPredictors used in X:")
print(FEATURES)
print(f"\nRegression target Y: {TARGET}")
print("All purple metadata and beige supporting outputs are excluded from X.")
print("The nine blue parameter columns, including ventilation rate, are used in X.")
print("Orientation mapping: 0=North, 90=East, 180=South, 270=West.")


# %% SECTION 6 - DATA-QUALITY AND BALANCE CHECKS

quality_summary = pd.DataFrame({
    "dtype": df[FEATURES + [TARGET]].dtypes.astype(str),
    "missing_values": df[FEATURES + [TARGET]].isna().sum(),
    "unique_values": df[FEATURES + [TARGET]].nunique(dropna=True),
})
quality_summary.to_csv(OUTPUT_DIR / "01_Data_Quality_Summary.csv")

balance_climate = (
    df.groupby(CLIMATE_GROUP, observed=True)
    .size()
    .rename("number_of_cases")
    .reset_index()
)
balance_zone_floor = (
    df.groupby([CLIMATE_GROUP, "floor"], observed=True)
    .size()
    .rename("number_of_cases")
    .reset_index()
)

print("\nCases per climate zone:")
print(balance_climate.to_string(index=False))
print("\nCases per climate zone and floor:")
print(balance_zone_floor.to_string(index=False))


# %% SECTION 7 - THREE-WAY 70/15/15 SPLIT

if not np.isclose(TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE, 1.0):
    raise ValueError("TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE must equal 1.0.")

# Preserve the distribution of both climate and floor in all three subsets.
# These labels are used only for splitting; they are not added as predictors.
split_strata = (
    df[[CLIMATE_GROUP, "floor"]]
    .astype("string")
    .fillna("missing")
    .agg(" | ".join, axis=1)
)

small_strata = split_strata.value_counts().loc[lambda counts: counts.lt(3)]
if not small_strata.empty:
    raise ValueError(
        "Each climate-floor group needs at least three cases for a 70/15/15 "
        "stratified split. Too-small groups: " + ", ".join(small_strata.index)
    )

# First isolate the hidden test set. The remaining 85% is then divided so that
# validation represents exactly 15% of the complete dataset.
train_validation_index, test_index = train_test_split(
    df.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=split_strata,
)

validation_fraction_of_remainder = VALIDATION_SIZE / (TRAIN_SIZE + VALIDATION_SIZE)
train_index, validation_index = train_test_split(
    train_validation_index,
    test_size=validation_fraction_of_remainder,
    random_state=RANDOM_STATE,
    stratify=split_strata.loc[train_validation_index],
)

X_train = X.loc[train_index].reset_index(drop=True)
y_train = y.loc[train_index].reset_index(drop=True)
X_validation = X.loc[validation_index].reset_index(drop=True)
y_validation = y.loc[validation_index].reset_index(drop=True)
X_test = X.loc[test_index].reset_index(drop=True)
y_test = y.loc[test_index].reset_index(drop=True)

split_label = pd.Series(index=df.index, dtype="string")
split_label.loc[train_index] = "Training"
split_label.loc[validation_index] = "Validation"
split_label.loc[test_index] = "Test"

split_summary = pd.DataFrame({
    "Dataset_Split": ["Training", "Validation", "Test"],
    "Cases": [len(train_index), len(validation_index), len(test_index)],
    "Percentage": [
        100 * len(train_index) / len(df),
        100 * len(validation_index) / len(df),
        100 * len(test_index) / len(df),
    ],
})

split_assignment_columns = [
    column for column in ["case_id", CLIMATE_GROUP, "floor", TARGET]
    if column in df.columns
]
split_assignments = df[split_assignment_columns].copy()
split_assignments.insert(0, "Dataset_Split", split_label)
split_balance = (
    split_assignments
    .groupby(["Dataset_Split", CLIMATE_GROUP, "floor"], observed=True)
    .size()
    .rename("Cases")
    .reset_index()
)

print("\nThree-way dataset split:")
print(split_summary.round(2).to_string(index=False))
print("Split stratification: location + floor")

split_summary.to_csv(
    OUTPUT_DIR / "02_Data_Split_Summary.csv", index=False
)
split_assignments.to_csv(
    OUTPUT_DIR / "03_Data_Split_Assignments.csv", index=False
)
split_balance.to_csv(
    OUTPUT_DIR / "04_Data_Split_Balance.csv", index=False
)


# %% SECTION 8 - PREPROCESSING PIPELINE

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)


# %% SECTION 9 - DEFINE THE THREE REGRESSION MODELS

regressors = {
    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "XGBoost": XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "CatBoost": CatBoostRegressor(
        loss_function="RMSE",
        iterations=700,
        learning_rate=0.05,
        depth=7,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    ),
}

model_pipelines = {
    name: Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", regressor),
    ])
    for name, regressor in regressors.items()
}


# %% SECTION 10 - MODEL SELECTION ON THE 15% VALIDATION SET

# Each candidate is fitted only on the 70% training set. The validation set is
# used to compare the fixed candidate configurations and select the algorithm
# with the lowest validation RMSE. The hidden test set is not accessed here.
validation_metric_rows = []
validation_prediction_frames = []

validation_identity_columns = [
    column for column in ["case_id", CLIMATE_GROUP, "floor", TARGET]
    if column in df.columns
]

for model_name, pipeline in model_pipelines.items():
    print(f"\nFitting {model_name} on the 70% training set...")
    validation_model = clone(pipeline)
    validation_model.fit(X_train, y_train)
    validation_prediction = validation_model.predict(X_validation)
    metrics = regression_metrics(y_validation, validation_prediction)

    validation_metric_rows.append({
        "Model": model_name,
        "Training_Cases": len(X_train),
        "Validation_Cases": len(X_validation),
        "Validation_MAE": metrics["MAE"],
        "Validation_RMSE": metrics["RMSE"],
        "Validation_R2": metrics["R2"],
    })

    model_validation_results = (
        df.loc[validation_index, validation_identity_columns]
        .copy()
        .reset_index(drop=True)
    )
    model_validation_results.insert(0, "Model", model_name)
    model_validation_results["actual"] = y_validation.to_numpy()
    model_validation_results["predicted"] = validation_prediction
    model_validation_results["residual"] = (
        model_validation_results["actual"]
        - model_validation_results["predicted"]
    )
    model_validation_results["absolute_error"] = (
        model_validation_results["residual"].abs()
    )
    validation_prediction_frames.append(model_validation_results)

    print(
        f"  Validation RMSE={metrics['RMSE']:.3f}, "
        f"MAE={metrics['MAE']:.3f}, R2={metrics['R2']:.4f}"
    )

validation_comparison = (
    pd.DataFrame(validation_metric_rows)
    .sort_values("Validation_RMSE")
    .reset_index(drop=True)
)
validation_results = pd.concat(
    validation_prediction_frames, ignore_index=True
)

best_model_name = validation_comparison.loc[0, "Model"]
print("\nValidation-set model comparison:")
print(validation_comparison.round(4).to_string(index=False))
print(f"\nSelected model (lowest validation RMSE): {best_model_name}")

validation_comparison.to_csv(
    OUTPUT_DIR / "05_Validation_Model_Comparison.csv", index=False
)
validation_results.to_csv(
    OUTPUT_DIR / "06_Validation_Set_Predictions.csv", index=False
)


# %% SECTION 11 - REFIT ON 85% AND EVALUATE ON THE HIDDEN 15% TEST SET

# After model selection, combine training and validation. This lets the chosen
# algorithm learn from 85% of the cases while the test set remains untouched.
X_train_validation = pd.concat(
    [X_train, X_validation], ignore_index=True
)
y_train_validation = pd.concat(
    [y_train, y_validation], ignore_index=True
)

selected_test_model = clone(model_pipelines[best_model_name])
selected_test_model.fit(X_train_validation, y_train_validation)
y_test_pred = selected_test_model.predict(X_test)

test_metrics = regression_metrics(y_test, y_test_pred)
test_performance = pd.DataFrame([{
    "Selected_Model": best_model_name,
    "Selection_Criterion": "Lowest validation RMSE",
    "Training_Cases": len(X_train),
    "Validation_Cases": len(X_validation),
    "Refit_Training_Validation_Cases": len(X_train_validation),
    "Test_Cases": len(X_test),
    **test_metrics,
}])

print("\nUntouched test-set performance:")
print(test_performance.round(4).to_string(index=False))

test_results = df.loc[test_index].copy().reset_index(drop=True)
test_results["actual"] = y_test.to_numpy()
test_results["predicted"] = y_test_pred
test_results["residual"] = test_results["actual"] - test_results["predicted"]
test_results["absolute_error"] = test_results["residual"].abs()
test_results["error_direction"] = np.select(
    [test_results["residual"].gt(0), test_results["residual"].lt(0)],
    ["underprediction", "overprediction"],
    default="exact",
)

# Rank the largest independent test errors. A positive residual means that the
# actual result was higher than predicted, so the model underpredicted it.
largest_test_errors = (
    test_results.sort_values("absolute_error", ascending=False)
    .head(N_LARGEST_ERRORS_TO_REPORT)
    .copy()
)
largest_test_errors.insert(
    0, "error_rank", np.arange(1, len(largest_test_errors) + 1)
)

# Use the IQR rule to flag residual outliers without choosing an arbitrary
# degree-hour threshold.
absolute_error_q1 = test_results["absolute_error"].quantile(0.25)
absolute_error_q3 = test_results["absolute_error"].quantile(0.75)
absolute_error_iqr = absolute_error_q3 - absolute_error_q1
large_error_threshold = (
    absolute_error_q3 + RESIDUAL_IQR_MULTIPLIER * absolute_error_iqr
)
residual_outliers = (
    test_results.loc[test_results["absolute_error"].gt(large_error_threshold)]
    .sort_values("absolute_error", ascending=False)
    .reset_index(drop=True)
)

# If the same nine predictors occur more than once with different targets, the
# model is being asked to learn multiple Y values from identical X values. This
# indicates either an omitted varying input or inconsistent source results.
repeated_input_combinations = summarise_repeated_inputs(df, FEATURES, TARGET)
conflicting_input_combinations = repeated_input_combinations.loc[
    repeated_input_combinations["target_range"].gt(DUPLICATE_TARGET_TOLERANCE)
].reset_index(drop=True)

performance_by_climate = evaluate_by_group(test_results, [CLIMATE_GROUP])
performance_by_zone_floor = evaluate_by_group(
    test_results, [CLIMATE_GROUP, "floor"]
)

test_performance.to_csv(OUTPUT_DIR / "07_Selected_Model_Test_Performance.csv", index=False)
test_results.to_csv(OUTPUT_DIR / "08_Test_Set_Predictions.csv", index=False)
performance_by_climate.to_csv(OUTPUT_DIR / "09_Performance_By_Climate.csv", index=False)
performance_by_zone_floor.to_csv(
    OUTPUT_DIR / "10_Performance_By_Climate_And_Floor.csv", index=False
)
largest_test_errors.to_csv(
    OUTPUT_DIR / "11_Largest_Test_Errors.csv", index=False
)
residual_outliers.to_csv(
    OUTPUT_DIR / "12_IQR_Residual_Outliers.csv", index=False
)
repeated_input_combinations.to_csv(
    OUTPUT_DIR / "13_Repeated_Input_Combinations.csv", index=False
)
conflicting_input_combinations.to_csv(
    OUTPUT_DIR / "14_Conflicting_Input_Combinations.csv", index=False
)

diagnostic_columns = [
    column for column in [
        "error_rank", "case_id", CLIMATE_GROUP, "floor", "actual", "predicted",
        "residual", "absolute_error", "error_direction", *FEATURES,
    ]
    if column in largest_test_errors.columns
]
print(
    f"\nIQR large-error threshold: {large_error_threshold:.3f} degree-hours"
)
print(f"Residual outliers identified: {len(residual_outliers)}")
print(f"\nLargest {len(largest_test_errors)} untouched-test-set errors:")
print(largest_test_errors[diagnostic_columns].round(4).to_string(index=False))
print(
    "\nRepeated predictor combinations: "
    f"{len(repeated_input_combinations)}"
)
print(
    "Conflicting repeated combinations (same X, different Y): "
    f"{len(conflicting_input_combinations)}"
)
if not conflicting_input_combinations.empty:
    print(
        conflicting_input_combinations.head(20).round(4).to_string(index=False)
    )


# %% SECTION 12 - TEST-SET PLOTS

plt.figure(figsize=(7, 6))
sns.scatterplot(
    data=test_results,
    x="actual",
    y="predicted",
    hue=CLIMATE_GROUP,
    s=55,
    alpha=0.80,
    palette="tab10",
)
axis_min = min(test_results["actual"].min(), test_results["predicted"].min())
axis_max = max(test_results["actual"].max(), test_results["predicted"].max())
plt.plot([axis_min, axis_max], [axis_min, axis_max], "k--", linewidth=1.5, label="Ideal prediction")
plt.xlabel("Actual degree-hours above 26 C")
plt.ylabel("Predicted degree-hours above 26 C")
plt.title(f"Predicted vs actual values - {best_model_name}")
plt.legend(title="Climate zone", bbox_to_anchor=(1.02, 1), loc="upper left")
save_figure("08_Predicted_Vs_Actual.png")

plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=test_results,
    x="predicted",
    y="residual",
    hue=CLIMATE_GROUP,
    s=55,
    alpha=0.80,
    palette="tab10",
)
plt.axhline(0, color="black", linestyle="--", linewidth=1.5)

# Label the five largest errors so their Case IDs can be read directly from the
# residual graph as well as from the exported diagnostic tables.
if "case_id" in largest_test_errors.columns:
    for _, row in largest_test_errors.head(5).iterrows():
        plt.annotate(
            str(row["case_id"]),
            (row["predicted"], row["residual"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
plt.xlabel("Predicted degree-hours above 26 C")
plt.ylabel("Residual (actual - predicted)")
plt.title(f"Residual plot - {best_model_name}")
plt.legend(title="Climate zone", bbox_to_anchor=(1.02, 1), loc="upper left")
save_figure("09_Residual_Plot.png")

plt.figure(figsize=(8, 5))
sns.barplot(
    data=validation_comparison,
    x="Model",
    y="Validation_RMSE",
    color="#4C78A8",
)
plt.ylabel("Validation RMSE")
plt.xlabel("")
plt.title("Model comparison on the 15% validation set")
save_figure("10_Validation_Model_Comparison.png")


# %% SECTION 13 - INTERPRETABLE PERMUTATION FEATURE IMPORTANCE

# Because the permutation is performed on the complete pipeline, the result is
# reported using the original physical variables rather than separate one-hot
# categories. This makes the result easier to explain in the thesis.
permutation = permutation_importance(
    selected_test_model,
    X_test,
    y_test,
    scoring="neg_root_mean_squared_error",
    n_repeats=20,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

permutation_importance_table = (
    pd.DataFrame({
        "Feature": FEATURES,
        "Importance_Mean": permutation.importances_mean,
        "Importance_SD": permutation.importances_std,
    })
    .sort_values("Importance_Mean", ascending=False)
    .reset_index(drop=True)
)
permutation_importance_table.to_csv(
    OUTPUT_DIR / "15_Permutation_Feature_Importance.csv", index=False
)

plt.figure(figsize=(9, 6))
plot_data = permutation_importance_table.sort_values("Importance_Mean")
plt.barh(
    plot_data["Feature"],
    plot_data["Importance_Mean"],
    xerr=plot_data["Importance_SD"],
    color="#59A14F",
    alpha=0.90,
)
plt.xlabel("Increase in test RMSE after feature permutation")
plt.ylabel("")
plt.title(f"Permutation feature importance - {best_model_name}")
save_figure("12_Permutation_Feature_Importance.png")


# %% SECTION 14 - REFIT THE SELECTED MODEL ON ALL CASES

# The untouched test set above is used only for the final unbiased evaluation.
# After that evaluation, the selected algorithm is refitted on all available
# simulations to create the final ranking/deployment model.
final_model = clone(model_pipelines[best_model_name])
final_model.fit(X, y)

df_ranked = df.copy()
df_ranked["predicted_degree_hours_above_26"] = final_model.predict(X)
df_ranked["prediction_error"] = (
    df_ranked[TARGET] - df_ranked["predicted_degree_hours_above_26"]
)

joblib.dump(final_model, OUTPUT_DIR / "13_Final_Selected_Model.joblib")


# %% SECTION 15 - MODEL-NATIVE FEATURE IMPORTANCE

def native_feature_importance(fitted_pipeline, model_name):
    """Return importance for the one-hot encoded features of a tree model."""
    fitted_preprocessor = fitted_pipeline.named_steps["preprocess"]
    fitted_regressor = fitted_pipeline.named_steps["model"]
    if not hasattr(fitted_regressor, "feature_importances_"):
        return pd.DataFrame()
    transformed_names = fitted_preprocessor.get_feature_names_out()
    return (
        pd.DataFrame({
            "Model": model_name,
            "Transformed_Feature": transformed_names,
            "Importance": fitted_regressor.feature_importances_,
        })
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


selected_native_importance = native_feature_importance(final_model, best_model_name)
selected_native_importance.to_csv(
    OUTPUT_DIR / "14_Selected_Model_Native_Feature_Importance.csv", index=False
)

# Keep the requested XGBoost importance available even if another model wins.
if best_model_name == "XGBoost":
    xgboost_importance = selected_native_importance.copy()
else:
    xgboost_full_model = clone(model_pipelines["XGBoost"])
    xgboost_full_model.fit(X, y)
    xgboost_importance = native_feature_importance(xgboost_full_model, "XGBoost")

xgboost_importance.to_csv(
    OUTPUT_DIR / "15_XGBoost_Feature_Importance.csv", index=False
)

if not xgboost_importance.empty:
    plt.figure(figsize=(9, 7))
    top_xgb = xgboost_importance.head(20).sort_values("Importance")
    plt.barh(top_xgb["Transformed_Feature"], top_xgb["Importance"], color="#F28E2B")
    plt.xlabel("XGBoost feature importance")
    plt.ylabel("")
    plt.title("Top 20 XGBoost transformed-feature importances")
    save_figure("16_XGBoost_Feature_Importance.png")


# %% SECTION 16 - IDENTIFY THE BEST CLIMATE-ADAPTIVE RETROFIT CASES

best_by_climate = rank_best_cases(df_ranked, [CLIMATE_GROUP])
best_by_zone_floor = rank_best_cases(
    df_ranked, [CLIMATE_GROUP, "floor"]
)

# For this sheet, "no overheating" means zero occupied degree-hours above 26 C.
# The separately retained source flag may represent a different comfort criterion.
no_overheating_mask = (
    df_ranked["overheating_from_degree_hours"].eq(False).fillna(False)
)
best_no_overheating = rank_best_cases(
    df_ranked,
    [CLIMATE_GROUP, "floor"],
    eligibility_mask=no_overheating_mask,
)

# Use an easy-to-read column order in the thesis output tables.
report_first = [
    column for column in [
        "location", "floor", "case_id", "ventilation_rate_l_s_person",
        "window_to_floor_ratio", "building_orientation", "wall_thermal_transmittance",
        "roof_thermal_transmittance", "window_thermal_transmittance", "shgc",
        TARGET, "predicted_degree_hours_above_26", "overheating_reported",
        "overheating_from_degree_hours", "overheating_flag_matches_degree_hours",
        "tmean_school_occupied_hours", "tmax_school_occupied_hours",
        "tmin_school_occupied_hours",
        "hours_above_26_degrees_school_occupied_hours",
        "hours_exceeding_the_category_i_upper_comfort_limit_with_ventilation",
        "hours_exceeding_the_category_ii_upper_comfort_limit_with_ventilation",
        "hours_exceeding_the_category_iii_upper_comfort_limit_with_ventilation",
        "overheating_without_ventilation",
    ]
    if column in df_ranked.columns
]
remaining_columns = [column for column in df_ranked.columns if column not in report_first]
report_column_order = report_first + remaining_columns

best_by_climate = best_by_climate.reindex(columns=report_column_order)
best_by_zone_floor = best_by_zone_floor.reindex(columns=report_column_order)
best_no_overheating = best_no_overheating.reindex(columns=report_column_order)
df_ranked = df_ranked.reindex(columns=report_column_order)

print("\nBest predicted retrofit case by climate zone:")
print(best_by_climate[report_first].to_string(index=False))
print("\nBest predicted retrofit case by climate zone and floor:")
print(best_by_zone_floor[report_first].to_string(index=False))


# %% SECTION 17 - EXPORT ALL THESIS RESULTS TO ONE EXCEL WORKBOOK

excel_path = OUTPUT_DIR / "ML_Thesis_Results.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    quality_summary.reset_index(names="Column").to_excel(
        writer, sheet_name="Data_Quality", index=False
    )
    overheating_flag_check.to_excel(
        writer, sheet_name="Overheating_Flag_Check", index=False
    )
    overheating_flag_mismatches.to_excel(
        writer, sheet_name="Overheating_Mismatches", index=False
    )
    balance_climate.to_excel(writer, sheet_name="Balance_By_Climate", index=False)
    balance_zone_floor.to_excel(writer, sheet_name="Balance_Zone_Floor", index=False)
    split_summary.to_excel(writer, sheet_name="Split_Summary", index=False)
    split_assignments.to_excel(writer, sheet_name="Split_Assignments", index=False)
    split_balance.to_excel(writer, sheet_name="Split_Balance", index=False)
    validation_comparison.to_excel(
        writer, sheet_name="Validation_Comparison", index=False
    )
    validation_results.to_excel(
        writer, sheet_name="Validation_Predictions", index=False
    )
    test_performance.to_excel(writer, sheet_name="Test_Performance", index=False)
    test_results.to_excel(writer, sheet_name="Test_Predictions", index=False)
    performance_by_climate.to_excel(writer, sheet_name="Performance_Climate", index=False)
    performance_by_zone_floor.to_excel(writer, sheet_name="Performance_Zone_Floor", index=False)
    largest_test_errors.to_excel(writer, sheet_name="Largest_Test_Errors", index=False)
    residual_outliers.to_excel(writer, sheet_name="Residual_Outliers", index=False)
    repeated_input_combinations.to_excel(writer, sheet_name="Repeated_Inputs", index=False)
    conflicting_input_combinations.to_excel(
        writer, sheet_name="Conflicting_Inputs", index=False
    )
    permutation_importance_table.to_excel(
        writer, sheet_name="Feature_Importance", index=False
    )
    xgboost_importance.to_excel(writer, sheet_name="XGBoost_Importance", index=False)
    best_by_climate.to_excel(writer, sheet_name="Best_By_Climate_Zone", index=False)
    best_by_zone_floor.to_excel(writer, sheet_name="Best_By_Zone_Floor", index=False)
    best_no_overheating.to_excel(writer, sheet_name="Best_No_Overheating", index=False)
    df_ranked.to_excel(writer, sheet_name="All_Case_Predictions", index=False)

print("\nAnalysis complete.")
print(f"Selected model: {best_model_name}")
print(f"Results folder: {OUTPUT_DIR.resolve()}")
print(f"Main Excel output: {excel_path.resolve()}")
