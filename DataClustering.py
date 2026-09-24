# -*- coding: utf-8 -*-

"""
THESIS - SECTION 4.2.1
Distribution of Degree-Hours

PURPOSE
-------
Analyze the existing final aggregated modelling dataset using the existing
school-occupied cumulative degree-hour output.

The script DOES NOT:
- open hourly IESVE temperature files
- recalculate degree-hours
- recalculate Tmin, Tmax, or Tmean
- change occupancy definitions
- average cases by climate, floor, orientation, ventilation, etc.
- remove zero values
- remove extreme finite values
- overwrite the original Excel workbook

Expected dataset:
- 46,080 rows
- one row per simulation case

Required packages:
pip install pandas numpy matplotlib openpyxl
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. USER SETTINGS
# ============================================================

INPUT_FILE = Path(
    r"E:\Thesis\Thesis Data\Master Data.xlsx"
)

# Leave as None.
# The script will search the workbook for the correct worksheet.
SHEET_NAME = None

# Expected total number of simulation cases.
EXPECTED_ROWS = 46080

# Histogram bins requested for thesis figure.
N_BINS = 45

# Optional supplementary log1p diagnostic.
# The original-scale histogram always remains the main thesis figure.
CREATE_LOG1P_DIAGNOSTIC = True

# Folder created next to your original Excel file.
OUTPUT_FOLDER_NAME = "Section_4_2_1_Degree_Hours_Distribution"


# ============================================================
# 2. TARGET DEFINITION
# ============================================================

# This is the LOGICAL target name.
# It is NOT required to match Excel spacing exactly.
#
# Your Excel column:
# Degree Hours   Above 26 degrees school occupied hours

TARGET_LOGICAL_NAME = (
    "degree hours above 26 degrees school occupied hours"
)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def normalize_column_name(text):
    """
    Normalize a column name ONLY for matching.

    This handles:
    - uppercase/lowercase differences
    - multiple spaces
    - hidden non-breaking spaces
    - underscores
    - hyphens
    - punctuation

    IMPORTANT:
    The original Excel column name is preserved after matching.
    """

    text = str(text)

    # Replace hidden/non-breaking spaces.
    text = text.replace("\xa0", " ")

    # Lowercase.
    text = text.lower()

    # Normalize separators.
    text = text.replace("_", " ")
    text = text.replace("-", " ")

    # Remove punctuation.
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def find_case_id_column(columns):
    """
    Find the simulation identifier column.

    Your dataset includes 'Case ID'.
    This function also tolerates minor formatting differences.
    """

    accepted = {
        "case id",
        "caseid",
        "simulation id",
        "simulation case id",
        "sim id"
    }

    matches = []

    for col in columns:

        normalized = normalize_column_name(col)

        if normalized in accepted:
            matches.append(col)

    return matches


# ============================================================
# 4. CHECK INPUT FILE
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        "\nINPUT FILE NOT FOUND\n\n"
        "Python searched for:\n"
        + str(INPUT_FILE)
        + "\n\n"
        "Change INPUT_FILE at the top of this script "
        "to the location of your final aggregated dataset."
    )


print("\n" + "=" * 80)
print("SECTION 4.2.1 - DISTRIBUTION OF DEGREE-HOURS")
print("=" * 80)

print("\nInput file:")
print(INPUT_FILE.resolve())


# ============================================================
# 5. CREATE OUTPUT FOLDER
# ============================================================

output_dir = (
    INPUT_FILE.parent
    / OUTPUT_FOLDER_NAME
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

print("\nOutput folder:")
print(output_dir.resolve())


# ============================================================
# 6. INSPECT WORKBOOK
# ============================================================

xls = pd.ExcelFile(INPUT_FILE)

print("\nWorksheets found:")

for sheet in xls.sheet_names:
    print("  - " + str(sheet))


if SHEET_NAME is not None:

    if SHEET_NAME not in xls.sheet_names:

        raise RuntimeError(
            "\nRequested worksheet not found:\n"
            + str(SHEET_NAME)
        )

    sheets_to_check = [SHEET_NAME]

else:

    sheets_to_check = xls.sheet_names


# ============================================================
# 7. FIND THE EXACT TARGET COLUMN
# ============================================================

candidate_pairs = []


for sheet in sheets_to_check:

    header_df = pd.read_excel(
        INPUT_FILE,
        sheet_name=sheet,
        nrows=0
    )

    for col in header_df.columns:

        normalized = normalize_column_name(col)

        if normalized == TARGET_LOGICAL_NAME:

            candidate_pairs.append(
                (
                    sheet,
                    col
                )
            )


# ============================================================
# 8. HANDLE TARGET IDENTIFICATION
# ============================================================

if len(candidate_pairs) == 0:

    print("\n" + "!" * 80)
    print("TARGET COLUMN NOT FOUND")
    print("!" * 80)

    print(
        "\nThe exact logical target searched was:"
    )

    print(
        repr(TARGET_LOGICAL_NAME)
    )

    print(
        "\nRelevant columns found in the workbook:"
    )

    relevant_found = False

    for sheet in sheets_to_check:

        header_df = pd.read_excel(
            INPUT_FILE,
            sheet_name=sheet,
            nrows=0
        )

        for col in header_df.columns:

            normalized = normalize_column_name(col)

            if (
                "degree" in normalized
                and "school" in normalized
            ):

                relevant_found = True

                print(
                    "\nWorksheet: "
                    + repr(sheet)
                )

                print(
                    "Original column:"
                )

                print(
                    repr(col)
                )

                print(
                    "Normalized:"
                )

                print(
                    repr(normalized)
                )

    if not relevant_found:

        print(
            "\nNo degree-hour school columns were detected."
        )

    raise RuntimeError(
        "\nSTOPPED WITHOUT GUESSING.\n"
        "The school-occupied degree-hour target could not "
        "be identified uniquely."
    )


if len(candidate_pairs) > 1:

    print("\n" + "!" * 80)
    print("MULTIPLE MATCHING TARGETS FOUND")
    print("!" * 80)

    for sheet, col in candidate_pairs:

        print(
            "\nWorksheet: "
            + repr(sheet)
        )

        print(
            "Column: "
            + repr(col)
        )

    raise RuntimeError(
        "\nSTOPPED WITHOUT GUESSING.\n"
        "The same target exists in multiple worksheets."
    )


# Exactly one target found.
selected_sheet, target_column = candidate_pairs[0]


print("\n" + "=" * 80)
print("TARGET COLUMN CONFIRMED")
print("=" * 80)

print(
    "Exact worksheet:"
)

print(
    repr(selected_sheet)
)

print(
    "\nExact original Excel target column:"
)

print(
    repr(target_column)
)


# ============================================================
# 9. LOAD FINAL AGGREGATED DATASET
# ============================================================

df = pd.read_excel(
    INPUT_FILE,
    sheet_name=selected_sheet
)


total_rows = len(df)
total_columns = len(df.columns)


print("\n" + "=" * 80)
print("DATASET DIMENSIONS")
print("=" * 80)

print(
    "Rows    : {:,}".format(
        total_rows
    )
)

print(
    "Columns : {:,}".format(
        total_columns
    )
)


# ============================================================
# 10. TARGET DATA TYPE CHECK
# ============================================================

target_raw = df[target_column]


# Convert only an ANALYTICAL COPY.
# Original dataframe column remains unchanged.

target_numeric = pd.to_numeric(
    target_raw,
    errors="coerce"
)


# Count values that existed but could not be converted.

non_missing_raw_mask = (
    target_raw.notna()
)

non_numeric_mask = (
    non_missing_raw_mask
    & target_numeric.isna()
)

non_numeric_count = int(
    non_numeric_mask.sum()
)


# Missing after conversion.

missing_count = int(
    target_numeric.isna().sum()
)


# ============================================================
# 11. INFINITY CHECKS
# ============================================================

target_array = target_numeric.to_numpy(
    dtype=float,
    na_value=np.nan
)


positive_infinity_count = int(
    np.isposinf(
        target_array
    ).sum()
)


negative_infinity_count = int(
    np.isneginf(
        target_array
    ).sum()
)


total_infinity_count = (
    positive_infinity_count
    + negative_infinity_count
)


# ============================================================
# 12. FINITE VALID TARGET VALUES
# ============================================================

finite_mask = (
    target_numeric.notna()
    & np.isfinite(
        target_numeric
    )
)


valid_target = (
    target_numeric
    .loc[finite_mask]
    .astype(float)
)


valid_count = len(
    valid_target
)


negative_degree_hour_count = int(
    (
        valid_target < 0
    ).sum()
)


zero_degree_hour_count = int(
    (
        valid_target == 0
    ).sum()
)


# ============================================================
# 13. DUPLICATE FULL-ROW CHECK
# ============================================================

duplicate_rows_involved = int(
    df.duplicated(
        keep=False
    ).sum()
)


duplicate_rows_excess = int(
    df.duplicated(
        keep="first"
    ).sum()
)


# ============================================================
# 14. CASE ID CHECK
# ============================================================

case_id_candidates = find_case_id_column(
    df.columns
)


case_id_column = None
unique_case_ids = None
duplicate_case_id_rows = None
duplicate_case_id_excess = None
missing_case_ids = None


if len(case_id_candidates) == 1:

    case_id_column = (
        case_id_candidates[0]
    )

    missing_case_ids = int(
        df[
            case_id_column
        ]
        .isna()
        .sum()
    )

    unique_case_ids = int(
        df[
            case_id_column
        ]
        .nunique(
            dropna=True
        )
    )

    duplicate_case_id_rows = int(
        df[
            case_id_column
        ]
        .duplicated(
            keep=False
        )
        .sum()
    )

    duplicate_case_id_excess = int(
        df[
            case_id_column
        ]
        .duplicated(
            keep="first"
        )
        .sum()
    )


elif len(case_id_candidates) > 1:

    print(
        "\nMultiple possible simulation ID columns detected:"
    )

    for col in case_id_candidates:
        print(
            "  - "
            + repr(col)
        )


# ============================================================
# 15. CONFIRM ONE ROW PER SIMULATION CASE
# ============================================================

if case_id_column is not None:

    one_row_per_case = (
        missing_case_ids == 0
        and duplicate_case_id_excess == 0
        and unique_case_ids == total_rows
    )

    if one_row_per_case:

        one_row_per_case_text = (
            "YES - confirmed using "
            + repr(case_id_column)
        )

    else:

        one_row_per_case_text = (
            "NO / ISSUE DETECTED using "
            + repr(case_id_column)
        )

else:

    one_row_per_case = None

    one_row_per_case_text = (
        "NOT CONFIRMABLE - no unique case ID column identified"
    )


expected_rows_match = (
    total_rows == EXPECTED_ROWS
)


# ============================================================
# 16. PRINT COMPLETE DATA QUALITY CHECKS
# ============================================================

print("\n" + "=" * 80)
print("DATA QUALITY CHECKS")
print("=" * 80)


print(
    "1. Total dataset rows:"
)

print(
    "   {:,}".format(
        total_rows
    )
)


print(
    "\n2. One row per simulation case:"
)

print(
    "   "
    + one_row_per_case_text
)


print(
    "\n3. Expected number of rows:"
)

print(
    "   Expected: {:,}".format(
        EXPECTED_ROWS
    )
)

print(
    "   Actual  : {:,}".format(
        total_rows
    )
)

print(
    "   Match   : "
    + (
        "YES"
        if expected_rows_match
        else "NO"
    )
)


print(
    "\n4. Exact target column:"
)

print(
    "   "
    + repr(target_column)
)


print(
    "\n5. Target numeric:"
)

print(
    "   "
    + (
        "YES"
        if non_numeric_count == 0
        else "NO"
    )
)

print(
    "   Non-numeric non-missing values: {:,}"
    .format(
        non_numeric_count
    )
)


print(
    "\n6. Missing target values:"
)

print(
    "   {:,}".format(
        missing_count
    )
)


print(
    "\n7. Infinite target values:"
)

print(
    "   Positive infinity: {:,}"
    .format(
        positive_infinity_count
    )
)

print(
    "   Negative infinity: {:,}"
    .format(
        negative_infinity_count
    )
)


print(
    "\n8. Negative degree-hour values:"
)

print(
    "   {:,}".format(
        negative_degree_hour_count
    )
)


print(
    "\n9. Exactly zero degree-hour values:"
)

print(
    "   {:,}".format(
        zero_degree_hour_count
    )
)


print(
    "\n10. Duplicate checks:"
)

print(
    "   Rows involved in full duplicates: {:,}"
    .format(
        duplicate_rows_involved
    )
)

print(
    "   Excess full duplicate rows: {:,}"
    .format(
        duplicate_rows_excess
    )
)


if case_id_column is not None:

    print(
        "   Case ID column: "
        + repr(case_id_column)
    )

    print(
        "   Missing Case IDs: {:,}"
        .format(
            missing_case_ids
        )
    )

    print(
        "   Unique Case IDs: {:,}"
        .format(
            unique_case_ids
        )
    )

    print(
        "   Rows with duplicated Case IDs: {:,}"
        .format(
            duplicate_case_id_rows
        )
    )

    print(
        "   Excess duplicated Case IDs: {:,}"
        .format(
            duplicate_case_id_excess
        )
    )


# ============================================================
# 17. QUALITY WARNINGS
# ============================================================

quality_warnings = []


if not expected_rows_match:

    quality_warnings.append(
        "Dataset contains {:,} rows instead of {:,}."
        .format(
            total_rows,
            EXPECTED_ROWS
        )
    )


if non_numeric_count > 0:

    quality_warnings.append(
        "{:,} existing target values are non-numeric."
        .format(
            non_numeric_count
        )
    )


if total_infinity_count > 0:

    quality_warnings.append(
        "{:,} infinite target values detected."
        .format(
            total_infinity_count
        )
    )


if negative_degree_hour_count > 0:

    quality_warnings.append(
        "{:,} negative degree-hour values detected."
        .format(
            negative_degree_hour_count
        )
    )


if duplicate_rows_excess > 0:

    quality_warnings.append(
        "{:,} excess fully duplicated rows detected."
        .format(
            duplicate_rows_excess
        )
    )


if (
    case_id_column is not None
    and duplicate_case_id_excess > 0
):

    quality_warnings.append(
        "{:,} excess duplicated Case IDs detected."
        .format(
            duplicate_case_id_excess
        )
    )


if quality_warnings:

    print("\n" + "!" * 80)
    print("DATA QUALITY WARNINGS")
    print("!" * 80)

    for warning in quality_warnings:

        print(
            "- "
            + warning
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No observations have been deleted, corrected, capped, "
        "winsorised, or replaced."
    )


# ============================================================
# 18. CHECK VALID TARGET COUNT
# ============================================================

if valid_count == 0:

    raise RuntimeError(
        "\nNo finite numeric target values are available "
        "for the histogram."
    )


# ============================================================
# 19. DESCRIPTIVE STATISTICS
# ============================================================

minimum = float(
    valid_target.min()
)


percentile_25 = float(
    valid_target.quantile(
        0.25
    )
)


median = float(
    valid_target.median()
)


mean = float(
    valid_target.mean()
)


percentile_75 = float(
    valid_target.quantile(
        0.75
    )
)


maximum = float(
    valid_target.max()
)


# SAMPLE standard deviation.
# ddof=1 is explicitly used.

standard_deviation_sample = float(
    valid_target.std(
        ddof=1
    )
)


skewness = float(
    valid_target.skew()
)


zero_percentage = (
    zero_degree_hour_count
    / valid_count
    * 100.0
)


# ============================================================
# 20. CREATE DESCRIPTIVE STATISTICS TABLE
# ============================================================

statistics_data = [

    [
        "Total dataset rows",
        total_rows,
        "Count"
    ],

    [
        "Valid target values",
        valid_count,
        "Count"
    ],

    [
        "Missing target values",
        missing_count,
        "Count"
    ],

    [
        "Minimum",
        minimum,
        "degC h"
    ],

    [
        "25th percentile",
        percentile_25,
        "degC h"
    ],

    [
        "Median",
        median,
        "degC h"
    ],

    [
        "Mean",
        mean,
        "degC h"
    ],

    [
        "75th percentile",
        percentile_75,
        "degC h"
    ],

    [
        "Maximum",
        maximum,
        "degC h"
    ],

    [
        "Standard deviation (sample, ddof=1)",
        standard_deviation_sample,
        "degC h"
    ],

    [
        "Skewness",
        skewness,
        "Dimensionless"
    ],

    [
        "Zero-degree-hour cases",
        zero_degree_hour_count,
        "Count"
    ],

    [
        "Zero-degree-hour cases",
        zero_percentage,
        "%"
    ]
]


stats_df = pd.DataFrame(
    statistics_data,
    columns=[
        "Statistic",
        "Value",
        "Unit"
    ]
)


# ============================================================
# 21. PRINT DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)


print(
    "Total dataset rows             : {:,}"
    .format(
        total_rows
    )
)

print(
    "Valid target values            : {:,}"
    .format(
        valid_count
    )
)

print(
    "Missing target values          : {:,}"
    .format(
        missing_count
    )
)

print(
    "Minimum                        : {:,.2f} degC h"
    .format(
        minimum
    )
)

print(
    "25th percentile                : {:,.2f} degC h"
    .format(
        percentile_25
    )
)

print(
    "Median                         : {:,.2f} degC h"
    .format(
        median
    )
)

print(
    "Mean                           : {:,.2f} degC h"
    .format(
        mean
    )
)

print(
    "75th percentile                : {:,.2f} degC h"
    .format(
        percentile_75
    )
)

print(
    "Maximum                        : {:,.2f} degC h"
    .format(
        maximum
    )
)

print(
    "Sample standard deviation      : {:,.2f} degC h"
    .format(
        standard_deviation_sample
    )
)

print(
    "Skewness                       : {:.3f}"
    .format(
        skewness
    )
)

print(
    "Zero-degree-hour cases         : {:,}"
    .format(
        zero_degree_hour_count
    )
)

print(
    "Zero-degree-hour cases (%)     : {:.2f}%"
    .format(
        zero_percentage
    )
)


# ============================================================
# 22. CREATE DATA QUALITY TABLE
# ============================================================

quality_data = [

    [
        "Input file",
        str(
            INPUT_FILE.resolve()
        )
    ],

    [
        "Worksheet used",
        selected_sheet
    ],

    [
        "Target column",
        target_column
    ],

    [
        "Total dataset rows",
        total_rows
    ],

    [
        "Expected dataset rows",
        EXPECTED_ROWS
    ],

    [
        "Expected row count matched",
        expected_rows_match
    ],

    [
        "One row per simulation case",
        one_row_per_case_text
    ],

    [
        "Non-numeric non-missing target values",
        non_numeric_count
    ],

    [
        "Missing target values",
        missing_count
    ],

    [
        "Positive infinite values",
        positive_infinity_count
    ],

    [
        "Negative infinite values",
        negative_infinity_count
    ],

    [
        "Negative degree-hour values",
        negative_degree_hour_count
    ],

    [
        "Exactly zero degree-hour values",
        zero_degree_hour_count
    ],

    [
        "Rows involved in full duplicates",
        duplicate_rows_involved
    ],

    [
        "Excess full duplicate rows",
        duplicate_rows_excess
    ],

    [
        "Case ID column",
        (
            case_id_column
            if case_id_column is not None
            else "Not uniquely identified"
        )
    ],

    [
        "Unique Case IDs",
        (
            unique_case_ids
            if unique_case_ids is not None
            else "NA"
        )
    ],

    [
        "Rows with duplicated Case IDs",
        (
            duplicate_case_id_rows
            if duplicate_case_id_rows is not None
            else "NA"
        )
    ],

    [
        "Excess duplicated Case IDs",
        (
            duplicate_case_id_excess
            if duplicate_case_id_excess is not None
            else "NA"
        )
    ],

    [
        "Valid finite target values used",
        valid_count
    ]
]


quality_df = pd.DataFrame(
    quality_data,
    columns=[
        "Check",
        "Result"
    ]
)


# ============================================================
# 23. EXPORT STATISTICS TABLE
# ============================================================

statistics_excel_path = (
    output_dir
    / "Section_4_2_1_Descriptive_Statistics.xlsx"
)


statistics_csv_path = (
    output_dir
    / "Section_4_2_1_Descriptive_Statistics.csv"
)


with pd.ExcelWriter(
    statistics_excel_path,
    engine="openpyxl"
) as writer:

    stats_df.to_excel(
        writer,
        sheet_name="Descriptive Statistics",
        index=False
    )

    quality_df.to_excel(
        writer,
        sheet_name="Data Quality Checks",
        index=False
    )


stats_df.to_csv(
    statistics_csv_path,
    index=False
)


# ============================================================
# 24. HISTOGRAM SETTINGS
# ============================================================

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10
    }
)


# ============================================================
# 25. RELATIVE FREQUENCY WEIGHTS
# ============================================================

# Every valid simulation case contributes the same amount:
#
# 100 / number of valid cases
#
# Therefore all histogram bins together equal approximately 100%.

weights = np.ones(
    valid_count,
    dtype=float
) * (
    100.0
    / valid_count
)


# ============================================================
# 26. CREATE PRINCIPAL THESIS HISTOGRAM
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9.0,
        5.8
    )
)


hist_percentages, bin_edges, patches = ax.hist(

    valid_target.to_numpy(),

    bins=N_BINS,

    weights=weights,

    edgecolor="black",

    linewidth=0.5,

    alpha=0.80
)


# ============================================================
# 27. ADD MEAN LINE
# ============================================================

ax.axvline(

    mean,

    linestyle="--",

    linewidth=1.8,

    label=(
        "Mean = {:,.2f} degC h"
        .format(
            mean
        )
    )
)


# ============================================================
# 28. ADD MEDIAN LINE
# ============================================================

ax.axvline(

    median,

    linestyle="-.",

    linewidth=1.8,

    label=(
        "Median = {:,.2f} degC h"
        .format(
            median
        )
    )
)


# ============================================================
# 29. FIGURE LABELS
# ============================================================

ax.set_title(
    "Distribution of School-Occupied Degree-Hours Above 26 degC"
)


ax.set_xlabel(
    "School-occupied degree-hours above 26 degC (degC h)"
)


ax.set_ylabel(
    "Simulation cases (%)"
)


# ============================================================
# 30. GRID AND LEGEND
# ============================================================

ax.grid(
    axis="y",
    alpha=0.25,
    linewidth=0.8
)


ax.set_axisbelow(
    True
)


ax.legend(
    frameon=False
)


# ============================================================
# 31. PRESERVE ORIGINAL SCALE
# ============================================================

# Keep zero values.
# Keep maximum values.
# Do NOT cap or remove extreme values.

ax.set_xlim(
    left=min(
        0.0,
        minimum
    )
)


# ============================================================
# 32. ADD VALID SAMPLE SIZE
# ============================================================

fig.text(

    0.99,

    0.01,

    "Valid simulation cases: N = {:,}"
    .format(
        valid_count
    ),

    ha="right",

    va="bottom",

    fontsize=9
)


fig.tight_layout(
    rect=[
        0,
        0.035,
        1,
        1
    ]
)


# ============================================================
# 33. OUTPUT FIGURE PATHS
# ============================================================

png_path = (
    output_dir
    / "Figure_4_2_1_Degree_Hours_Distribution_Original_Scale.png"
)


pdf_path = (
    output_dir
    / "Figure_4_2_1_Degree_Hours_Distribution_Original_Scale.pdf"
)


svg_path = (
    output_dir
    / "Figure_4_2_1_Degree_Hours_Distribution_Original_Scale.svg"
)


# ============================================================
# 34. SAVE PRINCIPAL FIGURE
# ============================================================

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)


fig.savefig(
    pdf_path,
    bbox_inches="tight"
)


fig.savefig(
    svg_path,
    bbox_inches="tight"
)


# ============================================================
# 35. CHECK HISTOGRAM PERCENTAGES
# ============================================================

histogram_percentage_sum = float(
    hist_percentages.sum()
)


print("\n" + "=" * 80)
print("HISTOGRAM VALIDATION")
print("=" * 80)


print(
    "Number of bins:"
)

print(
    "  {}"
    .format(
        N_BINS
    )
)


print(
    "Sum of percentages across all bins:"
)

print(
    "  {:.10f}%"
    .format(
        histogram_percentage_sum
    )
)


# Show principal figure.

plt.show()


plt.close(
    fig
)


# ============================================================
# 36. OPTIONAL LOG1P DIAGNOSTIC
# ============================================================

log_png_path = None
log_pdf_path = None


if (
    CREATE_LOG1P_DIAGNOSTIC
    and skewness > 2
    and minimum >= 0
):

    log_values = np.log1p(
        valid_target.to_numpy()
    )


    fig_log, ax_log = plt.subplots(
        figsize=(
            9.0,
            5.8
        )
    )


    ax_log.hist(

        log_values,

        bins=N_BINS,

        weights=weights,

        edgecolor="black",

        linewidth=0.5,

        alpha=0.80
    )


    ax_log.set_title(
        "Supplementary Diagnostic - Log1p Distribution of School-Occupied Degree-Hours"
    )


    ax_log.set_xlabel(
        "log1p(School-occupied degree-hours above 26 degC)"
    )


    ax_log.set_ylabel(
        "Simulation cases (%)"
    )


    ax_log.grid(
        axis="y",
        alpha=0.25,
        linewidth=0.8
    )


    ax_log.set_axisbelow(
        True
    )


    fig_log.text(

        0.99,

        0.01,

        (
            "Supplementary diagnostic only - "
            "original target unchanged; N = {:,}"
        )
        .format(
            valid_count
        ),

        ha="right",

        va="bottom",

        fontsize=9
    )


    fig_log.tight_layout(
        rect=[
            0,
            0.035,
            1,
            1
        ]
    )


    log_png_path = (
        output_dir
        / "Supplementary_Log1p_Degree_Hours_Distribution.png"
    )


    log_pdf_path = (
        output_dir
        / "Supplementary_Log1p_Degree_Hours_Distribution.pdf"
    )


    fig_log.savefig(
        log_png_path,
        dpi=300,
        bbox_inches="tight"
    )


    fig_log.savefig(
        log_pdf_path,
        bbox_inches="tight"
    )


    plt.show()


    plt.close(
        fig_log
    )


    print(
        "\nSupplementary log1p histogram created."
    )

    print(
        "Reason: skewness = {:.3f}"
        .format(
            skewness
        )
    )


else:

    print(
        "\nNo supplementary log1p histogram was generated."
    )


# ============================================================
# 37. CREATE FACTUAL SUMMARY TEXT FILE
# ============================================================

summary_path = (
    output_dir
    / "Section_4_2_1_Analysis_Summary.txt"
)


summary_lines = [

    "SECTION 4.2.1 - DISTRIBUTION OF DEGREE-HOURS",

    "=" * 70,

    "",

    "SOURCE",

    "Input file: "
    + str(
        INPUT_FILE.resolve()
    ),

    "Worksheet: "
    + str(
        selected_sheet
    ),

    "Target column: "
    + str(
        target_column
    ),

    "",

    "DATA QUALITY",

    "Total dataset rows: {:,}"
    .format(
        total_rows
    ),

    "Expected rows: {:,}"
    .format(
        EXPECTED_ROWS
    ),

    "Expected row count matched: {}"
    .format(
        expected_rows_match
    ),

    "One row per simulation case: "
    + one_row_per_case_text,

    "Valid finite target values: {:,}"
    .format(
        valid_count
    ),

    "Missing target values: {:,}"
    .format(
        missing_count
    ),

    "Positive infinite values: {:,}"
    .format(
        positive_infinity_count
    ),

    "Negative infinite values: {:,}"
    .format(
        negative_infinity_count
    ),

    "Negative degree-hour values: {:,}"
    .format(
        negative_degree_hour_count
    ),

    "Zero-degree-hour cases: {:,}"
    .format(
        zero_degree_hour_count
    ),

    "",

    "DESCRIPTIVE STATISTICS",

    "Minimum: {:.2f} degC h"
    .format(
        minimum
    ),

    "25th percentile: {:.2f} degC h"
    .format(
        percentile_25
    ),

    "Median: {:.2f} degC h"
    .format(
        median
    ),

    "Mean: {:.2f} degC h"
    .format(
        mean
    ),

    "75th percentile: {:.2f} degC h"
    .format(
        percentile_75
    ),

    "Maximum: {:.2f} degC h"
    .format(
        maximum
    ),

    "Sample standard deviation: {:.2f} degC h"
    .format(
        standard_deviation_sample
    ),

    "Skewness: {:.3f}"
    .format(
        skewness
    ),

    "Zero-degree-hour percentage: {:.2f}%"
    .format(
        zero_percentage
    ),

    "",

    "HISTOGRAM",

    "Bins: {}"
    .format(
        N_BINS
    ),

    "Percentage sum: {:.10f}%"
    .format(
        histogram_percentage_sum
    )
]


summary_path.write_text(
    "\n".join(
        summary_lines
    ),
    encoding="utf-8"
)


# ============================================================
# 38. FACTUAL DISTRIBUTION SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FACTUAL DISTRIBUTION SUMMARY")
print("=" * 80)


print(
    "The analysis contains {:,} valid finite simulation cases."
    .format(
        valid_count
    )
)


print(
    "School-occupied degree-hours above 26 degC range "
    "from {:,.2f} to {:,.2f} degC h."
    .format(
        minimum,
        maximum
    )
)


print(
    "The 25th percentile is {:,.2f} degC h."
    .format(
        percentile_25
    )
)


print(
    "The median is {:,.2f} degC h."
    .format(
        median
    )
)


print(
    "The mean is {:,.2f} degC h."
    .format(
        mean
    )
)


print(
    "The 75th percentile is {:,.2f} degC h."
    .format(
        percentile_75
    )
)


print(
    "The sample standard deviation is {:,.2f} degC h."
    .format(
        standard_deviation_sample
    )
)


print(
    "The skewness is {:.3f}."
    .format(
        skewness
    )
)


print(
    "{:,} cases have zero degree-hours, representing {:.2f}% "
    "of valid cases."
    .format(
        zero_degree_hour_count,
        zero_percentage
    )
)


# ============================================================
# 39. PRINT ALL OUTPUT PATHS
# ============================================================

print("\n" + "=" * 80)
print("SAVED OUTPUTS")
print("=" * 80)


print(
    "\nStatistics Excel:"
)

print(
    statistics_excel_path.resolve()
)


print(
    "\nStatistics CSV:"
)

print(
    statistics_csv_path.resolve()
)


print(
    "\nPrincipal histogram PNG - 300 dpi:"
)

print(
    png_path.resolve()
)


print(
    "\nPrincipal histogram PDF - vector:"
)

print(
    pdf_path.resolve()
)


print(
    "\nPrincipal histogram SVG - vector:"
)

print(
    svg_path.resolve()
)


print(
    "\nAnalysis summary:"
)

print(
    summary_path.resolve()
)


if log_png_path is not None:

    print(
        "\nSupplementary log1p PNG:"
    )

    print(
        log_png_path.resolve()
    )


if log_pdf_path is not None:

    print(
        "\nSupplementary log1p PDF:"
    )

    print(
        log_pdf_path.resolve()
    )


print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(
    "\nOriginal input workbook was NOT modified."
)
