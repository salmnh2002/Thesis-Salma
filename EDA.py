import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

file_path = r"E:\Thesis\Thesis Data\Master Data.xlsx"
output_folder = r"E:\Thesis\Thesis Data\EDA Results"

os.makedirs(output_folder, exist_ok=True)


# ============================================================
# 2. ANALYSIS-PERIOD DENOMINATORS
# ============================================================

# These values must match the filters used to create the temperature summary:
# All hours: 8,760 annual hours.
# Office hours: weekdays, 08:00-16:00 inclusive = 2,349 hours in 2026.
# School-occupied hours: the defined school calendar and exclusions = 1,611 hours.

ALL_HOURS_COUNT = 8760
OFFICE_HOURS_COUNT = 2349
SCHOOL_OCCUPIED_HOURS_COUNT = 1611


# ============================================================
# 3. LOAD DATA AND CLEAN COLUMN NAMES
# ============================================================

df = pd.read_excel(file_path)


def clean_column_name(name):
    """Convert Excel headings to consistent lowercase snake_case names."""
    name = str(name).replace("\xa0", " ").replace("\n", " ").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


df.columns = [clean_column_name(col) for col in df.columns]

print("Available columns after cleaning:")
print(df.columns.tolist())


def resolve_column(label, *candidates):
    """Return the first available candidate name or report a clear error."""
    cleaned_candidates = [clean_column_name(candidate) for candidate in candidates]

    for candidate in cleaned_candidates:
        if candidate in df.columns:
            return candidate

    raise KeyError(
        f"Missing column for '{label}'. Expected one of: {cleaned_candidates}"
    )


# ============================================================
# 4. RESOLVE THE NEW HEADINGS
# ============================================================

location_col = resolve_column("Location", "Location")
case_col = resolve_column("Case ID", "Case ID")

hours_26_all_col = resolve_column(
    "Hours above 26 degrees - all hours",
    "Hours above 26 degrees all hours",
    "Hours above 26 all hours",
)

hours_26_office_col = resolve_column(
    "Hours above 26 degrees - office hours",
    "Hours above 26 degrees office hours",
    "Hours above 26 office hours",
)

hours_26_school_col = resolve_column(
    "Hours above 26 degrees - school-occupied hours",
    "Hours above 26 degrees school occupied hours",
    "Hours above 26 school occupied",
    "Hours above 26 school occupied hours",
)

degree_26_school_col = resolve_column(
    "Degree-hours above 26 degrees - school-occupied hours",
    "Degree Hours Above 26 degrees school occupied hours",
    "Degree hours above 26 school occupied",
    "Degree hours above 26 school occupied hours",
)

vent_col = resolve_column(
    "Ventilation rate",
    "Ventilation Rate (L/s/person)",
    "Ventilation lps person",
)

wfr_col = resolve_column(
    "Window-to-floor ratio",
    "Window to Floor Ratio",
    "Window to floor ratio WFR",
)

orientation_col = resolve_column(
    "Building orientation",
    "Building Orientation",
)

wall_u_col = resolve_column(
    "Wall thermal transmittance",
    "Wall Thermal Transmittance",
    "Wall U Value",
)

roof_u_col = resolve_column(
    "Roof thermal transmittance",
    "Roof Thermal Transmittance",
    "Roof U Value",
)

window_u_col = resolve_column(
    "Window thermal transmittance",
    "Window Thermal Transmittance",
    "Window U Value",
)

g_col = resolve_column(
    "Solar heat gain coefficient",
    "SHGC",
    "Solar Heat Gain Coefficient",
)

floor_col = resolve_column("Floor", "Floor")

# The code accepts both the recommended headings ending in "With Ventilation"
# and the original headings without that ending. In both cases, these are
# interpreted as the ventilated results.

cat1_with_col = resolve_column(
    "Category I exceedance hours with ventilation",
    "Hours exceeding the Category I upper comfort limit With Ventilation",
    "Hours exceeding the Category I upper comfort limit",
)

cat2_with_col = resolve_column(
    "Category II exceedance hours with ventilation",
    "Hours exceeding the Category II upper comfort limit With Ventilation",
    "Hours exceeding the Category II upper comfort limit",
)

cat3_with_col = resolve_column(
    "Category III exceedance hours with ventilation",
    "Hours exceeding the Category III upper comfort limit With Ventilation",
    "Hours exceeding the Category III upper comfort limit",
)

overheating_with_col = resolve_column(
    "Overheating with ventilation",
    "Overheating With Ventilation",
    "Overheating",
)

cat1_without_col = resolve_column(
    "Category I exceedance hours without ventilation",
    "Hours exceeding the Category I upper comfort limit Without Ventilation",
)

cat2_without_col = resolve_column(
    "Category II exceedance hours without ventilation",
    "Hours exceeding the Category II upper comfort limit Without Ventilation",
)

cat3_without_col = resolve_column(
    "Category III exceedance hours without ventilation",
    "Hours exceeding the Category III upper comfort limit Without Ventilation",
)

overheating_without_col = resolve_column(
    "Overheating without ventilation",
    "Overheating Without Ventilation",
)


# ============================================================
# 5. CLEAN VALUES
# ============================================================

df[location_col] = df[location_col].astype(str).str.strip().str.title()

location_map = {
    "A": "A - Lampedusa",
    "B": "B - Palermo",
    "C": "C - Napoli",
    "D": "D - Roma",
    "E": "E - Milano",
    "F": "F - Cuneo",
    "Lampedusa": "A - Lampedusa",
    "Palermo": "B - Palermo",
    "Napoli": "C - Napoli",
    "Roma": "D - Roma",
    "Milano": "E - Milano",
    "Cuneo": "F - Cuneo",
}

df["climate_zone"] = df[location_col].map(location_map).fillna(df[location_col])
climate_col = "climate_zone"

original_floor = df[floor_col].copy()

floor_clean = df[floor_col].astype(str).str.strip().str.title()
df[floor_col] = floor_clean.map(
    {
        "G": "Ground",
        "M": "Middle",
        "T": "Top",
        "Ground": "Ground",
        "Middle": "Middle",
        "Top": "Top",
    }
)
df[floor_col] = df[floor_col].fillna(original_floor)


def convert_binary(series):
    """Convert common Boolean/yes-no/0-1 values to numeric 0 or 1."""
    text_values = series.astype(str).str.upper().str.strip()
    mapped = text_values.map(
        {
            "TRUE": 1,
            "FALSE": 0,
            "YES": 1,
            "NO": 0,
            "1": 1,
            "0": 0,
            "1.0": 1,
            "0.0": 0,
        }
    )

    numeric_values = pd.to_numeric(series, errors="coerce")
    numeric_values = numeric_values.where(numeric_values.isin([0, 1]))
    return mapped.fillna(numeric_values)


df["overheating_with_num"] = convert_binary(df[overheating_with_col])
df["overheating_without_num"] = convert_binary(df[overheating_without_col])

numeric_cols = [
    hours_26_all_col,
    hours_26_office_col,
    hours_26_school_col,
    degree_26_school_col,
    vent_col,
    wfr_col,
    orientation_col,
    wall_u_col,
    roof_u_col,
    window_u_col,
    g_col,
    cat1_with_col,
    cat2_with_col,
    cat3_with_col,
    cat1_without_col,
    cat2_without_col,
    cat3_without_col,
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

standard_climate_order = [
    "A - Lampedusa",
    "B - Palermo",
    "C - Napoli",
    "D - Roma",
    "E - Milano",
    "F - Cuneo",
]

present_climates = set(df[climate_col].dropna().unique())
climate_order = [
    climate for climate in standard_climate_order if climate in present_climates
]
climate_order += sorted(present_climates.difference(standard_climate_order))


# ============================================================
# 6. CALCULATE VENTILATION BENEFIT
# ============================================================

# Positive values mean ventilation reduced the number of exceedance hours.
# Negative values mean more exceedance hours occurred with ventilation.

df["cat_i_reduction_hours"] = df[cat1_without_col] - df[cat1_with_col]
df["cat_ii_reduction_hours"] = df[cat2_without_col] - df[cat2_with_col]
df["cat_iii_reduction_hours"] = df[cat3_without_col] - df[cat3_with_col]

reduction_cols = [
    "cat_i_reduction_hours",
    "cat_ii_reduction_hours",
    "cat_iii_reduction_hours",
]


# ============================================================
# 7. DATA-QUALITY CHECKS
# ============================================================

paired_cols = [
    cat1_with_col,
    cat2_with_col,
    cat3_with_col,
    cat1_without_col,
    cat2_without_col,
    cat3_without_col,
]

missing_pair_mask = df[paired_cols].isna().any(axis=1)

invalid_order_with = ~(
    (df[cat1_with_col] >= df[cat2_with_col])
    & (df[cat2_with_col] >= df[cat3_with_col])
)

invalid_order_without = ~(
    (df[cat1_without_col] >= df[cat2_without_col])
    & (df[cat2_without_col] >= df[cat3_without_col])
)

# Category I-III exceedance values are calculated for all annual hours,
# so their valid range is 0-8,760 hours.
invalid_range = pd.Series(False, index=df.index)
for col in paired_cols:
    invalid_range = invalid_range | (df[col] < 0) | (
        df[col] > ALL_HOURS_COUNT
    )

comfort_check_failures = df.loc[
    missing_pair_mask | invalid_order_with | invalid_order_without | invalid_range,
    [location_col, case_col, vent_col] + paired_cols,
].copy()

comfort_check_failures["missing_with_without_pair"] = missing_pair_mask.loc[
    comfort_check_failures.index
]
comfort_check_failures["invalid_category_order_with_ventilation"] = (
    invalid_order_with.loc[comfort_check_failures.index]
)
comfort_check_failures["invalid_category_order_without_ventilation"] = (
    invalid_order_without.loc[comfort_check_failures.index]
)
comfort_check_failures["outside_valid_annual_hour_range"] = invalid_range.loc[
    comfort_check_failures.index
]

if comfort_check_failures.empty:
    print("Comfort-limit checks passed.")
else:
    print(
        "Warning:",
        len(comfort_check_failures),
        "rows failed at least one comfort-limit check.",
    )

vent_balance_check = (
    df.groupby([climate_col, vent_col], dropna=False)
    .size()
    .reset_index(name="count_cases")
)


# ============================================================
# 8. HELPER FUNCTIONS
# ============================================================


def save_current_plot(filename):
    path = os.path.join(output_folder, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", filename)


def plot_hours_and_degree_by_parameter(
    dataframe,
    parameter_col,
    title,
    xlabel,
    filename,
):
    summary = (
        dataframe.groupby([climate_col, parameter_col], dropna=True)
        .agg(
            avg_hours_above_26=(hours_26_school_col, "mean"),
            avg_degree_hours_above_26=(degree_26_school_col, "mean"),
            count_cases=(case_col, "size"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for climate in climate_order:
        subset = summary[summary[climate_col] == climate].sort_values(parameter_col)

        if subset.empty:
            continue

        axes[0].plot(
            subset[parameter_col],
            subset["avg_hours_above_26"],
            marker="o",
            markersize=4,
            label=climate,
        )

        axes[1].plot(
            subset[parameter_col],
            subset["avg_degree_hours_above_26"],
            marker="o",
            markersize=4,
            label=climate,
        )

    axes[0].set_title("Average hours above 26 degrees C")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Average hours above 26 degrees C")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Average degree-hours above 26 degrees C")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Average degree-hours above 26 degrees C")
    axes[1].grid(True, alpha=0.3)

    axes[1].legend(
        title="Climate zone",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    fig.suptitle(title, fontsize=14)
    save_current_plot(filename)
    return summary


# ============================================================
# 9. SUPPORTING - ALL, OFFICE AND SCHOOL PERIOD COMPARISON
# ============================================================

# Percentages are used because the three periods contain different numbers
# of evaluated hours.

period_percentage_data = pd.DataFrame(
    {
        "All hours": df[hours_26_all_col] / ALL_HOURS_COUNT * 100,
        "Office hours": df[hours_26_office_col] / OFFICE_HOURS_COUNT * 100,
        "School-occupied hours": (
            df[hours_26_school_col] / SCHOOL_OCCUPIED_HOURS_COUNT * 100
        ),
        climate_col: df[climate_col],
    }
)

period_percentage_by_climate = (
    period_percentage_data.groupby(climate_col)[
        ["All hours", "Office hours", "School-occupied hours"]
    ]
    .mean()
    .reindex(climate_order)
)

plt.figure(figsize=(11, 6))
period_percentage_by_climate.plot(kind="bar", ax=plt.gca())
plt.title("Percentage of Hours Above 26 degrees C by Analysis Period and Climate Zone")
plt.xlabel("Climate zone and location")
plt.ylabel("Hours above 26 degrees C (%)")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Analysis period")
plt.grid(axis="y", alpha=0.3)
save_current_plot("00_hours_above_26_percentage_by_period_and_climate.png")


# ============================================================
# 10. SUPPORTING - OVERHEATING BY CLIMATE ZONE (WITH VENTILATION)
# ============================================================

overheating_by_climate = (
    df.groupby(climate_col)["overheating_with_num"]
    .mean()
    .reindex(climate_order)
    * 100
)

plt.figure(figsize=(9, 5))
overheating_by_climate.plot(kind="bar")
plt.title("Overheating Percentage by Climate Zone - With Ventilation")
plt.xlabel("Climate zone and location")
plt.ylabel("Overheating (%)")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
save_current_plot("S01_overheating_percentage_by_climate_zone.png")


# ============================================================
# 11. MAIN - DEGREE-HOURS BY CLIMATE ZONE
# ============================================================

degree_data = [
    df[df[climate_col] == climate][degree_26_school_col].dropna()
    for climate in climate_order
]

nonempty_pairs = [
    (climate, data)
    for climate, data in zip(climate_order, degree_data)
    if not data.empty
]

if nonempty_pairs:
    valid_labels, valid_degree_data = zip(*nonempty_pairs)
    plt.figure(figsize=(10, 5))
    plt.boxplot(valid_degree_data, tick_labels=valid_labels)
    plt.title(
        "Degree-Hours Above 26 degrees C During School-Occupied Hours by Climate Zone"
    )
    plt.xlabel("Climate zone and location")
    plt.ylabel("Degree-hours above 26 degrees C")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    save_current_plot("01_degree_hours_above_26_by_climate_zone.png")

degree_by_climate = (
    df.groupby(climate_col)[degree_26_school_col]
    .describe()
    .reindex(climate_order)
)


# ============================================================
# 12. MAIN - CATEGORY I, II AND III EXCEEDANCE VS DEGREE-HOURS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

scatter_metrics = [
    (
        cat1_with_col,
        "Category I exceedance vs degree-hours",
        "Annual hours exceeding Category I upper limit",
    ),
    (
        cat2_with_col,
        "Category II exceedance vs degree-hours",
        "Annual hours exceeding Category II upper limit",
    ),
    (
        cat3_with_col,
        "Category III exceedance vs degree-hours",
        "Annual hours exceeding Category III upper limit",
    ),
]

for ax, (metric_col, title, ylabel) in zip(axes, scatter_metrics):
    for climate in climate_order:
        subset = df[df[climate_col] == climate]
        ax.scatter(
            subset[degree_26_school_col],
            subset[metric_col],
            alpha=0.45,
            s=8,
            label=climate,
        )

    ax.set_title(title)
    ax.set_xlabel("Degree-hours above 26 degrees C")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

axes[2].legend(
    title="Climate zone",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)

fig.suptitle(
    "Annual Upper Comfort-Limit Exceedance vs "
    "School-Occupied Degree-Hours - With Ventilation",
    fontsize=14,
)
save_current_plot("02_comfort_limit_exceedance_vs_degree_hours.png")


# ============================================================
# 13. MAIN - DEGREE-HOURS BY ORIENTATION AND SHGC
# ============================================================

orientation_shgc = (
    df.groupby([orientation_col, g_col])[degree_26_school_col]
    .mean()
    .reset_index()
)

orientation_pivot = orientation_shgc.pivot(
    index=orientation_col,
    columns=g_col,
    values=degree_26_school_col,
)

plt.figure(figsize=(9, 5))
orientation_pivot.plot(kind="bar", ax=plt.gca())
plt.title("Degree-Hours Above 26 degrees C by Orientation and SHGC")
plt.xlabel("Building orientation (degrees)")
plt.ylabel("Average degree-hours above 26 degrees C")
plt.xticks(rotation=0)
plt.legend(title="SHGC")
plt.grid(axis="y", alpha=0.3)
save_current_plot("03_degree_hours_by_orientation_and_SHGC.png")


# ============================================================
# 14. MAIN - WINDOW U-VALUE
# ============================================================

window_u_hours_degree = plot_hours_and_degree_by_parameter(
    df,
    window_u_col,
    "Window U-Value Effect by Climate Zone",
    "Window U-value (W/m^2K)",
    "04_window_u_value_by_climate_zone.png",
)


# ============================================================
# 15. MAIN - WALL U-VALUE
# ============================================================

wall_u_hours_degree = plot_hours_and_degree_by_parameter(
    df,
    wall_u_col,
    "Wall U-Value Effect by Climate Zone",
    "Wall U-value (W/m^2K)",
    "05_wall_u_value_by_climate_zone.png",
)


# ============================================================
# 16. MAIN - ROOF U-VALUE, TOP FLOOR ONLY
# ============================================================

df_top = df[df[floor_col] == "Top"].copy()

roof_u_hours_degree_top = plot_hours_and_degree_by_parameter(
    df_top,
    roof_u_col,
    "Roof U-Value Effect by Climate Zone - Top Floor Only",
    "Roof U-value (W/m^2K)",
    "06_roof_u_value_by_climate_zone_top_floor_only.png",
)


# ============================================================
# 17. NEW MAIN - VENTILATION BENEFIT BY RATE AND CLIMATE ZONE
# ============================================================

vent_benefit_by_rate = (
    df.groupby([climate_col, vent_col])
    .agg(
        avg_reduction_cat_i=("cat_i_reduction_hours", "mean"),
        avg_reduction_cat_ii=("cat_ii_reduction_hours", "mean"),
        avg_reduction_cat_iii=("cat_iii_reduction_hours", "mean"),
        count_cases=(case_col, "size"),
    )
    .reset_index()
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

vent_panels = [
    ("avg_reduction_cat_i", "Category I"),
    ("avg_reduction_cat_ii", "Category II"),
    ("avg_reduction_cat_iii", "Category III"),
]

for ax, (reduction_col, category_label) in zip(axes, vent_panels):
    for climate in climate_order:
        subset = vent_benefit_by_rate[
            vent_benefit_by_rate[climate_col] == climate
        ].sort_values(vent_col)

        if subset.empty:
            continue

        ax.plot(
            subset[vent_col],
            subset[reduction_col],
            marker="o",
            markersize=4,
            label=climate,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(category_label)
    ax.set_xlabel("Ventilation rate (L/s/person)")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel(
    "Average annual reduction in exceedance hours\n"
    "(without ventilation - with ventilation)"
)

axes[2].legend(
    title="Climate zone",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)

fig.suptitle(
    "Annual Reduction in Upper Comfort-Limit Exceedance Hours "
    "Due to Ventilation",
    fontsize=14,
)
save_current_plot("07_ventilation_benefit_by_rate_climate_and_category.png")


# ============================================================
# 18. SUPPORTING - WITH VS WITHOUT VENTILATION BY CLIMATE ZONE
# ============================================================

vent_condition_by_climate = (
    df.groupby(climate_col)
    .agg(
        cat_i_with=(cat1_with_col, "mean"),
        cat_i_without=(cat1_without_col, "mean"),
        cat_ii_with=(cat2_with_col, "mean"),
        cat_ii_without=(cat2_without_col, "mean"),
        cat_iii_with=(cat3_with_col, "mean"),
        cat_iii_without=(cat3_without_col, "mean"),
    )
    .reindex(climate_order)
)

fig, axes = plt.subplots(1, 3, figsize=(19, 5), sharey=True)
x_positions = np.arange(len(climate_order))
bar_width = 0.38

condition_panels = [
    ("cat_i_with", "cat_i_without", "Category I"),
    ("cat_ii_with", "cat_ii_without", "Category II"),
    ("cat_iii_with", "cat_iii_without", "Category III"),
]

for ax, (with_col, without_col, category_label) in zip(axes, condition_panels):
    ax.bar(
        x_positions - bar_width / 2,
        vent_condition_by_climate[with_col],
        width=bar_width,
        label="With ventilation",
    )
    ax.bar(
        x_positions + bar_width / 2,
        vent_condition_by_climate[without_col],
        width=bar_width,
        label="Without ventilation",
    )
    ax.set_title(category_label)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(climate_order, rotation=45, ha="right")
    ax.set_xlabel("Climate zone and location")
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel(
    "Average annual hours exceeding the upper comfort limit"
)
axes[2].legend(
    title="Condition",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)

fig.suptitle(
    "Annual Upper Comfort-Limit Exceedance "
    "With and Without Ventilation",
    fontsize=14,
)
save_current_plot("S02_with_vs_without_ventilation_by_climate_and_category.png")


# ============================================================
# 19. SUPPORTING - OVERHEATING BY FLOOR AND CLIMATE ZONE
# ============================================================

floor_climate = (
    df.groupby([climate_col, floor_col])["overheating_with_num"]
    .mean()
    .reset_index()
)

floor_climate["overheating_percent"] = (
    floor_climate["overheating_with_num"] * 100
)

floor_climate_pivot = floor_climate.pivot(
    index=climate_col,
    columns=floor_col,
    values="overheating_percent",
).reindex(climate_order)

plt.figure(figsize=(10, 5))
floor_climate_pivot.plot(kind="bar", ax=plt.gca())
plt.title("Overheating by Floor Position and Climate Zone - With Ventilation")
plt.xlabel("Climate zone and location")
plt.ylabel("Overheating (%)")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Floor position")
plt.grid(axis="y", alpha=0.3)
save_current_plot("S03_overheating_by_floor_and_climate_zone.png")


# ============================================================
# 20. SUPPORTING - DEGREE-HOURS BY FLOOR AND CLIMATE ZONE
# ============================================================

floor_degree = (
    df.groupby([climate_col, floor_col])[degree_26_school_col]
    .mean()
    .reset_index()
)

floor_degree_pivot = floor_degree.pivot(
    index=climate_col,
    columns=floor_col,
    values=degree_26_school_col,
).reindex(climate_order)

plt.figure(figsize=(10, 5))
floor_degree_pivot.plot(kind="bar", ax=plt.gca())
plt.title("Degree-Hours Above 26 degrees C by Floor Position and Climate Zone")
plt.xlabel("Climate zone and location")
plt.ylabel("Average degree-hours above 26 degrees C")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Floor position")
plt.grid(axis="y", alpha=0.3)
save_current_plot("08_degree_hours_by_floor_and_climate_zone.png")


# ============================================================
# 21. SUPPORTING - WFR ANALYSIS
# ============================================================

wfr_summary = (
    df.groupby(wfr_col)
    .agg(
        avg_hours_above_26=(hours_26_school_col, "mean"),
        avg_degree_hours=(degree_26_school_col, "mean"),
        overheating_percent=(
            "overheating_with_num",
            lambda values: values.mean() * 100,
        ),
    )
    .reset_index()
    .sort_values(wfr_col)
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(
    wfr_summary[wfr_col],
    wfr_summary["avg_hours_above_26"],
    marker="o",
    markersize=4,
)
axes[0].set_title("WFR vs hours above 26 degrees C")
axes[0].set_xlabel("Window-to-floor ratio (WFR)")
axes[0].set_ylabel("Average hours above 26 degrees C")
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    wfr_summary[wfr_col],
    wfr_summary["avg_degree_hours"],
    marker="o",
    markersize=4,
)
axes[1].set_title("WFR vs degree-hours above 26 degrees C")
axes[1].set_xlabel("Window-to-floor ratio (WFR)")
axes[1].set_ylabel("Average degree-hours above 26 degrees C")
axes[1].grid(True, alpha=0.3)

axes[2].plot(
    wfr_summary[wfr_col],
    wfr_summary["overheating_percent"],
    marker="o",
    markersize=4,
)
axes[2].set_title("WFR vs overheating")
axes[2].set_xlabel("Window-to-floor ratio (WFR)")
axes[2].set_ylabel("Overheating (%)")
axes[2].grid(True, alpha=0.3)

save_current_plot("S04_WFR_hours_degree_hours_and_overheating.png")


# ============================================================
# 22. SUPPORTING - COMFORT-LIMIT EXCEEDANCE BY CLIMATE ZONE
# ============================================================

comfort_exceedance_by_climate = (
    df.groupby(climate_col)[[cat1_with_col, cat2_with_col, cat3_with_col]]
    .mean()
    .reindex(climate_order)
)

comfort_exceedance_by_climate.columns = [
    "Category I",
    "Category II",
    "Category III",
]

plt.figure(figsize=(10, 5))
comfort_exceedance_by_climate.plot(kind="bar", ax=plt.gca())
plt.title(
    "Annual Upper Comfort-Limit Exceedance Hours "
    "by Climate Zone - With Ventilation"
)
plt.xlabel("Climate zone and location")
plt.ylabel("Average annual hours exceeding the upper comfort limit")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Comfort category")
plt.grid(axis="y", alpha=0.3)
save_current_plot("S05_comfort_limit_exceedance_by_climate_zone.png")


# ============================================================
# 23. DIAGNOSTIC CHECKS - VENTILATION
# ============================================================

vent_check_summary = (
    df.groupby([climate_col, vent_col])
    .agg(
        count_cases=(case_col, "size"),
        avg_cat_i_with=(cat1_with_col, "mean"),
        avg_cat_i_without=(cat1_without_col, "mean"),
        avg_cat_i_reduction=("cat_i_reduction_hours", "mean"),
        avg_cat_ii_with=(cat2_with_col, "mean"),
        avg_cat_ii_without=(cat2_without_col, "mean"),
        avg_cat_ii_reduction=("cat_ii_reduction_hours", "mean"),
        avg_cat_iii_with=(cat3_with_col, "mean"),
        avg_cat_iii_without=(cat3_without_col, "mean"),
        avg_cat_iii_reduction=("cat_iii_reduction_hours", "mean"),
        avg_window_u=(window_u_col, "mean"),
        avg_wall_u=(wall_u_col, "mean"),
        avg_roof_u=(roof_u_col, "mean"),
        avg_shgc=(g_col, "mean"),
        avg_wfr=(wfr_col, "mean"),
    )
    .reset_index()
)

vent_check_floor = (
    df.groupby([climate_col, vent_col, floor_col])
    .size()
    .reset_index(name="count_cases")
)

vent_check_orientation = (
    df.groupby([climate_col, vent_col, orientation_col])
    .size()
    .reset_index(name="count_cases")
)

vent_check_window_u = (
    df.groupby([climate_col, vent_col, window_u_col])
    .size()
    .reset_index(name="count_cases")
)

vent_check_wall_u = (
    df.groupby([climate_col, vent_col, wall_u_col])
    .size()
    .reset_index(name="count_cases")
)

vent_check_roof_u = (
    df.groupby([climate_col, vent_col, roof_u_col])
    .size()
    .reset_index(name="count_cases")
)

vent_check_shgc = (
    df.groupby([climate_col, vent_col, g_col])
    .size()
    .reset_index(name="count_cases")
)

vent_check_wfr = (
    df.groupby([climate_col, vent_col, wfr_col])
    .size()
    .reset_index(name="count_cases")
)


# ============================================================
# 24. DIAGNOSTIC CHECKS - ROOF U-VALUE, TOP FLOOR ONLY
# ============================================================

roof_top_check_summary = (
    df_top.groupby([climate_col, roof_u_col])
    .agg(
        count_cases=(case_col, "size"),
        avg_hours_above_26=(hours_26_school_col, "mean"),
        avg_degree_hours=(degree_26_school_col, "mean"),
        avg_cat_i=(cat1_with_col, "mean"),
        avg_cat_ii=(cat2_with_col, "mean"),
        avg_cat_iii=(cat3_with_col, "mean"),
        avg_window_u=(window_u_col, "mean"),
        avg_wall_u=(wall_u_col, "mean"),
        avg_shgc=(g_col, "mean"),
        avg_wfr=(wfr_col, "mean"),
        avg_ventilation=(vent_col, "mean"),
    )
    .reset_index()
)

roof_top_check_orientation = (
    df_top.groupby([climate_col, roof_u_col, orientation_col])
    .size()
    .reset_index(name="count_cases")
)

roof_top_check_shgc = (
    df_top.groupby([climate_col, roof_u_col, g_col])
    .size()
    .reset_index(name="count_cases")
)


# ============================================================
# 25. DIAGNOSTIC CHECKS - WALL U-VALUE
# ============================================================

wall_check_summary = (
    df.groupby([climate_col, wall_u_col])
    .agg(
        count_cases=(case_col, "size"),
        avg_hours_above_26=(hours_26_school_col, "mean"),
        avg_degree_hours=(degree_26_school_col, "mean"),
        avg_window_u=(window_u_col, "mean"),
        avg_roof_u=(roof_u_col, "mean"),
        avg_shgc=(g_col, "mean"),
        avg_ventilation=(vent_col, "mean"),
        avg_wfr=(wfr_col, "mean"),
    )
    .reset_index()
)

wall_check_floor = (
    df.groupby([climate_col, wall_u_col, floor_col])
    .size()
    .reset_index(name="count_cases")
)

wall_check_orientation = (
    df.groupby([climate_col, wall_u_col, orientation_col])
    .size()
    .reset_index(name="count_cases")
)


# ============================================================
# 26. CORRELATION MATRIX
# ============================================================

# The correlation matrix keeps the original ventilated outcomes. The new
# ventilation benefit is analysed separately in Sections 17 and 18.

corr_cols = [
    hours_26_school_col,
    degree_26_school_col,
    vent_col,
    wfr_col,
    orientation_col,
    wall_u_col,
    roof_u_col,
    window_u_col,
    g_col,
    cat1_with_col,
    cat2_with_col,
    cat3_with_col,
    "overheating_with_num",
]

corr_labels = [
    "Hours >26 degrees C",
    "Degree-hours >26 degrees C",
    "Ventilation",
    "WFR",
    "Orientation",
    "Wall U-value",
    "Roof U-value",
    "Window U-value",
    "SHGC",
    "Cat. I exceedance",
    "Cat. II exceedance",
    "Cat. III exceedance",
    "Overheating",
]

corr_df = df[corr_cols].corr()

plt.figure(figsize=(12, 9))
plt.imshow(corr_df, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
plt.colorbar(label="Pearson correlation coefficient")
plt.xticks(range(len(corr_labels)), corr_labels, rotation=90)
plt.yticks(range(len(corr_labels)), corr_labels)
plt.title(
    "Correlation Matrix - School Overheating and Annual Comfort Outcomes"
)
save_current_plot("09_correlation_matrix.png")

corr_df_labeled = corr_df.copy()
corr_df_labeled.index = corr_labels
corr_df_labeled.columns = corr_labels


# ============================================================
# 27. SUMMARY EXCEL TABLES
# ============================================================

summary_file = os.path.join(output_folder, "EDA_summary_tables_revised.xlsx")

with pd.ExcelWriter(summary_file, engine="openpyxl") as writer:
    period_percentage_by_climate.to_excel(writer, sheet_name="PeriodPct_Climate")
    overheating_by_climate.to_excel(writer, sheet_name="S_Overheat_Climate")
    degree_by_climate.to_excel(writer, sheet_name="Degree_Climate")

    orientation_shgc.to_excel(
        writer,
        sheet_name="Degree_Orient_SHGC",
        index=False,
    )
    orientation_pivot.to_excel(writer, sheet_name="Orient_SHGC_Pivot")

    window_u_hours_degree.to_excel(
        writer,
        sheet_name="Window_U_Hours_Degree",
        index=False,
    )
    wall_u_hours_degree.to_excel(
        writer,
        sheet_name="Wall_U_Hours_Degree",
        index=False,
    )
    roof_u_hours_degree_top.to_excel(
        writer,
        sheet_name="Roof_U_Hours_Degree_Top",
        index=False,
    )

    vent_benefit_by_rate.to_excel(
        writer,
        sheet_name="Vent_Benefit_Rate",
        index=False,
    )
    vent_condition_by_climate.to_excel(
        writer,
        sheet_name="Vent_Condition_Climate",
    )
    vent_balance_check.to_excel(
        writer,
        sheet_name="Vent_Balance",
        index=False,
    )

    floor_climate.to_excel(writer, sheet_name="S_Overheat_Floor", index=False)
    floor_climate_pivot.to_excel(writer, sheet_name="Overheat_Floor_Pivot")
    floor_degree.to_excel(writer, sheet_name="Degree_Floor", index=False)
    floor_degree_pivot.to_excel(writer, sheet_name="Degree_Floor_Pivot")

    wfr_summary.to_excel(writer, sheet_name="S_WFR_Summary", index=False)
    comfort_exceedance_by_climate.to_excel(
        writer,
        sheet_name="S_Comfort_Exceedance",
    )

    vent_check_summary.to_excel(
        writer,
        sheet_name="Check_Vent_Summary",
        index=False,
    )
    vent_check_floor.to_excel(
        writer,
        sheet_name="Check_Vent_Floor",
        index=False,
    )
    vent_check_orientation.to_excel(
        writer,
        sheet_name="Check_Vent_Orient",
        index=False,
    )
    vent_check_window_u.to_excel(
        writer,
        sheet_name="Check_Vent_Window_U",
        index=False,
    )
    vent_check_wall_u.to_excel(
        writer,
        sheet_name="Check_Vent_Wall_U",
        index=False,
    )
    vent_check_roof_u.to_excel(
        writer,
        sheet_name="Check_Vent_Roof_U",
        index=False,
    )
    vent_check_shgc.to_excel(
        writer,
        sheet_name="Check_Vent_SHGC",
        index=False,
    )
    vent_check_wfr.to_excel(
        writer,
        sheet_name="Check_Vent_WFR",
        index=False,
    )

    roof_top_check_summary.to_excel(
        writer,
        sheet_name="Check_Roof_Top_Summary",
        index=False,
    )
    roof_top_check_orientation.to_excel(
        writer,
        sheet_name="Check_Roof_Top_Orient",
        index=False,
    )
    roof_top_check_shgc.to_excel(
        writer,
        sheet_name="Check_Roof_Top_SHGC",
        index=False,
    )

    wall_check_summary.to_excel(
        writer,
        sheet_name="Check_Wall_Summary",
        index=False,
    )
    wall_check_floor.to_excel(
        writer,
        sheet_name="Check_Wall_Floor",
        index=False,
    )
    wall_check_orientation.to_excel(
        writer,
        sheet_name="Check_Wall_Orient",
        index=False,
    )

    corr_df_labeled.to_excel(writer, sheet_name="Correlation")
    comfort_check_failures.to_excel(
        writer,
        sheet_name="Comfort_Check_Failures",
        index=False,
    )

print("Saved: EDA_summary_tables_revised.xlsx")
print("EDA finished successfully.")
print("Figures and summary tables saved in:")
print(output_folder)