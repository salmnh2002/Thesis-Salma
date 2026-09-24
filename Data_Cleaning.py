import pandas as pd

# =========================
# 1. LOAD FILE
# =========================
df = pd.read_excel(r"E:\Thesis\Thesis Data\Master Data Scenario\Master_NoCaseID.xlsx")

# =========================
# 2. CLEAN VALUES
# =========================
df.columns = df.columns.str.strip()

df["location"] = df["location"].astype(str).str.strip().str.title()
df["floor"] = df["floor"].astype(str).str.strip().str.upper()

df["ventilation_lps_person"] = pd.to_numeric(df["ventilation_lps_person"], errors="coerce").round(2)
df["window_to_floor_ratio"] = pd.to_numeric(df["window_to_floor_ratio"], errors="coerce").round(2)
df["building_orientation"] = pd.to_numeric(df["building_orientation"], errors="coerce").round(0)
df["wall_u_value"] = pd.to_numeric(df["wall_u_value"], errors="coerce").round(2)
df["roof_u_value"] = pd.to_numeric(df["roof_u_value"], errors="coerce").round(2)
df["window_u_value"] = pd.to_numeric(df["window_u_value"], errors="coerce").round(2)

# =========================
# 3. MAPPING DICTIONARIES
# =========================
vent_map = {
    2.4: "V1",
    5.0: "V2",
    8.0: "V3",
    10.0: "V4"
}

floor_map = {
    "G": "G",
    "M": "M",
    "T": "T"
}

wall_map = {
    1.1: "O1",
    0.7: "O2",
    0.32: "O3",
    0.15: "O4"
}

roof_map = {
    1.35: "R1",
    0.8: "R2",
    0.32: "R3",
    0.14: "R4"
}

window_map = {
    5.27: "G1",
    2.8: "G2",
    1.8: "G3",
    1.5: "G4",
    1.0: "G5"
}

orientation_map = {
    90.0: "E",
    0.0: "N",
    180.0: "S",
    270.0: "W"
}

wfr_map = {
    0.12: "W1",
    0.22: "W2"
}

location_map = {
    "Lampedusa": "A",
    "Palermo": "B",
    "Napoli": "C",
    "Roma": "D",
    "Milan": "E",
    "Cuneo": "F"
}

# =========================
# 4. CREATE CASE_ID
# Format: ETE-V1W1-R1O1G1
# =========================
df["Case_ID"] = (
    df["location"].map(location_map)
    + df["floor"].map(floor_map)
    + df["building_orientation"].map(orientation_map)
    + "-"
    + df["ventilation_lps_person"].map(vent_map)
    + df["window_to_floor_ratio"].map(wfr_map)
    + "-"
    + df["roof_u_value"].map(roof_map)
    + df["wall_u_value"].map(wall_map)
    + df["window_u_value"].map(window_map)
)

# Optional: move Case_ID to first column
cols = ["Case_ID"] + [col for col in df.columns if col != "Case_ID"]
df = df[cols]

# =========================
# 5. SAVE OUTPUT
# =========================
df.to_excel(r"C:\Users\salma\Downloads\MasterData_withCaseID.xlsx", index=False)

print(df[["Case_ID"]].head(10))
print("\nDone. File saved successfully.")