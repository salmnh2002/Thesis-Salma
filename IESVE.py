import os
import time
from itertools import product

import iesve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# INPUT
# =========================
folder = r"C:\Users\salma\Desktop\Thesis_Results"
period = ""

location_list = ["Roma"]   # ["Cuneo", "Milano", "Roma", "Napoli", "Palermo", "Lampedusa"]
floor = "T"                 # "G" = Ground, "M" = Middle, "T" = Top
position_on_floor = "M"     # "M" = middle, "C" = corner

ventilation_list = [2.4, 5.0, 8.0, 10.0]   # l/s/person
wfr_list = [0.12, 0.22]                     # kept only for results naming; you said you will change WFR manually
orient_list = [90, 0, 180, 270]            # East, North, South, West

wall_list = [1.1, 0.7, 0.32, 0.15]
roof_list = [1.35, 0.8, 0.32, 0.14]
window_u_list = [5.27, 2.8, 1.8, 1.5, 1.0]
shgc_list = [0.87, 0.65, 0.60, 0.55, 0.35]

window_combinations = list(zip(window_u_list, shgc_list))

window_infiltration_map = {
    5.27: "0.5 ACH",
    2.8: "0.5 ACH",
    1.8: "0.5 ACH",
    1.5: "0.25 ACH",
    1.0: "0.25 ACH"
}

wall_construction_map = {
    1.1: "WALL2",
    0.7: "WALL3",
    0.32: "WALL31",
    0.15: "WALL311"
}

roof_construction_map = {
    1.35: "ROOF",
    0.8: "ROOF2",
    0.32: "ROOF21",
    0.14: "ROOF141"
}

window_construction_map = {
    (5.27, 0.87): "EXTW1",
    (2.8, 0.65): "EXTW",
    (1.8, 0.60): "EXTW2",
    (1.5, 0.55): "EXTW21",
    (1.0, 0.35): "EXTW211"
}


# =========================
# HELPERS
# =========================
def find_construction_id_by_name(project_obj, target_name, construction_class):
    construction_ids = project_obj.get_construction_ids(construction_class)

    for cid in construction_ids:
        if str(cid).strip() == str(target_name).strip():
            return cid

    raise ValueError(f"Construction not found: {target_name}")


def select_body_with_openings(model):
    bodies = model.get_bodies(False)
    selected_body = None

    print("----- BODY CHECK -----")
    for i, b in enumerate(bodies):
        try:
            sfs = b.get_surfaces()
            total_openings = sum(len(sf.get_openings()) for sf in sfs)
            print("Body index:", i, "Body id:", b.id, "Total openings:", total_openings)

            if total_openings > 0 and selected_body is None:
                selected_body = b
        except Exception as e:
            print("Error reading body", i, ":", e)

    if selected_body is None:
        raise ValueError("No body with window openings was found in the model.")

    print("Selected body id:", selected_body.id)
    return selected_body


def get_window_and_roof_from_body(body):
    surfaces = body.get_surfaces()

    roof = None
    window = None
    surface_with_window = None

    print("----- SURFACE CHECK -----")
    for i, surface in enumerate(surfaces):
        props = surface.get_properties()
        tilt = props.get("tilt", None)
        openings = surface.get_openings()

        print(f"Surface {i}: tilt={tilt}, number_of_openings={len(openings)}")

        if tilt == 0 or tilt == 0.0:
            roof = surface

        if len(openings) > 0 and window is None:
            window = openings[0]
            surface_with_window = surface

    if window is None or surface_with_window is None:
        raise ValueError("A body was selected, but no window opening could be read from its surfaces.")

    return roof, window, surface_with_window


def tm52_status(percentC1, percentC2, percentC3):
    fail_count = 0
    if percentC1 > 3:
        fail_count += 1
    if percentC2 > 0:
        fail_count += 1
    if percentC3 > 0:
        fail_count += 1
    return "True" if fail_count >= 2 else "False"


# =========================
# TM52
# =========================
def exceeding_hours_TM52(f, climatic_zone, body_id):
    ext_temp = np.array(f.get_weather_results("Temperature", "Dry-bulb temperature"))

    alpha = 0.8
    days = 7
    daily_means = ext_temp.reshape(-1, 24).mean(axis=1)
    Trm = np.zeros_like(daily_means)

    for i in range(len(daily_means)):
        temps = daily_means[max(0, i - days):i][::-1]
        if len(temps) > 0:
            weighted = sum((1 - alpha) * (alpha ** n) * t for n, t in enumerate(temps))
            Trm[i] = weighted / (1 - alpha ** len(temps))
        else:
            Trm[i] = daily_means[i]

    Trm = np.repeat(Trm, 24)

    Tmin1 = 0.33 * Trm + 18.8 - 3
    Tmax1 = 0.33 * Trm + 18.8 + 2
    Tmax1_vent = Tmax1 + 2

    Tmin2 = 0.33 * Trm + 18.8 - 4
    Tmax2 = 0.33 * Trm + 18.8 + 3
    Tmax2_vent = Tmax2 + 2

    Tmin3 = 0.33 * Trm + 18.8 - 5
    Tmax3 = 0.33 * Trm + 18.8 + 4
    Tmax3_vent = Tmax3 + 2

    Tmax = 0.33 * Trm + 18.8 + 3
    Tmax_vent = Tmax + 2

    temp = np.array(
        f.get_room_results(
            body_id,
            "Room air temperature&Room radiant temperature",
            "Operative temperature (TM 52/CIBSE)",
            "z"
        )
    )

    date_index = pd.date_range("2025-01-01", periods=8760, freq="H")
    df = pd.DataFrame({
        "DateTime": date_index,
        "IndoorTemp": temp,
        "Threshold": Tmax,
        "ThresholdVent": Tmax_vent,
        "Tmin1": Tmin1,
        "Tmax1": Tmax1,
        "Tmax1Vent": Tmax1_vent,
        "Tmin2": Tmin2,
        "Tmax2": Tmax2,
        "Tmax2Vent": Tmax2_vent,
        "Tmin3": Tmin3,
        "Tmax3": Tmax3,
        "Tmax3Vent": Tmax3_vent
    })

    df["Month"] = df["DateTime"].dt.month
    df["Weekday"] = df["DateTime"].dt.weekday
    df["Hour"] = df["DateTime"].dt.hour

    occupied_months = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
    occupied_weekdays = [0, 1, 2, 3, 4]
    occupied_hours = range(8, 18)

    df["Occupied"] = (
        df["Month"].isin(occupied_months) &
        df["Weekday"].isin(occupied_weekdays) &
        df["Hour"].isin(occupied_hours)
    )

    if climatic_zone == "A":
        summer_mask = (df["DateTime"] >= "2025-03-16") & (df["DateTime"] < "2025-12-01")
    elif climatic_zone == "B":
        summer_mask = (df["DateTime"] >= "2025-04-01") & (df["DateTime"] < "2025-12-01")
    elif climatic_zone == "C":
        summer_mask = (df["DateTime"] >= "2025-04-01") & (df["DateTime"] < "2025-11-15")
    elif climatic_zone == "D":
        summer_mask = (df["DateTime"] >= "2025-04-16") & (df["DateTime"] < "2025-11-01")
    elif climatic_zone == "E":
        summer_mask = (df["DateTime"] >= "2025-04-16") & (df["DateTime"] < "2025-10-15")
    elif climatic_zone == "F":
        summer_mask = (df["DateTime"] >= "2025-04-16") & (df["DateTime"] < "2025-10-15")
    else:
        summer_mask = pd.Series([True] * len(df), index=df.index)

    df_summer = df.loc[summer_mask].copy()

    df_summer["Exceeds"] = (df_summer["IndoorTemp"] > (df_summer["Threshold"] + 1)) & df_summer["Occupied"]
    df_summer["ExceedsVent"] = (df_summer["IndoorTemp"] > (df_summer["ThresholdVent"] + 1)) & df_summer["Occupied"]

    occupied_count = df_summer["Occupied"].sum()
    exceed_count = df_summer["Exceeds"].sum()
    exceed_vent_count = df_summer["ExceedsVent"].sum()

    percentC1 = exceed_count / occupied_count * 100 if occupied_count > 0 else 0
    percentC1Vent = exceed_vent_count / occupied_count * 100 if occupied_count > 0 else 0

    df_occ = df_summer[df_summer["Occupied"]].copy()

    df_occ["C2"] = np.where(
        df_occ["IndoorTemp"] > df_occ["Threshold"],
        df_occ["IndoorTemp"] - df_occ["Threshold"],
        0
    )
    dailyDH = df_occ.groupby(df_occ["DateTime"].dt.date)["C2"].sum()
    days_exceeded_C2 = (dailyDH > 6).sum()
    tot_days = len(dailyDH)
    percentC2 = days_exceeded_C2 / tot_days * 100 if tot_days > 0 else 0

    df_occ["C2Vent"] = np.where(
        df_occ["IndoorTemp"] > df_occ["ThresholdVent"],
        df_occ["IndoorTemp"] - df_occ["ThresholdVent"],
        0
    )
    dailyDHVent = df_occ.groupby(df_occ["DateTime"].dt.date)["C2Vent"].sum()
    days_exceeded_C2Vent = (dailyDHVent > 6).sum()
    tot_daysVent = len(dailyDHVent)
    percentC2Vent = days_exceeded_C2Vent / tot_daysVent * 100 if tot_daysVent > 0 else 0

    C3_exceed_hours = ((df_summer["IndoorTemp"] > df_summer["Threshold"] + 4) & df_summer["Occupied"]).sum()
    percentC3 = C3_exceed_hours / occupied_count * 100 if occupied_count > 0 else 0

    C3_exceed_hoursVent = ((df_summer["IndoorTemp"] > df_summer["ThresholdVent"] + 4) & df_summer["Occupied"]).sum()
    percentC3Vent = C3_exceed_hoursVent / occupied_count * 100 if occupied_count > 0 else 0

    overheating = tm52_status(percentC1, percentC2, percentC3)
    overheatingVent = tm52_status(percentC1Vent, percentC2Vent, percentC3Vent)

    dist = df_occ["C2"]
    distVent = df_occ["C2Vent"]
    dist_df = pd.DataFrame({"without_vent": dist, "with_vent": distVent})

    df_summer["low"] = (df_summer["IndoorTemp"] < df_summer["Tmin3"]) & df_summer["Occupied"]
    df_summer["lowBand3"] = (df_summer["IndoorTemp"] > df_summer["Tmin3"]) & (df_summer["IndoorTemp"] < df_summer["Tmin2"]) & df_summer["Occupied"]
    df_summer["lowBand2"] = (df_summer["IndoorTemp"] > df_summer["Tmin2"]) & (df_summer["IndoorTemp"] < df_summer["Tmin1"]) & df_summer["Occupied"]
    df_summer["Band1"] = (df_summer["IndoorTemp"] > df_summer["Tmin1"]) & (df_summer["IndoorTemp"] < df_summer["Tmax1"]) & df_summer["Occupied"]
    df_summer["highBand2"] = (df_summer["IndoorTemp"] > df_summer["Tmax1"]) & (df_summer["IndoorTemp"] < df_summer["Tmax2"]) & df_summer["Occupied"]
    df_summer["highBand3"] = (df_summer["IndoorTemp"] > df_summer["Tmax2"]) & (df_summer["IndoorTemp"] < df_summer["Tmax3"]) & df_summer["Occupied"]
    df_summer["high"] = (df_summer["IndoorTemp"] > df_summer["Tmax3"]) & df_summer["Occupied"]

    df_summer["lowVent"] = (df_summer["IndoorTemp"] < df_summer["Tmin3"]) & df_summer["Occupied"]
    df_summer["lowBand3Vent"] = (df_summer["IndoorTemp"] > df_summer["Tmin3"]) & (df_summer["IndoorTemp"] < df_summer["Tmin2"]) & df_summer["Occupied"]
    df_summer["lowBand2Vent"] = (df_summer["IndoorTemp"] > df_summer["Tmin2"]) & (df_summer["IndoorTemp"] < df_summer["Tmin1"]) & df_summer["Occupied"]
    df_summer["Band1Vent"] = (df_summer["IndoorTemp"] > df_summer["Tmin1"]) & (df_summer["IndoorTemp"] < df_summer["Tmax1Vent"]) & df_summer["Occupied"]
    df_summer["highBand2Vent"] = (df_summer["IndoorTemp"] > df_summer["Tmax1Vent"]) & (df_summer["IndoorTemp"] < df_summer["Tmax2Vent"]) & df_summer["Occupied"]
    df_summer["highBand3Vent"] = (df_summer["IndoorTemp"] > df_summer["Tmax2Vent"]) & (df_summer["IndoorTemp"] < df_summer["Tmax3Vent"]) & df_summer["Occupied"]
    df_summer["highVent"] = (df_summer["IndoorTemp"] > df_summer["Tmax3Vent"]) & df_summer["Occupied"]

    low = df_summer["low"].sum() / occupied_count * 100 if occupied_count > 0 else 0
    lowBand3 = df_summer["lowBand3"].sum() / occupied_count * 100 if occupied_count > 0 else 0
    lowBand2 = df_summer["lowBand2"].sum() / occupied_count * 100 if occupied_count > 0 else 0
    Band1 = df_summer["Band1"].sum() / occupied_count * 100 if occupied_count > 0 else 0
    highBand2 = df_summer["highBand2"].sum() / occupied_count * 100 if occupied_count > 0 else 0
    highBand3 = df_summer["highBand3"].sum() / occupied_count * 100 if occupied_count > 0 else 0
    high = df_summer["high"].sum() / occupied_count * 100 if occupied_count > 0 else 0

    lowVent = df_summer["lowVent"].sum() / occupied_count * 100 if occupied_count > 0 else 0
    lowBand3Vent = df_summer["lowBand3Vent"].sum() / occupied_count * 100 if occupied_count > 0 else 0
    lowBand2Vent = df_summer["lowBand2Vent"].sum() / occupied_count * 100 if occupied_count > 0 else 0
    Band1Vent = df_summer["Band1Vent"].sum() / occupied_count * 100 if occupied_count > 0 else 0
    highBand2Vent = df_summer["highBand2Vent"].sum() / occupied_count * 100 if occupied_count > 0 else 0
    highBand3Vent = df_summer["highBand3Vent"].sum() / occupied_count * 100 if occupied_count > 0 else 0
    highVent = df_summer["highVent"].sum() / occupied_count * 100 if occupied_count > 0 else 0

    return (
        temp, Trm, Tmax,
        percentC1, percentC2, percentC3, overheating,
        Tmax_vent, percentC1Vent, percentC2Vent, percentC3Vent, overheatingVent,
        dist_df,
        low, lowBand3, lowBand2, Band1, highBand2, highBand3, high,
        lowVent, lowBand3Vent, lowBand2Vent, Band1Vent, highBand2Vent, highBand3Vent, highVent
    )


# =========================
# LOCATION DATA
# =========================
loc_dict1 = {'altitude': 534, 'latitude': 44.39, 'longitude': 7.54,
             'city': 'Cuneo', 'country': 'Italy',
             'time_zone': 1.0, 'dst_correction': 0, 'dst_from_month': iesve.month.april, 'dst_to_month': iesve.month.october,
             'cooling_loads_percentile': 0.4, 'heating_loads_percentile': 99.6,
             'external_CO2': 400, 'ref_air_density': 1.2, 'winter_drybulb': -4.65,
             'ground_reflectance_summer': 0.2, 'ground_reflectance_summer_from_month': iesve.month.april,
             'ground_reflectance_summer_to_month': iesve.month.october, 'ground_reflectance_winter': 0.2,
             'weather_file': "ITA_PM_Cuneo-Levaldigi.AP.161170_TMYx.2009-2023.epw"}

loc_dict2 = {'altitude': 108, 'latitude': 45.44, 'longitude': 9.28,
             'city': 'Milano Linate', 'country': 'Italy',
             'time_zone': 1.0, 'dst_correction': 0, 'dst_from_month': iesve.month.april, 'dst_to_month': iesve.month.october,
             'cooling_loads_percentile': 0.4, 'heating_loads_percentile': 99.6,
             'external_CO2': 400, 'ref_air_density': 1.2, 'winter_drybulb': -4.65,
             'ground_reflectance_summer': 0.2, 'ground_reflectance_summer_from_month': iesve.month.april,
             'ground_reflectance_summer_to_month': iesve.month.october, 'ground_reflectance_winter': 0.2,
             'weather_file': "ITA_LM_Milano-Linate.AP.160800_TMYx.2009-2023.epw"}

loc_dict3 = {'altitude': 24, 'latitude': 41.95, 'longitude': 12.5,
             'city': 'Roma Urbe', 'country': 'Italy',
             'time_zone': 1.0, 'dst_correction': 0, 'dst_from_month': iesve.month.april, 'dst_to_month': iesve.month.october,
             'cooling_loads_percentile': 0.4, 'heating_loads_percentile': 99.6,
             'external_CO2': 400, 'ref_air_density': 1.2, 'winter_drybulb': -4.65,
             'ground_reflectance_summer': 0.2, 'ground_reflectance_summer_from_month': iesve.month.april,
             'ground_reflectance_summer_to_month': iesve.month.october, 'ground_reflectance_winter': 0.2,
             'weather_file': "ITA_LZ_Roma.Urbe.Rgnl.AP.162350_TMYx.2009-2023.epw"}

loc_dict4 = {'altitude': 72, 'latitude': 40.88, 'longitude': 14.29,
             'city': 'Napoli Capodichino', 'country': 'Italy',
             'time_zone': 1.0, 'dst_correction': 0, 'dst_from_month': iesve.month.april, 'dst_to_month': iesve.month.october,
             'cooling_loads_percentile': 0.4, 'heating_loads_percentile': 99.6,
             'external_CO2': 400, 'ref_air_density': 1.2, 'winter_drybulb': -4.65,
             'ground_reflectance_summer': 0.2, 'ground_reflectance_summer_from_month': iesve.month.april,
             'ground_reflectance_summer_to_month': iesve.month.october, 'ground_reflectance_winter': 0.2,
             'weather_file': "ITA_CM_Napoli-Capodichino.AP.162890_TMYx.2009-2023.epw"}

loc_dict5 = {'altitude': 34, 'latitude': 38.18, 'longitude': 13.10,
             'city': 'Palermo Punta Raisi', 'country': 'Italy',
             'time_zone': 1.0, 'dst_correction': 0, 'dst_from_month': iesve.month.april, 'dst_to_month': iesve.month.october,
             'cooling_loads_percentile': 0.4, 'heating_loads_percentile': 99.6,
             'external_CO2': 400, 'ref_air_density': 1.2, 'winter_drybulb': 6.20,
             'ground_reflectance_summer': 0.2, 'ground_reflectance_summer_from_month': iesve.month.april,
             'ground_reflectance_summer_to_month': iesve.month.october, 'ground_reflectance_winter': 0.2,
             'weather_file': "ITA_SC_Palermo-Falcone.Borsellino.AP.164050_TMYx.2009-2023.epw"}

loc_dict6 = {'altitude': 21, 'latitude': 35.50, 'longitude': 12.62,
             'city': 'Lampedusa', 'country': 'Italy',
             'time_zone': 1.0, 'dst_correction': 0, 'dst_from_month': iesve.month.april, 'dst_to_month': iesve.month.october,
             'cooling_loads_percentile': 0.4, 'heating_loads_percentile': 99.6,
             'external_CO2': 400, 'ref_air_density': 1.2, 'winter_drybulb': -4.65,
             'ground_reflectance_summer': 0.2, 'ground_reflectance_summer_from_month': iesve.month.april,
             'ground_reflectance_summer_to_month': iesve.month.october, 'ground_reflectance_winter': 0.2,
             'weather_file': "ITA_SC_Lampedusa.AP.164900_TMYx.2009-2023.epw"}


# =========================
# SCENARIOS
# =========================
scenarios_df = pd.DataFrame(
    product(
        ventilation_list,
        wfr_list,
        orient_list,
        wall_list,
        roof_list,
        window_combinations
    ),
    columns=[
        "ventilation_lps_person",
        "window_to_floor_ratio",
        "building_orientation",
        "wall_u_value",
        "roof_u_value",
        "window_combo"
    ]
)

scenarios_df[["window_u_value", "solar_heat_gain_coefficient"]] = pd.DataFrame(
    scenarios_df["window_combo"].tolist(),
    index=scenarios_df.index
)
scenarios_df.drop(columns=["window_combo"], inplace=True)

scenarios_df["floor"] = floor
scenarios_df["position_on_floor"] = position_on_floor


# =========================
# PREP OUTPUTS
# =========================
os.makedirs(folder, exist_ok=True)

temp_df = pd.DataFrame(index=range(0, 8760))
dist_df = pd.DataFrame()

scenarios_df_results = pd.DataFrame(columns=scenarios_df.columns)
extra_cols = [
    "C1", "C2", "C3", "Overheating",
    "C1_VENT", "C2_VENT", "C3_VENT", "Overheating_VENT",
    "low", "lowBand3", "lowBand2", "Band1", "highBand2", "highBand3", "high",
    "lowVent", "lowBand3Vent", "lowBand2Vent", "Band1Vent", "highBand2Vent", "highBand3Vent", "highVent"
]
for col in extra_cols:
    scenarios_df_results[col] = ""

n_sim = scenarios_df.shape[0]
print("Number of scenarios:", n_sim)
print("More or less 11 seconds per simulation -->", n_sim * 11 / 60, "minutes")


# =========================
# SIMULATION LOOP
# =========================
for location in location_list:
    if location == "Cuneo":
        loc_dict = loc_dict1
        specific = "-CU"
        aps_name = "Shoebox3-CU.aps"
        climatic_zone = "F"
    elif location == "Milano":
        loc_dict = loc_dict2
        specific = "-MI"
        aps_name = "Shoebox3-MI.aps"
        climatic_zone = "E"
    elif location == "Roma":
        loc_dict = loc_dict3
        specific = "-RO"
        aps_name = "Shoebox3-RO.aps"
        climatic_zone = "D"
    elif location == "Napoli":
        loc_dict = loc_dict4
        specific = "-NA"
        aps_name = "Shoebox3-NA.aps"
        climatic_zone = "C"
    elif location == "Palermo":
        loc_dict = loc_dict5
        specific = "-PA"
        aps_name = "Shoebox3-PA.aps"
        climatic_zone = "B"
    elif location == "Lampedusa":
        loc_dict = loc_dict6
        specific = "-LA"
        aps_name = "Shoebox3-LA.aps"
        climatic_zone = "A"
    else:
        raise ValueError(f"Unsupported location: {location}")

    print(f"Start of the simulations for {location}")

    project = iesve.VEProject.get_current_project()
    model = project.models[0]
    body = select_body_with_openings(model)

    templates = project.thermal_templates(True)
    template = templates[1]

    air_exchanges = project.air_exchanges()
    infSingle = air_exchanges[0]
    ventBase = air_exchanges[1]
    infDouble = air_exchanges[2]
    ventACH = air_exchanges[3]
    ventACH2 = air_exchanges[8]
    ventACH3 = air_exchanges[10]

    db = iesve.VECdbDatabase.get_current_database()
    projects_for_ids = db.get_projects()
    project_list_for_ids = projects_for_ids[0]
    project_for_ids = project_list_for_ids[0]

    for line in range(0, n_sim):
        vent = scenarios_df.at[line, "ventilation_lps_person"]
        wfr = scenarios_df.at[line, "window_to_floor_ratio"]
        orientation = scenarios_df.at[line, "building_orientation"]
        wall_u = scenarios_df.at[line, "wall_u_value"]
        roof_u = scenarios_df.at[line, "roof_u_value"]
        window_u = scenarios_df.at[line, "window_u_value"]
        g_value = scenarios_df.at[line, "solar_heat_gain_coefficient"]

        print(
            "Simulation start for case {} - vent {} l/s/person, WFR {}, orientation {}, wall U {}, roof U {}, window U {}, SHGC {}".format(
                line + 1, vent, wfr, orientation, wall_u, roof_u, window_u, g_value
            )
        )

        if vent == 2.4:
            template.add_air_exchange(ventBase)
            try:
                template.remove_air_exchange(ventACH)
            except Exception:
                print("No change")
        elif vent == 5.0:
            template.add_air_exchange(ventACH)
            try:
                template.remove_air_exchange(ventBase)
            except Exception:
                print("No change")
        elif vent == 8.0:
            template.add_air_exchange(ventACH2)
            try:
                template.remove_air_exchange(ventBase)
            except Exception:
                print("No change")
        elif vent == 10.0:
            template.add_air_exchange(ventACH3)
            try:
                template.remove_air_exchange(ventBase)
            except Exception:
                print("No change")

        infiltration = window_infiltration_map[window_u]

        if infiltration == "0.5 ACH":
            template.add_air_exchange(infSingle)
            try:
                template.remove_air_exchange(infDouble)
            except Exception:
                print("No change")
        elif infiltration == "0.25 ACH":
            template.add_air_exchange(infDouble)
            try:
                template.remove_air_exchange(infSingle)
            except Exception:
                print("No change")

        wall_construction_name = wall_construction_map[wall_u]
        roof_construction_name = roof_construction_map[roof_u]
        window_construction_name = window_construction_map[(window_u, g_value)]

        print("Wall construction name:", wall_construction_name)
        print("Roof construction name:", roof_construction_name)
        print("Window construction name:", window_construction_name)

        idWall = find_construction_id_by_name(project_for_ids, wall_construction_name, iesve.construction_class.opaque)
        idRoof = find_construction_id_by_name(project_for_ids, roof_construction_name, iesve.construction_class.opaque)
        idWindow = find_construction_id_by_name(project_for_ids, window_construction_name, iesve.construction_class.glazed)

        body.select()
        geom = iesve.VEGeometry
        geom.set_building_orientation(int(orientation))

        roof, window, surface_with_window = get_window_and_roof_from_body(body)

        constructionWindow = project_for_ids.get_construction(idWindow, iesve.construction_class.glazed)
        body.assign_construction_to_opening(constructionWindow, surface_with_window, window.get_id())

        constructionWall = project_for_ids.get_construction(idWall, iesve.construction_class.opaque)
        constructionRoof = project_for_ids.get_construction(idRoof, iesve.construction_class.opaque)

        surfaces = body.get_surfaces()
        for i, surface in enumerate(surfaces):
            props = surface.get_properties()
            tilt = props.get("tilt", None)

            if tilt == 90.0 or tilt == 90:
                try:
                    body.assign_construction(constructionWall, surface)
                    print(f"Wall construction assigned to surface {i}")
                except Exception as e:
                    print(f"Skipped surface {i}: {e}")

        if floor == "T" and roof is not None:
            try:
                body.assign_construction(constructionRoof, roof)
                print("Roof construction assigned")
            except Exception as e:
                print("Roof assignment skipped:", e)

        suncast = model.suncast()
        suncast.run(1, 12, 15, True)
        print("Suncast done")

        time.sleep(3)
        print("Ready")

        s = iesve.ApacheSim()
        s.set_options(results_filename=aps_name, suncast=True)
        s.run_simulation()

        f = iesve.ResultsReader()
        f.open_aps_data(aps_name)

        (
            temp, Trm, Tmax,
            percentC1, percentC2, percentC3, overheating,
            Tmax_vent, percentC1Vent, percentC2Vent, percentC3Vent, overheatingVent,
            dist, low, lowBand3, lowBand2, Band1, highBand2, highBand3, high,
            lowVent, lowBand3Vent, lowBand2Vent, Band1Vent, highBand2Vent, highBand3Vent, highVent
        ) = exceeding_hours_TM52(f, climatic_zone, body.id)

        add_after = scenarios_df_results.shape[0]

        scenarios_df_results.at[add_after, "ventilation_lps_person"] = vent
        scenarios_df_results.at[add_after, "window_to_floor_ratio"] = wfr
        scenarios_df_results.at[add_after, "building_orientation"] = orientation
        scenarios_df_results.at[add_after, "wall_u_value"] = wall_u
        scenarios_df_results.at[add_after, "roof_u_value"] = roof_u
        scenarios_df_results.at[add_after, "window_u_value"] = window_u
        scenarios_df_results.at[add_after, "solar_heat_gain_coefficient"] = g_value
        scenarios_df_results.at[add_after, "floor"] = floor
        scenarios_df_results.at[add_after, "position_on_floor"] = position_on_floor

        scenarios_df_results.at[add_after, "C1"] = percentC1
        scenarios_df_results.at[add_after, "C2"] = percentC2
        scenarios_df_results.at[add_after, "C3"] = percentC3
        scenarios_df_results.at[add_after, "Overheating"] = overheating
        scenarios_df_results.at[add_after, "C1_VENT"] = percentC1Vent
        scenarios_df_results.at[add_after, "C2_VENT"] = percentC2Vent
        scenarios_df_results.at[add_after, "C3_VENT"] = percentC3Vent
        scenarios_df_results.at[add_after, "Overheating_VENT"] = overheatingVent

        scenarios_df_results.at[add_after, "low"] = low
        scenarios_df_results.at[add_after, "lowBand3"] = lowBand3
        scenarios_df_results.at[add_after, "lowBand2"] = lowBand2
        scenarios_df_results.at[add_after, "Band1"] = Band1
        scenarios_df_results.at[add_after, "highBand2"] = highBand2
        scenarios_df_results.at[add_after, "highBand3"] = highBand3
        scenarios_df_results.at[add_after, "high"] = high
        scenarios_df_results.at[add_after, "lowVent"] = lowVent
        scenarios_df_results.at[add_after, "lowBand3Vent"] = lowBand3Vent
        scenarios_df_results.at[add_after, "lowBand2Vent"] = lowBand2Vent
        scenarios_df_results.at[add_after, "Band1Vent"] = Band1Vent
        scenarios_df_results.at[add_after, "highBand2Vent"] = highBand2Vent
        scenarios_df_results.at[add_after, "highBand3Vent"] = highBand3Vent
        scenarios_df_results.at[add_after, "highVent"] = highVent

        col_name = "{}, floor {}, pos {}, vent {}, WFR {}, orient {}, wallU {}, roofU {}, winU {}, g {}".format(
            specific, floor, position_on_floor, vent, wfr, orientation, wall_u, roof_u, window_u, g_value
        )
        temp_df[col_name] = temp
        print("Simulation", line + 1, "done")

        bins = np.arange(0, 11, 1).tolist()

        counts1, _, _ = plt.hist(
            dist["without_vent"],
            weights=np.ones(len(dist["without_vent"])) / len(dist["without_vent"]) if len(dist["without_vent"]) > 0 else None,
            bins=bins,
            cumulative=True
        )
        plt.clf()

        counts2, _, _ = plt.hist(
            dist["with_vent"],
            weights=np.ones(len(dist["with_vent"])) / len(dist["with_vent"]) if len(dist["with_vent"]) > 0 else None,
            bins=bins,
            cumulative=True
        )
        plt.clf()

        dist_df["bins"] = np.arange(0, 10, 1).tolist()
        dist_df[col_name] = counts1
        dist_df[col_name + " VENT"] = counts2

    dist_df = dist_df.transpose()

    temp_df = temp_df.copy()
    temp_df["Trm" + specific] = Trm
    temp_df["Tmax" + specific] = Tmax
    temp_df["TmaxVent" + specific] = Tmax_vent
    temp_df["Tout" + specific] = f.get_weather_results("Temperature", "Dry-bulb temperature")

    specifications = specific + "-" + floor + "-" + position_on_floor + period + ".xlsx"

    scenarios_path = os.path.join(folder, "Scenarios" + specifications)
    temperatures_path = os.path.join(folder, "Temperatures" + specifications)
    dist_path = os.path.join(folder, "Dist" + specifications)

    scenarios_df_results.to_excel(scenarios_path, index=False)
    temp_df.to_excel(temperatures_path, index=False)
    dist_df.to_excel(dist_path)

print("Finished")
