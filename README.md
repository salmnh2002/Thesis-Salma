# Thesis-Salma

**A Machine Learning Framework for Identifying Climate-Adaptive Envelope Retrofit Strategies to Improve Thermal Comfort and Reduce Overheating in Free-Floating School Buildings**

This repository contains the Python scripts and datasets used for my master’s thesis. The workflow moves from IESVE simulation results through data preparation and exploratory analysis to machine learning.

## Workflow

1. **`IESVE.py`** — IESVE simulation and initial results.
2. **`DataCleaning.py`** — Cleans and prepares the simulation data.
3. **`DataClustering.py`** — Uses `Master Temperature_Summary.xlsx` and produces `Master Data.xlsx`.
4. **`EDA.py`** — Uses `Master Data.xlsx` for exploratory data analysis and visualizations.
5. **Machine learning dataset preparation** — After reviewing the EDA results and discussing the findings, parameters are selected or excluded. The selected input parameters (X) and target variable (y) are combined in `ML Data.xlsx`.
6. **`MLFramework.py`** — Uses `ML Data.xlsx` to train and evaluate machine learning models for assessing overheating and envelope retrofit strategies.

## Data files

| File                              | Role                                                    |
| --------------------------------- | ------------------------------------------------------- |
| `Master Temperature_Summary.xlsx` | Input to `DataClustering.py`                            |
| `Master Data.xlsx`                | Output of `DataClustering.py` and input to `EDA.py`     |
| `ML Data.xlsx`                    | Dataset prepared after EDA and used by `MLFramework.py` |

The machine learning target is degree-hours above 26°C during school-occupied hours.
