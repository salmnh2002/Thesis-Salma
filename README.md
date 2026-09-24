# Thesis-Salma

**A Machine Learning Framework for Identifying Climate-Adaptive Envelope Retrofit Strategies to Improve Thermal Comfort and Reduce Overheating in Free-Floating School Buildings**

This repository contains the Python scripts used for my master’s thesis. The workflow starts with building simulation results from IESVE and continues through data preparation, analysis, and machine learning.

## Workflow

The files follow this order:

1. **`IESVE.py`** — IESVE simulation and initial results.
2. **`DataCleaning.py`** — Cleaning and preparing the simulation data.
3. **`DataClustering.py`** — Clustering analysis of the prepared data.
4. **`EDA.py`** — Exploratory data analysis and visualizations.
5. **`MachineLearningFramework.py`** — Training and evaluating machine learning models to assess overheating and envelope retrofit strategies.

The main prediction target is degree-hours above 26°C during school-occupied hours.
