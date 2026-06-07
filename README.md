# Fraud Detection System - Adey Innovations

## Overview
This project aims to improve fraud detection capabilities for **Adey Innovations Inc.** by analyzing and preprocessing two distinct transaction datasets: e-commerce transactions and bank credit card transactions. 

The primary objective of this interim phase (Task 1) is to perform comprehensive data cleaning, exploratory data analysis (EDA), and feature engineering to prepare the data for future machine learning modeling.

## Project Structure
# Fraud Detection Project

This project implements a modular, scalable, and reproducible machine learning pipeline for fraud detection. The system follows industry best practices by separating data processing logic into reusable modules and utilizing `scikit-learn` pipelines for data transformation.

## Project Structure

The project is organized into modular components to ensure clean code and easy maintenance:

```text
fraud-detection/
├── data/                   # Dataset storage (raw and processed)
├── notebooks/              # Jupyter notebooks for analysis and orchestration
│   ├── eda-fraud-data.ipynb           # Exploratory Data Analysis
│   ├── feature-engineering.ipynb         # Geolocation integration (IP-to-country mapping) and behavioral feature extraction.
│   └── modeling_preparation.ipynb  # Pipeline orchestration
├── src/                    # Core source code
│   ├── __init__.py
│   ├── cleaning.py         # Data loading and cleaning functions
│   ├── features.py         # Feature engineering logic
│   ├── transformation.py   # Preprocessing and encoding pipeline
│   └── sampling.py         # Data resampling (SMOTE) logic
├── requirements.txt        # Project dependencies
├── run_pipeline_test.py    # Automated pipeline verification script
└── README.md
```
## Key Accomplishments (Interim-1)
* **Data Cleaning:** Handled missing values, removed duplicates, and corrected data types for timestamps to ensure data integrity.
* **Exploratory Data Analysis:** Analyzed univariate and bivariate relationships and quantified the class imbalance between fraudulent and legitimate transactions.
* **Geolocation Enrichment:** Converted IP addresses to integers and utilized range-based merging to map transactions to specific countries.
* **Feature Engineering:** Created new behavioral features including `time_since_signup`, `hour_of_day`, `day_of_week`, and transaction velocity metrics to capture patterns indicative of fraudulent activity.

## Exploratory Data Analysis (EDA)
Below is the distribution of transactions, highlighting the class imbalance between fraudulent and legitimate activities.

![Class Distribution](visuals/fraud_class_distribution.png)

## How to Run
1. Ensure the required CSV files are placed in the `data/raw/` directory.
2. Install dependencies via `pip install -r requirements.txt`.
3. Execute the notebooks in the following order:
   - `eda-fraud-data.ipynb`
   - `feature-engineering.ipynb`