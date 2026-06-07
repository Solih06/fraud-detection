# Fraud Detection System - Adey Innovations

## Overview
This project aims to improve fraud detection capabilities for **Adey Innovations Inc.** by analyzing and preprocessing two distinct transaction datasets: e-commerce transactions and bank credit card transactions. 

The primary objective of this interim phase (Task 1) is to perform comprehensive data cleaning, exploratory data analysis (EDA), and feature engineering to prepare the data for future machine learning modeling.

## Project Structure
The repository is organized as follows:
* `data/raw/`: Contains the original `Fraud_Data.csv` and `IpAddress_to_Country.csv` datasets.
* `data/processed/`: Contains the cleaned and merged datasets prepared for modeling.
* `notebooks/`:
    * `eda-fraud-data.ipynb`: Data cleaning steps and exploratory data analysis, including class imbalance quantification.
    * `feature-engineering.ipynb`: Geolocation integration (IP-to-country mapping) and behavioral feature extraction.

## Key Accomplishments (Interim-1)
* **Data Cleaning:** Handled missing values, removed duplicates, and corrected data types for timestamps to ensure data integrity.
* **Exploratory Data Analysis:** Analyzed univariate and bivariate relationships and quantified the class imbalance between fraudulent and legitimate transactions.
* **Geolocation Enrichment:** Converted IP addresses to integers and utilized range-based merging to map transactions to specific countries.
* **Feature Engineering:** Created new behavioral features including `time_since_signup`, `hour_of_day`, `day_of_week`, and transaction velocity metrics to capture patterns indicative of fraudulent activity.

## How to Run
1. Ensure the required CSV files are placed in the `data/raw/` directory.
2. Install dependencies via `pip install -r requirements.txt`.
3. Execute the notebooks in the following order:
   - `eda-fraud-data.ipynb`
   - `feature-engineering.ipynb`