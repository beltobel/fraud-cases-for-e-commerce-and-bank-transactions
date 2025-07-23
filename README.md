# detection of fraud cases for e-commerce and bank transactions

## Overview

This project involves the preprocessing and exploratory data analysis (EDA) of two datasets, Fraud_Data.csv and creditcard.csv, to prepare them for fraud detection modeling. The tasks include data cleaning, feature engineering, handling class imbalance, and scaling features, with the goal of ensuring data quality and suitability for machine learning. This README provides an overview of the datasets, preprocessing steps, and instructions for using the preprocessed data.
Datasets

### Fraud_Data.csv:

**Description**: Contains user transaction data for fraud detection.
**Features**: user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address, class (target variable: 0 for non-fraud, 1 for fraud).
Size: 151,112 records.
**Key Insights**: Approximately 9.36% of transactions are fraudulent, indicating class imbalance.

### creditcard.csv:

**Description**: Contains anonymized credit card transaction data with PCA-transformed features.
**Features**: Time, Amount, V1 to V28 (PCA features), Class (target variable: 0 for non-fraud, 1 for fraud).
**Key Insights**: Highly imbalanced dataset with a fraud rate typically below 1%.

### IpAddress_to_Country.csv:

**Description**: Maps IP addresses to countries for enriching Fraud_Data.csv.

### Preprocessing Steps

The preprocessing and EDA were performed using Python 3.13.1 in a Jupyter Notebook (eda.ipynb). Key steps include:

**Data Cleaning:**

Checked for and confirmed no missing values in both datasets.
Removed duplicates using pandas.drop_duplicates().
**Corrected data types:**
Fraud_Data.csv: Converted signup_time and purchase_time to datetime, user_id and device_id to strings, ip_address to float.
creditcard.csv: Ensured Time and Amount as float, Class as integer.

**Feature Engineering:**

For Fraud_Data.csv:<br>
Created features: transaction_frequency, transaction_velocity, hour_of_day, day_of_week, time_since_signup.
Mapped IP addresses to countries using IpAddress_to_Country.csv.

For creditcard.csv: No additional feature engineering due to PCA features.

**Exploratory Data Analysis:**

Generated summary statistics, histograms, boxplots, and correlation heatmaps.
Identified class imbalance in Fraud_Data.csv (9.36% fraud) and creditcard.csv (<1% fraud).
Noted potential outliers in purchase_value and age for Fraud_Data.csv.

**Preprocessing for Modeling:**

**Feature Selection**: Used numerical (purchase_value, age, etc.) and categorical (source, browser, sex, country) features for Fraud_Data.csv; all features except Class for creditcard.csv.
**Encoding**: Applied OneHotEncoder to categorical features in Fraud_Data.csv.
**Imputation**: Used SimpleImputer with median for numerical features and most frequent value for categorical features.
**Class Imbalance**: Applied SMOTE to Fraud_Data.csv, achieving a 50:50 class distribution.
**Scaling**: Standardized features using StandardScaler.
**Data Splitting**: Split Fraud_Data.csv into 80% training and 20% testing sets with stratification.
**Output**: Saved preprocessed arrays as .npy files in the ../data/processed/ directory.

**Environment:**

Python libraries: pandas (2.2.3), numpy (2.2.0), scikit-learn (1.6.0), imblearn (0.13.0), matplotlib (3.9.3), seaborn (0.13.2).


## Model Training:

Trained a Random Forest Classifier (n_estimators=100, class_weight='balanced', random_state=42) on both datasets.

### Evaluation:





**Metrics**: AUC-PR, F1-score, confusion matrix, and classification report.

**Results**:


**Fraud_Data.csv:**

AUC-PR: 0.9270

F1-score: 0.8714

Confusion Matrix: [[26386, 1006], [4348, 18132]]

Accuracy: 0.89

**Classification Report:** Class 0 (Precision: 0.86, Recall: 0.96, F1: 0.91); Class 1 (Precision: 0.95, Recall: 0.81, F1: 0.87).

**creditcard.csv:**

AUC-PR: 0.0029

F1-score: 0.0000

Confusion Matrix: [[56644, 4], [98, 0]]

Accuracy: 1.00 (due to extreme imbalance)

**Classification Report**: Class 0 (Precision: 1.00, Recall: 1.00, F1: 1.00); Class 1 (Precision: 0.00, Recall: 0.00, F1: 0.00).

**Files**

eda.ipynb: handles EDA

preprocess_for_fraud_model_training.py: Handles data loading, cleaning, feature engineering, and preprocessing.

fraud_model_training.ipynb: Implements model training and evaluation.