import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

np.random.seed(42)

# 1. Load Datasets
fraud_data = pd.read_csv('../data/Fraud_Data.csv')
ip_to_country = pd.read_csv('../data/IpAddress_to_Country.csv')
creditcard_data = pd.read_csv('../data/creditcard.csv')

# 2. Handle Missing Values
fraud_data['purchase_value'].fillna(fraud_data['purchase_value'].median(), inplace=True)
fraud_data['age'].fillna(fraud_data['age'].median(), inplace=True)
fraud_data.dropna(subset=['source', 'browser', 'sex'], inplace=True)
creditcard_data.fillna(creditcard_data.median(numeric_only=True), inplace=True)

# 3. Data Cleaning
fraud_data.drop_duplicates(inplace=True)
creditcard_data.drop_duplicates(inplace=True)
fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
fraud_data['user_id'] = fraud_data['user_id'].astype(str)
fraud_data['device_id'] = fraud_data['device_id'].astype(str)
fraud_data['ip_address'] = fraud_data['ip_address'].astype(float)
fraud_data['class'] = fraud_data['class'].astype(int)
creditcard_data['Time'] = creditcard_data['Time'].astype(float)
creditcard_data['Amount'] = creditcard_data['Amount'].astype(float)
creditcard_data['Class'] = creditcard_data['Class'].astype(int)

# 4. Merge Datasets for Geolocation Analysis (vectorized)
fraud_data['ip_address_int'] = fraud_data['ip_address'].astype(int)
ip_to_country['lower_bound_ip_address'] = ip_to_country['lower_bound_ip_address'].astype(int)
ip_to_country['upper_bound_ip_address'] = ip_to_country['upper_bound_ip_address'].astype(int)

def fast_ip_country(ip):
    match = ip_to_country[(ip_to_country['lower_bound_ip_address'] <= ip) & (ip_to_country['upper_bound_ip_address'] >= ip)]
    return match['country'].values[0] if not match.empty else 'Unknown'
fraud_data['country'] = fraud_data['ip_address_int'].map(fast_ip_country)

# 5. Feature Engineering
fraud_data['transaction_frequency'] = fraud_data.groupby('user_id')['user_id'].transform('count')
fraud_data.sort_values(['user_id', 'purchase_time'], inplace=True)
fraud_data['time_diff'] = fraud_data.groupby('user_id')['purchase_time'].diff().dt.total_seconds()
fraud_data['transaction_velocity'] = fraud_data.groupby('user_id')['time_diff'].transform('mean')
fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600

# 6. Data Transformation
features = ['purchase_value', 'age', 'transaction_frequency', 'transaction_velocity', 'hour_of_day', 'day_of_week', 'time_since_signup']
X = fraud_data[features]
y = fraud_data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Fill any remaining NaNs in train and test sets (diagnostic print)
print("NaNs in X_train before fillna:\n", X_train.isnull().sum())
print("NaNs in X_test before fillna:\n", X_test.isnull().sum())

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print("NaNs in X_train after fillna:\n", X_train.isnull().sum())
print("NaNs in X_test after fillna:\n", X_test.isnull().sum())

# Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Scaling
scaler = StandardScaler()
X_train_smote_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# DataFrames for balanced data
X_train_smote_df = pd.DataFrame(X_train_smote_scaled, columns=features)
y_train_smote_df = pd.DataFrame(y_train_smote, columns=['class'])
train_data_smote = pd.concat([X_train_smote_df, y_train_smote_df], axis=1)
X_test_df = pd.DataFrame(X_test_scaled, columns=features)
y_test_df = pd.DataFrame(y_test, columns=['class'])
test_data = pd.concat([X_test_df, y_test_df], axis=1)
fraud_data_balanced = pd.concat([train_data_smote, test_data], axis=0).reset_index(drop=True)

# Encode Categorical Features
categorical_cols = ['source', 'browser', 'sex', 'country']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cols = encoder.fit_transform(fraud_data[categorical_cols])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))

# Merge encoded features with balanced numerical features
fraud_data_final = pd.concat([fraud_data_balanced, encoded_df.reset_index(drop=True)], axis=1)

# Include original columns for reference
fraud_data_final = pd.concat([
    fraud_data[['user_id', 'device_id', 'signup_time', 'purchase_time', 'ip_address', 'ip_address_int']].reset_index(drop=True),
    fraud_data_final
], axis=1)

# Save preprocessed datasets
fraud_data_final.to_csv('../data/preprocessed/Fraud_Data_preprocessed.csv', index=False)
creditcard_data[['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]].to_csv('../data/preprocessed/creditcard_preprocessed.csv', index=False)

print("Preprocessing complete. Datasets saved as 'Fraud_Data_preprocessed.csv' and 'creditcard_preprocessed.csv'.")