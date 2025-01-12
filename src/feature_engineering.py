import pandas as pd

def handle_missing_values(data):
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    return data

def encode_categorical_features(data):
    data = pd.get_dummies(data, drop_first=True)
    return data

def scale_features(data, method='standard'):
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling method. Use 'standard' or 'minmax'.")
    
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    return data

def create_new_features(data):
    # Example: Create a new feature based on existing ones
    if 'feature1' in data.columns and 'feature2' in data.columns:
        data['new_feature'] = data['feature1'] * data['feature2']
    return data
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
