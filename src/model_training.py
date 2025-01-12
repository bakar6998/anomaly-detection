from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

def binarize_target(y):
    """
    Convert continuous values to binary.
    
    """
    return (y > 0).astype(int)

def train_model(dataset):
    """
    Train an Isolation Forest model and evaluate it.
    """
    # Check if the dataset is a file path or DataFrame
    if isinstance(dataset, str):
        df = pd.read_csv(dataset)
    elif isinstance(dataset, pd.DataFrame):
        df = dataset
    else:
        raise TypeError("The dataset parameter should be a file path (str) or a pandas DataFrame.")

    # Preprocess data
    X = df.iloc[:, :-1].values  # Feature columns
    y = df.iloc[:, -1].values  # Target column

    # Binarize y
    y = binarize_target(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(X_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Convert predictions from IsolationForest (-1: anomaly, 1: normal) to binary (0: normal, 1: anomaly)
    y_pred_train = np.array([1 if x == -1 else 0 for x in y_pred_train])
    y_pred_test = np.array([1 if x == -1 else 0 for x in y_pred_test])

    # Ensure both y_train and y_test are NumPy arrays
    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)

    # Evaluate
    train_report = classification_report(y_train, y_pred_train)
    test_report = classification_report(y_test, y_pred_test)

    return model, train_report, test_report
