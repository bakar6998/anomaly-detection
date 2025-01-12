import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def fit(data, contamination=0.1):
    """
    Train an Isolation Forest model on the given data.

    Parameters:
    - data: np.ndarray : The input data for training.
    - contamination: float : The proportion of anomalies in the data.

    Returns:
    - model: IsolationForest : The trained Isolation Forest model.
    """
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    model.fit(data)
    return model

def predict(model, data):
    """
    Predict anomalies using a trained Isolation Forest model.

    Parameters:
    - model: IsolationForest : The trained model.
    - data: np.ndarray : The input data for predictions.

    Returns:
    - predictions: np.ndarray : Predicted labels (-1 for anomaly, 1 for normal).
    """
    return model.predict(data)

def detect_anomalies(data, contamination=0.1):
    """
    Detect anomalies in the given data using Isolation Forest.

    Parameters:
    - data: np.ndarray : The input data.
    - contamination: float : The proportion of anomalies in the data.

    Returns:
    - anomalies: pd.DataFrame : Data points detected as anomalies.
    - normal_data: pd.DataFrame : Data points detected as normal.
    """
    model = fit(data, contamination)  # Train the model on the given data
    predictions = predict(model, data)  # Use the trained model to predict anomalies

    # Create a DataFrame to visualize anomalies
    results = pd.DataFrame(data, columns=["Feature 1", "Feature 2"])  # Adjust column names as needed
    results["Anomaly"] = predictions  # Add a column for predictions (-1 for anomaly, 1 for normal)

    # Flag anomalies with -1, and normal data with 1
    anomalies = results[results["Anomaly"] == -1]  # Only select rows where Anomaly is -1 (anomalies)
    normal_data = results[results["Anomaly"] == 1]  # Select rows where Anomaly is 1 (normal)

    return anomalies, normal_data

if __name__ == "__main__":
    # Example usage with sample data
    np.random.seed(42)  # For reproducibility
    X = np.random.randn(100, 2)  # Generate random normal data
    X = np.vstack([X, np.array([[10, 10]])])  # Add an outlier

    # Detect anomalies
    anomalies, normal_data = detect_anomalies(X, contamination=0.05)
    
    print("Anomalies detected:")
    print(anomalies)  # Prints the anomalies (rows where Anomaly = -1)
    
    print("\nNormal data points:")
    print(normal_data)  # Prints the normal data (rows where Anomaly = 1)
