from src.data_collection import fetch_data_from_csv
from src.data_preprocessing import preprocess_data
from src.feature_engineering import create_new_features 
from src.model_training import train_model
from src.anomaly_detection import detect_anomalies

def main():
    # Data pipeline
    raw_data_path = '/home/abubakar/headstarter/Anomaly-Detection/data/raw/FinancialMarketData.csv'
    data = fetch_data_from_csv(raw_data_path)
    cleaned_data = preprocess_data(data)
    features = create_new_features(cleaned_data)

    # Model pipeline
    model, train_report, test_report = train_model(features)

    # Print reports
    print("Training Report:\n", train_report)
    print("Testing Report:\n", test_report)

    # Anomaly detection
    anomalies = detect_anomalies(features, contamination=0.1)  
    print("Anomalies detected:", anomalies)

if __name__ == "__main__":
    main()
