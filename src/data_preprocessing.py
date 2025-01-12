import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 


def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the data by handling missing values and scaling features.
    """
    # Handle missing values
    df = df.dropna()
    
    # Convert categorical data to numeric
    df = pd.get_dummies(df, drop_first=True)
    
    # Feature scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    
    return pd.DataFrame(scaled_features, columns=df.columns)

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    Parameters:
    df (DataFrame): The input data.
    test_size (float): The proportion of the data to include in the test split.
    random_state (int): The seed used by the random number generator.
    
    Returns:
    DataFrame, DataFrame: The training and testing data.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

if __name__ == "__main__":
    file_path = 'path_to_your_data.csv'
    data = load_data(file_path)
    preprocessed_data = preprocess_data(data)
    train_data, test_data = split_data(preprocessed_data)
    
    # Save the preprocessed data
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)