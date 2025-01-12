import pandas as pd

def fetch_data_from_csv(file_path: str):
    """
    Fetches data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please provide a valid path.")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The provided CSV file is empty.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
