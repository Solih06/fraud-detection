import pandas as pd

def load_data(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def clean_data(df):
    # Move your cleaning logic here
    df = df.drop_duplicates()
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    return df.dropna(subset=['ip_address'])