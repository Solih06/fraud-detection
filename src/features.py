# src/features.py
import pandas as pd

def engineer_features(df):
    try:
        # Explicitly convert to datetime to avoid the subtraction error
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        # Time-based features
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        
        # Velocity features
        df['device_tx_count'] = df.groupby('device_id')['purchase_time'].transform('count')
        
        return df
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        return df