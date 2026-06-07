import pandas as pd
from src.cleaning import load_data, clean_data
from src.features import engineer_features
from src.transformation import get_preprocessor
from src.sampling import prepare_and_resample

def run_all():
    print("--- Starting Pipeline ---")
    
    # 1. Load & Clean
    df = load_data('data/raw/Fraud_Data.csv')
    df = clean_data(df)
    print("Data loaded and cleaned.")
    
    # 2. Engineer
    df = engineer_features(df)
    print("Features engineered.")
    
    # 3. Preprocess
    cols_to_drop = ['class', 'user_id', 'signup_time', 'purchase_time', 'ip_address']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df['class']
    
    preprocessor = get_preprocessor(X.columns)
    X_numeric = preprocessor.fit_transform(X)
    X_df = pd.DataFrame(X_numeric)
    print("Transformation complete.")
    
    # 4. Sample
    X_train, X_test, y_train, y_test = prepare_and_resample(X_df, y)
    print(f"Pipeline finished! Final training shape: {X_train.shape}")

if __name__ == "__main__":
    run_all()