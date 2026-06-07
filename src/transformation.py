# src/transformation.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def get_preprocessor(df_columns):
    # Dynamically select columns that exist in the dataframe
    numeric_features = [c for c in ['purchase_value', 'age'] if c in df_columns]
    categorical_features = [c for c in ['sex', 'browser', 'country_x', 'country_y'] if c in df_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    return preprocessor