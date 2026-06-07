# src/sampling.py
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def prepare_and_resample(X, y, test_size=0.2, random_state=42):
    """
    Splits the data and applies SMOTE to the training set only.
    Justification: SMOTE is used to address class imbalance by generating 
    synthetic samples for the minority class, ensuring the model learns 
    robust patterns without data leakage into the test set.
    """
    # 1. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 2. Apply SMOTE only to training data
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test