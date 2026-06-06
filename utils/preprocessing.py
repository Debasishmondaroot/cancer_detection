import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df, target_column="target"):
   # Correct target column
    target_column = "target"

    # Split features and labels
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Scale dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

