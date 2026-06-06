import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from utils.preprocessing import load_dataset, preprocess_data

# Load dataset
df = load_dataset("cancer_detection.csv")
print("Dataset Columns:", df.columns)

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("saved_model", exist_ok=True)
joblib.dump(model, "saved_model/cancer_model.pkl")
print("Model saved!")

