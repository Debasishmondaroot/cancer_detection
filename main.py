import joblib
import numpy as np

model = joblib.load("saved_model/cancer_model.pkl")

def predict(sample):
    sample = np.array(sample).reshape(1, -1)
    result = model.predict(sample)
    return result[0]

print("Cancer Detection System Ready!")
