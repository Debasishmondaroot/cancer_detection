from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np
import numpy
import sklearn
import os

print("=" * 50)
print("SKLEARN VERSION:", sklearn.__version__)
print("NUMPY VERSION:", numpy.__version__)
print("=" * 50)

app = Flask(__name__)
app.secret_key = "replace_this_with_a_random_secret"

MODEL_PATHS = [
    os.path.join("saved_model", "cancer_random_forest.pkl"),
    os.path.join("saved_model", "cancer_model.pkl"),
    "cancer_random_forest.pkl",
    "best_classifier_model.pkl"
]

model = None

for p in MODEL_PATHS:
    if os.path.exists(p):
        try:
            model = joblib.load(p)
            print(f"Loaded model from: {p}")
            print("MODEL TYPE:", type(model))
            break
        except Exception as e:
            print(f"Found file {p} but failed to load: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model

    if model is None:
        flash("Model file not found.")
        return redirect(url_for("index"))

    try:
        features_text = request.form.get("features", "").strip()

        if not features_text:
            flash("Please enter comma-separated feature values.")
            return redirect(url_for("index"))

        parts = [p.strip() for p in features_text.split(",") if p.strip()]
        sample = [float(x) for x in parts]

        arr = np.array(sample).reshape(1, -1)

        prediction = model.predict(arr)
        raw_prediction = int(prediction[0])

        probability = None
        try:
            probability = model.predict_proba(arr).tolist()
        except Exception as e:
            print("Probability error:", e)

        label_text = (
            "Malignant (Cancer)"
            if raw_prediction == 1
            else "Benign (No Cancer)"
        )

        return render_template(
            "result.html",
            prediction=label_text,
            probability=probability,
            raw=raw_prediction
        )

    except Exception as e:
        print("Prediction Error:", str(e))
        flash(f"Error during prediction: {str(e)}")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
```
