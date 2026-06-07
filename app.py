from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "replace_this_with_a_random_secret"

# Try loading model from saved_model folder (two common names)
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
            break
        except Exception as e:
            print(f"Found file {p} but failed to load: {e}")

@app.route("/")
def index():
    # Display simple input form with placeholders for feature values
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        flash("Model file not found. Place the trained model (cancer_random_forest.pkl or cancer_model.pkl) inside the saved_model/ folder or project root.")
        return redirect(url_for("index"))

    try:
        # Read features from form; expect all numeric inputs separated by commas or separate fields
        # We'll accept a single big textarea named 'features' with comma-separated values
        features_text = request.form.get("features", "").strip()
        if not features_text:
            flash("Please enter comma-separated feature values in the box.")
            return redirect(url_for("index"))

        # parse numbers
        parts = [p.strip() for p in features_text.split(",") if p.strip()!='']
        sample = [float(x) for x in parts]

        # model expects shape (1, n_features)
        arr = np.array(sample).reshape(1, -1)
        pred = model.predict(arr)
        proba = None
        try:
            proba = model.predict_proba(arr).tolist()
        except:
            proba = None

        label = int(pred[0])
        label_text = "Malignant (Cancer)" if label==1 else "Benign (No Cancer)"

        return render_template("result.html", prediction=label_text, probability=proba, raw=label)
    except Exception as e:
        flash(f"Error during prediction: {e}")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
