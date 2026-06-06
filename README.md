# Cancer Detection Flask App

Simple Flask app that loads a trained Random Forest model and performs predictions on comma-separated feature input.

## How to use

1. Put your trained model file (e.g. `cancer_random_forest.pkl` or `cancer_model.pkl`) inside the `saved_model/` folder in the project root.
2. Ensure the model was trained on the same feature order as shown in the web form.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the app:
   ```
   python app.py
   ```
5. Open http://127.0.0.1:5000 in your browser.

## Notes
- The app expects numeric inputs separated by commas.
- If you saved a scaler during training, you must load and apply it before prediction — this demo assumes raw numeric scale matching the training pipeline or that model includes preprocessing.