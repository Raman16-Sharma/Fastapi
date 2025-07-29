import joblib
import os
import pandas as pd

# Load model and label encoders
model_path = os.path.join(os.path.dirname(__file__), '../model/loan_model.joblib')
encoders_path = os.path.join(os.path.dirname(__file__), '../model/label_encoders.joblib')

model = joblib.load(model_path)
label_encoders = joblib.load(encoders_path)

def predict_loan_default(input_data: dict) -> str:
    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Apply label encoding
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])
        else:
            raise ValueError(f"Missing column: {col}")

    # Predict using model
    prediction = model.predict(df)[0]
    return "Approved" if prediction == 0 else "Rejected"
