from fastapi import FastAPI, HTTPException
from app.schema.input_data import LoanInput
from app.service.predict import predict_loan_default

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Loan Default Prediction API is running."}

@app.post("/predict")
def predict(input_data: LoanInput):
    try:
        result = predict_loan_default(input_data.dict())
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
