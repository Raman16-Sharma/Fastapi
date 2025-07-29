from pydantic import BaseModel

class LoanInput(BaseModel):
    person_age: int
    person_income: int
    person_emp_length: int
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    person_gender: str
    person_education: str
    person_home_ownership: str
    loan_intent: str
    previous_loan_defaults_on_file: str
