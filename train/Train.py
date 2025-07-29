import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib

# === Step 1: Load dataset ===
data_path = os.path.join(os.path.dirname(__file__), '../app/data/loan_data.csv')
loan_data = pd.read_csv(data_path)

# === Step 2: Encode categorical columns ===
label_encoders = {}
categorical_columns = [
    'person_gender',
    'person_education',
    'person_home_ownership',
    'loan_intent',
    'previous_loan_defaults_on_file'
]

for col in categorical_columns:
    if col in loan_data.columns:
        le = LabelEncoder()
        loan_data[col] = le.fit_transform(loan_data[col])
        label_encoders[col] = le
    else:
        raise ValueError(f"Column '{col}' not found in dataset!")

# === Step 3: Define features and target ===
target_column = 'loan_status'
if target_column not in loan_data.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset!")

X = loan_data.drop(target_column, axis=1)
y = loan_data[target_column]

# === Step 4: Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 5: Grid Search with Cross-Validation ===
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# === Step 6: Final model training ===
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train, y_train)

# === Step 7: Save model and encoders ===
model_dir = os.path.join(os.path.dirname(__file__), '../app/model')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(best_rf_model, os.path.join(model_dir, 'loan_model.joblib'))
joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.joblib'))

print("✅ Model and encoders saved successfully!")

