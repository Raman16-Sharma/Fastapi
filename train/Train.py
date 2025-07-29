import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
loan_data = pd.read_csv('/mnt/data/loan_data.csv')

# Preprocessing: Handle categorical columns with LabelEncoder
label_encoders = {}
categorical_columns = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']

for col in categorical_columns:
    le = LabelEncoder()
    loan_data[col] = le.fit_transform(loan_data[col])
    label_encoders[col] = le

# Prepare features and target variable
X = loan_data.drop('loan_status', axis=1)  # Features
y = loan_data['loan_status']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [10, 20, 30, None],  # Depth of trees
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required at a leaf node
    'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider for splitting
}

# Initialize Random Forest and GridSearchCV
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_rf_model = grid_search.best_estimator_

# Train the best model on the entire training set
best_rf_model.fit(X_train, y_train)

# Save the model and encoders using joblib for deployment
joblib.dump(best_rf_model, 'best_loan_classifier_model.pkl')
joblib.dump(label_encoders, 'best_label_encoders.pkl')

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model accuracy on the test set
test_accuracy = best_rf_model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
