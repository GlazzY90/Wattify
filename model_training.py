import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import joblib
import os
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score,
    accuracy_score, classification_report
)
from sklearn.model_selection import GridSearchCV

# === 1. LOAD DATA SPLIT ===
data = joblib.load("processed/dataset_split.pkl")

X_train = data["X_train"]
X_val   = data["X_val"]
X_test  = data["X_test"]

y_reg_train = data["y_reg_train"]
y_reg_val   = data["y_reg_val"]
y_reg_test  = data["y_reg_test"]

y_clf_train = data["y_clf_train"]
y_clf_val   = data["y_clf_val"]
y_clf_test  = data["y_clf_test"]

fitur = data["feature_order"]

# === 2. TRAINING REGRESSION MODEL (Random Forest) ===
reg_model = RandomForestRegressor(random_state=42)
reg_model.fit(X_train, y_reg_train)

y_pred_reg = reg_model.predict(X_test)

print("=== EVALUASI REGRESI ===")
print("MAE :", mean_absolute_error(y_reg_test, y_pred_reg))
print("RMSE:", root_mean_squared_error(y_reg_test, y_pred_reg))
print("R2  :", r2_score(y_reg_test, y_pred_reg))

# === 3. TRAINING CLASSIFICATION MODEL (Logistic Regression with GridSearchCV) ===
numeric_features = ['num_rooms', 'num_people', 'housearea', 'ave_monthly_income', 'num_children']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# Grid search setup
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l2"],
    "solver": ["lbfgs", "saga"],
    "max_iter": [500, 1000]
}

grid_clf = GridSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_clf.fit(X_train_scaled, y_clf_train)

# Best model
clf_model = grid_clf.best_estimator_

y_pred_clf = clf_model.predict(X_test_scaled)

print("\n=== EVALUASI KLASIFIKASI ===")
print("Best Params:", grid_clf.best_params_)
print("Akurasi:", accuracy_score(y_clf_test, y_pred_clf))
print("Classification Report:\n", classification_report(y_clf_test, y_pred_clf))

# === 4. SAVE MODELS, SCALER, AND METRICS ===
os.makedirs("model", exist_ok=True)

joblib.dump(reg_model, "model/regressor.pkl")
joblib.dump(clf_model, "model/classifier.pkl")
joblib.dump(scaler, "model/scaler.pkl")

with open("model/feature_order.json", "w") as f:
    json.dump(fitur, f)

# Save model accuracies
metrics = {
    "regressor_r2": r2_score(y_reg_test, y_pred_reg),
    "classifier_accuracy": accuracy_score(y_clf_test, y_pred_clf)
}
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f)

print("\nOK Model berhasil dilatih dan disimpan.")