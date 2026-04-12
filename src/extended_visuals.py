import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve
from sklearn.inspection import PartialDependenceDisplay

# 1. Setup & Data Loading
data_path = 'data/raw/train-ml-smoker-status-prediction.csv'
os.makedirs('data/visualizations', exist_ok=True)
df = pd.read_csv(data_path)

# 2. Simple Preprocessing (Using Top Features from your report)
features = ['hemoglobin', 'height(cm)', 'Gtp', 'triglyceride', 'age', 'weight(kg)', 'waist(cm)', 'HDL', 'LDL', 'ALT']
X = df[features]
y = df['smoking']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train Model
model = GradientBoostingClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("✔ Model trained. Generating advanced visuals...")

# --- Visual 1: Calibration Curve ---
plt.figure(figsize=(8, 6))
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', label='Gradient Boosting')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
plt.title('Calibration Curve (Reliability Diagram)')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend()
plt.savefig('data/visualizations/diagnostic_calibration.png')

# --- Visual 2: Learning Curve (Evidence for more data) ---
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train_scaled, y_train, cv=3, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 5), scoring='roc_auc'
)
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation Score')
plt.title('Learning Curves (AUC)')
plt.xlabel('Training Samples')
plt.ylabel('AUC Score')
plt.legend()
plt.savefig('data/visualizations/diagnostic_learning_curve.png')

# --- Visual 3: Threshold vs Precision-Recall ---
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions[:-1], label='Precision', color='blue')
plt.plot(thresholds, recalls[:-1], label='Recall', color='green')
plt.axvline(0.5, color='red', linestyle=':', label='Default 0.5 Cutoff')
plt.title('Precision and Recall vs. Decision Threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.savefig('data/visualizations/diagnostic_threshold_tuning.png')

# --- Visual 4: Partial Dependence Plot (Biological Impact) ---
print("✔ Calculating Partial Dependence (this may take a moment)...")
fig, ax = plt.subplots(figsize=(12, 4))
PartialDependenceDisplay.from_estimator(model, X_train_scaled, features=[0, 2], 
                                        feature_names=features, ax=ax)
plt.suptitle('Partial Dependence: Hemoglobin and Gtp impact on Smoking Prob.')
plt.tight_layout()
plt.savefig('data/visualizations/diagnostic_partial_dependence.png')

print("✔ DONE. New files created in data/visualizations:")
print("  - diagnostic_calibration.png")
print("  - diagnostic_learning_curve.png")
print("  - diagnostic_threshold_tuning.png")
print("  - diagnostic_partial_dependence.png")
