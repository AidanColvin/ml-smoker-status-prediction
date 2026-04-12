import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, f1_score

# 1. Setup Directories
os.makedirs('data/visualizations', exist_ok=True)

# 2. Data Loading (Reading from your existing data/raw folder)
print("Loading data from data/raw/...")
df = pd.read_csv('data/raw/train-ml-smoker-status-prediction.csv')

# 3. Preprocessing & Lasso Feature Selection
X = df.drop(['id', 'smoking'], axis=1)
y = df['smoking']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# LassoCV for objective feature selection
print("Running Lasso for feature selection...")
lasso = LogisticRegressionCV(cv=5, penalty='l1', solver='saga', max_iter=2000, random_state=42)
lasso.fit(X_scaled, y)
selected_features = X.columns[lasso.coef_[0] != 0].tolist()
print(f"Lasso selected {len(selected_features)} features.")

X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, stratify=y, random_state=42)

# Re-scale based on selected features only
scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train)
X_test_scaled = scaler_final.transform(X_test)

# 4. Model Training
models = {
    'Logistic Regression': LogisticRegression(C=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
}

results = []
test_metrics = {}

print("Training models...")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=StratifiedKFold(5), scoring='roc_auc')
    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    test_metrics[name] = {'AUC': roc_auc_score(y_test, y_prob), 'y_prob': y_prob}
    results.append({'Model': name, 'CV Mean AUC': np.mean(cv_scores), 'CV Std': np.std(cv_scores)})

# 5. Advanced Visualizations
sns.set_theme(style="whitegrid")

# Visual 1: Correlation Heatmap (Explaining Lasso drops)
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), cmap='RdBu_r', center=0, annot=False)
plt.title('Feature Correlation Heatmap (Raw Data)')
plt.savefig('data/visualizations/correlation_heatmap.png')
plt.close()

# Visual 2: KDE Plot (Biological signal in Hemoglobin)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='hemoglobin', hue='smoking', fill=True, palette='coolwarm')
plt.title('Hemoglobin Distribution: Smokers vs Non-Smokers')
plt.savefig('data/visualizations/kde_hemoglobin.png')
plt.close()

# Visual 3: ROC Curves
plt.figure(figsize=(8, 8))
for name in models:
    fpr, tpr, _ = roc_curve(y_test, test_metrics[name]['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC: {test_metrics[name]['AUC']:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve Comparison')
plt.legend()
plt.savefig('data/visualizations/roc_curves_new.png')
plt.close()

# Visual 4: Feature Importance
gb_imp = pd.Series(models['Gradient Boosting'].feature_importances_, index=selected_features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
gb_imp.head(10).plot(kind='barh', color='teal').invert_yaxis()
plt.title('Top 10 Features (Gradient Boosting)')
plt.savefig('data/visualizations/feature_importance_new.png')
plt.close()

print("All visualizations saved to data/visualizations/")
