# scripts/train_churn_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

# === 1. Load Raw Data ===
customers = pd.read_csv("data/raw/customers.csv")
transactions = pd.read_csv("data/raw/transactions.csv")
tickets = pd.read_csv("data/raw/support_tickets.csv")
sessions = pd.read_csv("data/raw/web_sessions.csv")
churn = pd.read_csv("data/raw/churn_labels.csv")

# === 2. Feature Engineering ===

# Tenure
customers['join_date'] = pd.to_datetime(customers['join_date'])
customers['tenure_days'] = (pd.to_datetime("today") - customers['join_date']).dt.days

# Transaction features
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
txn_features = transactions.groupby("customer_id").agg({
    "amount": ["sum", "mean", "count"],
    "transaction_date": lambda x: (pd.to_datetime("today") - x.max()).days
})
txn_features.columns = ["total_spent", "avg_txn_value", "num_transactions", "days_since_last_txn"]
txn_features.reset_index(inplace=True)

# Support tickets
ticket_features = tickets.groupby("customer_id").agg({
    "ticket_date": "count",
    "resolution_time": "mean"
}).rename(columns={
    "ticket_date": "num_tickets",
    "resolution_time": "avg_resolution_time"
}).reset_index()

ticket_features = ticket_features.merge(customers[['customer_id', 'tenure_days']], on='customer_id', how='left')
ticket_features['ticket_rate'] = (ticket_features['num_tickets'] / (ticket_features['tenure_days']/30 + 1)).round(2)

# Web sessions
sessions['session_date'] = pd.to_datetime(sessions['session_date'])
web_features = sessions.groupby("customer_id").agg({
    "duration_min": ["mean", "sum"],
    "pages_viewed": "mean",
    "session_date": lambda x: (pd.to_datetime("today") - x.max()).days
})
web_features.columns = ["avg_session_duration", "total_session_time", "avg_pages", "days_since_last_visit"]
web_features.reset_index(inplace=True)

# === 3. Merge All Data ===
df = customers[['customer_id', 'age', 'tenure_days']].copy()
df = df.merge(txn_features, on='customer_id', how='left')
df = df.merge(ticket_features.drop(columns='tenure_days'), on='customer_id', how='left')
df = df.merge(web_features, on='customer_id', how='left')
df = df.merge(churn, on='customer_id', how='left')

# === 4. Handle Missing Values ===
imputer = SimpleImputer(strategy='mean')
X = df.drop(columns=['customer_id', 'churn'])
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y = df['churn']

# === 5. Scale Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# === 6. Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# === 7. Apply SMOTE ===
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# === 8. Train XGBoost Model ===
model = XGBClassifier(
    scale_pos_weight=2.5,  # adjust if imbalance ratio is known
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5
)
model.fit(X_resampled, y_resampled)

# === 9. Evaluate Model ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("📊 Classification Report")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)
print(f"📈 ROC AUC Score: {auc:.4f}")

# === 10. Feature Importance ===
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n🔍 Top 10 Important Features:")
print(importances.head(10))

# === 11. Save Model & Scaler ===
joblib.dump(model, "scripts/best_churn_model.pkl")
joblib.dump(scaler, "scripts/scaler.pkl")
print("✅ Model and scaler saved to 'scripts/'")

# === 12. Save Final Dataset for Power BI ===
df.to_csv("data/processed/churn_dataset.csv", index=False)
print("📁 Final dataset saved to 'data/processed/churn_dataset.csv'")


