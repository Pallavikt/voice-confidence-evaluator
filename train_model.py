# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

DATA_CSV = "ravdess_dataset_full.csv"
MODEL_PATH = "final_confidence_model.pkl"
SCALER_PATH = "final_conf_scaler.pkl"

df = pd.read_csv(DATA_CSV)

feature_cols = [c for c in df.columns if c.startswith("voice_conf") or c.startswith("text_conf") or c.startswith("feat_")]
X = df[feature_cols]
y = df['final_conf']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("✅ Evaluation")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"\n✅ Model saved: {MODEL_PATH}")
print(f"✅ Scaler saved: {SCALER_PATH}")