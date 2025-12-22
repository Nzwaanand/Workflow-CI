import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# ===============================
# Konfigurasi MLflow
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLRUNS_PATH = os.path.join(BASE_DIR, "mlruns")
mlflow.set_tracking_uri(f"file:{MLRUNS_PATH}")
mlflow.set_experiment("Prediksi_Balita_Stunting_Bsc")

mlflow.sklearn.autolog()

# ===============================
# Path Dataset
# ===============================
DATA_PATH = os.path.join(BASE_DIR, "stunting_balita_preprocessing.csv")

# ===============================
# Training Function
# ===============================
def run_model():
    print("Memulai training model...")
    print("Mencari dataset di:", DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    print("Dataset berhasil diload")

    X = df.drop("Status Gizi", axis=1)
    y = df["Status Gizi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    # Langsung fit model dan log metrics/model
    print("Training model...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    mlflow.log_metric("accuracy", accuracy)

    # Simpan model & scaler
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.iloc[:5]
    )

    mlflow.sklearn.log_model(
        sk_model=scaler,
        artifact_path="scaler"
    )

    print("Training selesai dan model berhasil disimpan di MLflow.")

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    os.makedirs(MLRUNS_PATH, exist_ok=True)  # pastikan folder mlruns ada
    run_model()
