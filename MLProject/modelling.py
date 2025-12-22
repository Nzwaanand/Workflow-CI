import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# =========================
# MLflow config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mlruns_path = os.path.join(BASE_DIR, "mlruns")
mlflow.set_tracking_uri(f"file:{mlruns_path}")
mlflow.set_experiment("Prediksi_Balita_Stunting_CI")

# =========================
# Dataset path
# =========================
DATA_PATH = os.path.join(BASE_DIR, "stunting_balita_preprocessing.csv")

def run_model():
    print("Memulai training model...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print("Dataset berhasil diload")

    X = df.drop("Status Gizi", axis=1)
    y = df["Status Gizi"]

    # =========================
    # Train-test split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =========================
    # Scaling
    # =========================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================
    # Training & MLflow logging
    # =========================
    with mlflow.start_run(run_name="RandomForest_Stunting_CI") as run:
        # Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        first_class_label = list(report_dict.keys())[0]
        f1_first_class = report_dict.get(first_class_label, {}).get("f1-score", 0)

        cm = confusion_matrix(y_test, y_pred)
        tn = cm[0][0]
        fp = cm[0][1] if cm.shape[1] > 1 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric(f"f1_class_{first_class_label}", f1_first_class)
        mlflow.log_metric("specificity_class_0", specificity)

        # Log model with signature
        input_example = pd.DataFrame(X_train_scaled[:1], columns=X.columns)
        signature = infer_signature(X_train_scaled, model.predict(X_train_scaled))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            input_example=input_example,
            signature=signature
        )

        # Simpan run_id agar bisa dipakai di workflow CI
        run_id_path = os.path.join(BASE_DIR, "run_id.txt")
        with open(run_id_path, "w") as f:
            f.write(run.info.run_id)
        print(f"run_id.txt berhasil dibuat di {run_id_path}")

        print(f"Training selesai. Accuracy: {accuracy}")


if __name__ == "__main__":
    run_model()
