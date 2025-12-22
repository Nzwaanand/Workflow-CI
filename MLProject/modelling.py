import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import tempfile

# =========================
# CONFIG
# =========================
mlflow.set_experiment("Prediksi_Balita_Stunting_Advance")
mlflow.sklearn.autolog(log_models=False)  # manual logging untuk kontrol penuh

DATA_PATH = "stunting_balita_preprocessing.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset tidak ditemukan di: {DATA_PATH}")

# =========================
# TRAINING FUNCTION
# =========================
def run_advance():
    df = pd.read_csv(DATA_PATH)

    # Features & Target
    X = df.drop("Status Gizi", axis=1)
    y = df["Status Gizi"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest + GridSearch
    rf_param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=rf_param_grid,
        cv=3,
        n_jobs=-1,
        verbose=0
    )

    # Start MLflow run
    with mlflow.start_run(run_name="RandomForest_Stunting"):

        grid_search.fit(X_train_scaled, y_train)
        best_rf_model = grid_search.best_estimator_

        # Predictions
        y_pred = best_rf_model.predict(X_test_scaled)

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted")
        }

        # Log params
        mlflow.log_params(grid_search.best_params_)
        # Log metrics
        mlflow.log_metrics(metrics)

        # Log artifacts in temp folder
        with tempfile.TemporaryDirectory() as tmpdir:
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close()

            # Classification Report
            report_path = os.path.join(tmpdir, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write(classification_report(y_test, y_pred))
            mlflow.log_artifact(report_path)

        # Log model with signature
        input_example = pd.DataFrame(X_train_scaled[:1], columns=X.columns)
        signature = infer_signature(X_train_scaled, best_rf_model.predict(X_train_scaled))

        mlflow.sklearn.log_model(
            sk_model=best_rf_model,
            artifact_path="random_forest_model",
            input_example=input_example,
            signature=signature
        )

        print("✅ Training selesai, model sudah di-log di MLflow")
        print("Metrics:", metrics)


if __name__ == "__main__":
    run_advance()
