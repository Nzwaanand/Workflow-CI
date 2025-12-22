import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    classification_report
)

import mlflow
import mlflow.sklearn
import dagshub

# DAGSHUB CONFIG
DAGSHUB_USER = ("Nzwaanand")
DAGSHUB_REPO = "Membangun_Model"

dagshub.init(
    repo_owner=DAGSHUB_USER,
    repo_name=DAGSHUB_REPO,
    mlflow=True
)

mlflow.set_experiment("Prediksi_Balita_Stunting_Advance")

# DATASET PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "stunting_balita_preprocessing.csv")

# TRAINING PIPELINE
def run_advance():

    df = pd.read_csv(DATA_PATH)

    X = df.drop("Status Gizi", axis=1)
    y = df["Status Gizi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    with mlflow.start_run():

        # TRAIN
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # METRICS (MANUAL)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # ARTEFAK 1: CONFUSION MATRIX
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # ARTEFAK 2: CLASSIFICATION REPORT
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)

        mlflow.log_artifact("classification_report.txt")

        # MODEL
        mlflow.sklearn.log_model(model, "model")

        print("training selesai")
        print("Accuracy:", acc)


if __name__ == "__main__":
    run_advance()
