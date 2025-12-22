import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

def main():
    # =========================
    # CONFIG MLflow
    # =========================
    mlflow.set_experiment("Prediksi_Balita_Stunting_Adv")
    mlflow.sklearn.autolog(log_models=False)

    # =========================
    # LOAD DATA
    # =========================
    DATA_PATH = "stunting_balita_preprocessing.csv"
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df.drop("Status Gizi", axis=1)
    y = df["Status Gizi"]

    # =========================
    # TRAIN TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =========================
    # SCALING
    # =========================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================
    # GRID SEARCH RANDOM FOREST
    # =========================
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

    # =========================
    # TRAIN & MLflow LOG
    # =========================
    with mlflow.start_run(run_name="RandomForest_Stunting") as run:
        grid_search.fit(X_train_scaled, y_train)
        best_rf_model = grid_search.best_estimator_

        y_pred = best_rf_model.predict(X_test_scaled)

        # Classification report & metrics
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        f1_macro = report_dict["macro avg"]["f1-score"]
        # Ambil f1 kelas pertama jika ada
        first_class_label = list(report_dict.keys())[0]
        f1_first_class = report_dict.get(first_class_label, {}).get("f1-score", 0)

        # Confusion matrix dan specificity untuk kelas 0
        cm = confusion_matrix(y_test, y_pred)
        tn = cm[0][0]
        fp = cm[0][1] if cm.shape[1] > 1 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Log params & metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric(f"f1_class_{first_class_label}", f1_first_class)
        mlflow.log_metric("specificity_class_0", specificity)

        # Log model dengan signature
        input_example = pd.DataFrame(X_train_scaled[:1], columns=X.columns)
        signature = infer_signature(X_train_scaled, best_rf_model.predict(X_train_scaled))

        mlflow.sklearn.log_model(
            sk_model=best_rf_model,
            artifact_path="random_forest_model",
            input_example=input_example,
            signature=signature
        )

        print("✅ Model Random Forest untuk Stunting Balita sudah di-log ke MLflow.")
        print("Metrics utama: f1_macro =", f1_macro, "| specificity_class_0 =", specificity)

if __name__ == "__main__":
    main()
