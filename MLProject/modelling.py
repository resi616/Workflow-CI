import os
import argparse
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier

def train_model(n_estimators, max_depth):
    """
    Melatih model Random Forest Classifier dengan hyperparameter yang diterima
    dari parameter CLI. Autologging MLflow akan mencatat parameter dan metrik
    ke run yang aktif dari eksekusi MLflow Project.
    """
    data_dir = "student-performance-prediction-dataset_preprocessing"
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    mlflow.sklearn.autolog()

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth > 0 else None,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model trained with n_estimators={n_estimators}, max_depth={max_depth}")
    print(f"Akurasi model baseline: {accuracy:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Random Forest Classifier for Student Placement Status Prediction")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    print("Memulai pelatihan model CI")
    try:
        train_model(args.n_estimators, args.max_depth)
        print("Pelatihan model CI selesai dengan sukses")
    except Exception as e:
        print(f"Proses gagal: {e}")
