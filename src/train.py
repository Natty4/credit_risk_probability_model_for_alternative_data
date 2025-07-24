import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import sys, pathlib

ROOT_DIR = '../'
SRC_DIR = ROOT_DIR
sys.path.append(str(SRC_DIR))

RANDOM_STATE = 42

def load_data(path="data/processed/labeled_customers.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["CustomerId", "is_high_risk"])
    y = df["is_high_risk"]
    return X, y

def preprocess_features(X):
    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols)
    ])

    return preprocessor


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "report": classification_report(y_test, y_pred, digits=4)
    }


def train_and_log_model(name, model, X_train, X_test, y_train, y_test, preprocessor):
    with mlflow.start_run(run_name=name) as run:
        pipeline = Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("classifier", model)
        ])
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)

        mlflow.log_param("model", name)
        mlflow.log_metrics({k: round(v, 4) for k, v in metrics.items() if k != "report"})

        input_example = X_train.head(3)
        signature = infer_signature(X_train, pipeline.predict(X_train))

        mlflow.sklearn.log_model(
            pipeline,
            name="credit-risk-model",
            signature=signature,
            input_example=input_example
        )

        # Workaround for YAML serialization bug
        client = mlflow.tracking.MlflowClient()
        run_id = run.info.run_id
        tags = client.get_run(run_id).data.tags
        if "mlflow.log-model.history" in tags:
            client.delete_tag(run_id, "mlflow.log-model.history")

        # Register the best model
        if name == "RandomForest":
            result = mlflow.register_model(
                model_uri=f"runs:/{run_id}/credit-risk-model",
                name="credit-risk-rf-best"
            )
            print(f"✅ Registered {name} model as credit-risk-rf-best")

        print(f"\n{name} Evaluation:")
        print(metrics["report"])
        return name, metrics["roc_auc"]
    

def main():
    mlflow.set_experiment("credit-risk-model")

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    preprocessor = preprocess_features(X)

    models = {
        "LogisticRegression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=100)
    }

    best_model, best_score = None, -1
    for name, model in models.items():
        model_name, score = train_and_log_model(name, model, X_train, X_test, y_train, y_test, preprocessor)
        if score > best_score:
            best_score = score
            best_model = model_name

    print(f"\n✅ Best model: {best_model} with ROC-AUC: {best_score:.4f}")

if __name__ == "__main__":
    main()



