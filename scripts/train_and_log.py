import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


MLFLOW_TRACKING_URI =  "http://127.0.0.1:5000"
EXPERIMENT_NAME = "mlops_online"
REGISTERED_MODEL_NAME = "iris-classifier"


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    n_estimators = 100
    max_depth = 5

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("accuracy", accuracy)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_test[:2]
        )

        print("Run ID:", run.info.run_id)
        print("Accuracy:", accuracy)
        print("Model URI:", model_info.model_uri)


if __name__ == "__main__":
    main()
