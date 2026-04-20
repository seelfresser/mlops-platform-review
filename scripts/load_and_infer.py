import mlflow
import numpy as np

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

MODEL_URI = "models:/iris-classifier/latest"

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model = mlflow.pyfunc.load_model(MODEL_URI)

    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(sample)

    print("Input:", sample.tolist())
    print("Prediction:", prediction.tolist())


if __name__ == "__main__":
    main()