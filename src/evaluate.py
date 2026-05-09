import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, accuracy_score, f1_score
from dotenv import load_dotenv

try:
    from src.prepare_data import load_and_split  # python -m src.evaluate
except ModuleNotFoundError:
    from prepare_data import load_and_split       # python evaluate.py

load_dotenv()

DAGSHUB_USER  = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO  = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
)
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN


def main():

    model_uri = "models:/EmotionalClassifier@production"

    print(f"Baixando modelo: {model_uri}")
    print(f"Fonte: https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}")

    pipeline = mlflow.sklearn.load_model(model_uri)
    print("Modelo carregado com sucesso!")


    _, X_test, _, y_test = load_and_split()
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print("\n" + "=" * 50)
    print("Avaliação do modelo em produção:")
    print(classification_report(y_test, y_pred))
    print(f"Acurácia : {acc:.2%}")
    print(f"F1 (weighted): {f1:.3f}")
    print("=" * 50)

    joblib.dump(pipeline, "src/model.pkl")
    print("\nModelo salvo em src/model.pkl — pronto para o Streamlit!")


if __name__ == "__main__":
    main()
