import os
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from dotenv import load_dotenv

from prepare_data import load_and_split

load_dotenv()

DAGSHUB_USER  = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO  = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
)

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

mlflow.set_experiment("analise-emocional-v2")

def main(max_feature, ngram_max, c, run_name):
    MAX_FEATURES = max_feature
    NGRAM_MAX = ngram_max
    C = c
    RUN_NAME = run_name

    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

    with mlflow.start_run(run_name=RUN_NAME):

        mlflow.log_params({
            "max_features": MAX_FEATURES,
            "ngram_range":  f"(1,{NGRAM_MAX})",
            "C":            C,
            "solver":       "liblinear",
            "test_size":    0.2,
            "dataset":      "../data/AnaliseDeEmocoes_PT-br_balanceado.csv",
            "num_classes":  y_train.nunique(),
            "model":        "CalibratedLinearSVC"
        })

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=MAX_FEATURES,
                ngram_range=(1, NGRAM_MAX),
                strip_accents="unicode",
                sublinear_tf=True,
                min_df=2,
                max_df=0.90
            )),
            ("clf", CalibratedClassifierCV(
                LinearSVC(
                    C=C,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42
                ),
                cv=5
            )),
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="weighted")

        print("\n" + "=" * 50)
        print(classification_report(y_test, y_pred, zero_division=0))
        print("=" * 50)

        mlflow.log_metrics({
            "accuracy":    acc,
            "f1_weighted": f1,
        })

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="EmotionalClassifier",
        )

        print(f"\nRun '{RUN_NAME}' finalizado com sucesso!")
        print(f"Acurácia : {acc:.2%}")
        print(f"F1 (weighted): {f1:.3f}")
        print(f"\nVisualize em: https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}")


if __name__ == "__main__":

    parametros = {
        "exp-1-baseline": {"MAX_FEATURES": None, "NGRAM_MAX": 2, "C": 0.1, "RUN_NAME": "exp-1-baseline"},
        "exp-2-bigramas": {"MAX_FEATURES": None, "NGRAM_MAX": 2, "C": 1.0, "RUN_NAME": "exp-2-bigramas"},
        "exp-3-trigramas": {"MAX_FEATURES": None, "NGRAM_MAX": 3, "C": 0.5, "RUN_NAME": "exp-3-trigramas"},
    }

    for nome_exp, params in parametros.items():
        max_feature = params["MAX_FEATURES"]
        ngram_max = params["NGRAM_MAX"]
        c = params["C"]
        run_name = params["RUN_NAME"]

        print(f'\nIniciando Experimento: {run_name}')
        print(f'Parametros: MAX_FEATURES={max_feature} | NGRAM_MAX={ngram_max} | C={c}')

        main(max_feature, ngram_max, c, run_name)