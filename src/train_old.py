import os
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from dotenv import load_dotenv

# Import ajustado para funcionar como módulo
from src.prepare_data import load_and_split

load_dotenv()

# Configurações do DagsHub
DAGSHUB_USER  = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO  = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

mlflow.set_experiment("analise-emocional")

def main(max_feature, ngram_max, c, run_name):
    # Carregando os dados (Certifique-se que o caminho em prepare_data.py está correto)
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"\n Iniciando Run: {run_name}")
    print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

    with mlflow.start_run(run_name=run_name):
        # Logging de Parâmetros
        mlflow.log_params({
            "max_features": max_feature,
            "ngram_range": f"(1,{ngram_max})",
            "C": c,
            "solver": "lbfgs",
            "dataset": "data/AnaliseDeEmocoes_PT-br.csv",
        })

        # Construção do Pipeline (multi_class removido para compatibilidade)
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_feature,
                ngram_range=(1, ngram_max),
                strip_accents="unicode",
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                C=c,
                max_iter=1000,
                solver="lbfgs"
            )),
        ])

        # Treinamento
        pipeline.fit(X_train, y_train)

        # Avaliação
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print("-" * 30)
        print(classification_report(y_test, y_pred, zero_division=0))
        print("-" * 30)

        # Logging de Métricas
        mlflow.log_metrics({
            "accuracy": acc,
            "f1_weighted": f1,
        })

        # Registro do Modelo
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="EmotionalClassifier",
        )

        print(f" Run '{run_name}' finalizado!")
        print(f"Acurácia: {acc:.2%} | F1-Score: {f1:.3f}")

if __name__ == "__main__":
    # Dicionário atualizado com os valores solicitados
    parametros = {
        "exp-1-baseline": {
            "MAX_FEATURES": 3000, 
            "NGRAM_MAX": 1, 
            "C": 0.1, 
            "RUN_NAME": "exp-1-baseline"
        },
        "exp-2-bigramas": {
            "MAX_FEATURES": 5000, 
            "NGRAM_MAX": 2, 
            "C": 1.0, 
            "RUN_NAME": "exp-2-bigramas"
        },
        "exp-3-vocab-largo": {
            "MAX_FEATURES": 10000, 
            "NGRAM_MAX": 2, 
            "C": 10.0, 
            "RUN_NAME": "exp-3-vocab-largclso"
        },
    }

    for nome_exp, p in parametros.items():
        main(p["MAX_FEATURES"], p["NGRAM_MAX"], p["C"], p["RUN_NAME"])