
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Análise de emoções",
    page_icon="💬",
    layout="centered",
)

# ── Carrega o modelo (cache: não recarrega a cada interação) ─────────────────
@st.cache_resource
def load_model():
    """
    @st.cache_resource garante que o modelo é carregado apenas uma vez.
    Sem isso, o arquivo seria lido a cada clique do usuário.
    """
    return joblib.load("src/model.pkl")

pipeline = load_model()

# Mapeamento visual por emoção
EMOJI = {
    # macro-emoções (modelo atual)
    "Afeto_Alegria":        "😄",
    "Aprovacao_Admiracao":  "😊",
    "Raiva_Desaprovacao":   "😡",
    "Tristeza_Ansiedade":   "😢",
    "Surpresa_Curiosidade": "😲",
    "Neutro_Percepcao":     "😐",
    "Desejo":               "🌟",
    # emoções individuais (modelo antigo — compatibilidade)
    "alegria": "😄", "satisfacao": "😊", "tristeza": "😢",
    "raiva": "😡",  "medo": "😨",       "surpresa": "😲",
    "frustracao": "😤", "nojo": "🤢",   "neutro": "😐",
}

COLOR = {
    # macro-emoções (modelo atual)
    "Afeto_Alegria":        "green",
    "Aprovacao_Admiracao":  "limegreen",
    "Raiva_Desaprovacao":   "red",
    "Tristeza_Ansiedade":   "steelblue",
    "Surpresa_Curiosidade": "orange",
    "Neutro_Percepcao":     "gray",
    "Desejo":               "violet",
    # emoções individuais (modelo antigo — compatibilidade)
    "alegria": "green",     "satisfacao": "limegreen", "tristeza": "blue",
    "raiva": "red",         "medo": "purple",          "surpresa": "orange",
    "frustracao": "darkorange", "nojo": "brown",        "neutro": "gray",
}

# ── Interface principal ──────────────────────────────────────────────────────
st.title("Análise de emoções")
st.markdown(
    "Analisa emoções em reviews, classificando os textos conforme a emoção predominante."
    " usando TF-IDF + Regressão Logística com rastreamento no DagsHub."
)

st.divider()

# ── Seção 1: Análise de texto único ─────────────────────────────────────────
st.subheader("Analisar um review")

texto = st.text_area(
    label="Cole o texto do review aqui:",
    height=120,
    placeholder='Ex: "Exprese a sua emoção"' \
    '',
)

if st.button("Analisar emoção", type="secondary", use_container_width=True):
    if texto.strip():
        # Faz a predição
        pred    = pipeline.predict([texto])[0]
        classes = pipeline.classes_

        if hasattr(pipeline, "predict_proba"):
            probas = pipeline.predict_proba([texto])[0]
        else:
            scores = pipeline.decision_function([texto])[0]
            scores = scores - scores.max()
            exp_scores = np.exp(scores)
            probas = exp_scores / exp_scores.sum()

        emoji = EMOJI.get(pred, "")
        st.markdown(f"### {emoji} Emoção detectada: **{pred.upper()}**")

        st.markdown("##### Probabilidade por classe")
        for cls, prob in zip(classes, probas):
            st.progress(float(prob), text=f"{cls}: {prob:.0%}")
    else:
        st.warning("Digite ou cole um texto antes de analisar.")

st.divider()

# ── Seção 2: Análise em lote via CSV ────────────────────────────────────────
st.subheader("Análise em lote (CSV)")
st.markdown(
    "Envie um arquivo CSV com uma coluna chamada **`texto`**. "
    "O modelo vai classificar cada linha e você pode baixar o resultado."
)

uploaded_file = st.file_uploader(
    "Selecione o arquivo CSV",
    type=["csv"],
    help="O arquivo deve ter uma coluna chamada 'texto'",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "texto" not in df.columns:
        st.error("O CSV precisa ter uma coluna chamada 'texto'.")
    else:
        # Classifica todas as linhas
        df["emocao"] = pipeline.predict(df["texto"].astype(str))

        # Exibe o resultado
        st.success(f"{len(df)} reviews classificados!")
        st.dataframe(df, use_container_width=True)

       # Estatísticas rápidas por emoção
        contagem = df["emocao"].value_counts()

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        col7, col8, col9 = st.columns(3)

        col1.metric("Alegria",      contagem.get("alegria", 0))
        col2.metric("Satisfação",   contagem.get("satisfacao", 0))
        col3.metric("Tristeza",     contagem.get("tristeza", 0))

        col4.metric("Raiva",        contagem.get("raiva", 0))
        col5.metric("Medo",         contagem.get("medo", 0))
        col6.metric("Surpresa",     contagem.get("surpresa", 0))

        col7.metric("Frustração",   contagem.get("frustracao", 0))
        col8.metric("Nojo",         contagem.get("nojo", 0))
        col9.metric("Neutro",       contagem.get("neutro", 0))


        # Download do resultado
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar resultado como CSV",
            data=csv_bytes,
            file_name="resultado_emocao.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.divider()
st.caption(
    "Modelo: TF-IDF + Logistic Regression · "
    "Rastreamento: MLflow + DagsHub · "
    "Deploy: Docker + Render"
)
