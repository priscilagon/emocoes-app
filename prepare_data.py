import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

_DEFAULT_DATA = Path(__file__).parent / "data" / "AnaliseDeEmocoes_PT-br_balanceado.csv"

STOPWORDS_PT = set([
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando", "muito", "há", "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso", "ela", "entre", "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem", "nas", "me", "esse", "eles", "estão", "você", "tinha", "foram", "essa", "num", "nem", "suas", "meu", "às", "minha", "têm", "numa", "pelos", "elas", "havia", "seja", "qual", "será", "nós", "tenho", "lhe", "deles", "essas", "esses", "pelas", "este", "fosse", "dele", "tu", "te", "vocês", "vos", "lhes", "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso", "nossa", "nossos", "nossas", "dela", "delas", "esta", "estes", "estas", "aquele", "aquela", "aqueles", "aquelas", "isto", "aquilo", "estou", "estamos", "estive", "esteve", "estivemos", "estiveram", "estava", "estávamos", "estavam", "estivera", "estivéramos", "esteja", "estejamos", "estejam", "estivesse", "estivéssemos", "estivessem", "estiver", "estivermos", "estiverem", "hei", "havemos", "hão", "houve", "houvemos", "houveram", "houvera", "houvéramos", "haja", "hajamos", "hajam", "houvesse", "houvéssemos", "houvessem", "houver", "houvermos", "houverem", "houverei", "houverá", "houveremos", "houverão", "houveria", "houveríamos", "houveriam", "sou", "somos", "são", "éramos", "eram", "fui", "fomos", "fôramos", "sejamos", "sejam", "fôssemos", "fossem", "for", "formos", "forem", "serei", "seremos", "serão", "seria", "seríamos", "seriam", "temos", "tém", "tínhamos", "tinham", "tive", "teve", "tivemos", "tiveram", "tivera", "tivéramos", "tenha", "tenhamos", "tenham", "tivesse", "tivéssemos", "tivessem", "tiver", "tivermos", "tiverem", "terei", "terá", "teremos", "terão", "teria", "teríamos", "teriam"
])

# Mapeamento de 28 emoções para 7 Macro-Emoções
MAPA_MACRO_EMOCOES = {
    'amor': 'Afeto_Alegria', 'alegria': 'Afeto_Alegria', 'diversao': 'Afeto_Alegria', 'carinho': 'Afeto_Alegria', 'empolgacao': 'Afeto_Alegria',
    'aprovacao': 'Aprovacao_Admiracao', 'admiracao': 'Aprovacao_Admiracao', 'gratidao': 'Aprovacao_Admiracao', 'orgulho': 'Aprovacao_Admiracao', 'otimismo': 'Aprovacao_Admiracao', 'alivio': 'Aprovacao_Admiracao',
    'raiva': 'Raiva_Desaprovacao', 'irritacao': 'Raiva_Desaprovacao', 'desaprovacao': 'Raiva_Desaprovacao', 'nojo': 'Raiva_Desaprovacao',
    'tristeza': 'Tristeza_Ansiedade', 'decepcao': 'Tristeza_Ansiedade', 'luto': 'Tristeza_Ansiedade', 'remorso': 'Tristeza_Ansiedade', 'medo': 'Tristeza_Ansiedade', 'vergonha': 'Tristeza_Ansiedade', 'nervosismo': 'Tristeza_Ansiedade',
    'surpresa': 'Surpresa_Curiosidade', 'curiosidade': 'Surpresa_Curiosidade', 'confusao': 'Surpresa_Curiosidade',
    'neutro': 'Neutro_Percepcao', 'percepcao': 'Neutro_Percepcao',
    'desejo': 'Desejo'
}

def clean_text(text):
    text = str(text).lower()
    # Remove pontuações
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove números
    text = re.sub(r'\d+', ' ', text)
    # Remove stopwords
    words = text.split()
    words = [w for w in words if w not in STOPWORDS_PT]
    return ' '.join(words)

def load_and_split(
    path: str = str(_DEFAULT_DATA),
    test_size: float = 0.2,
    seed: int = 42,
):
    df = pd.read_csv(path, sep=";")

    df = df.dropna(subset=["texto"])
    df = df[df["texto"].str.strip() != ""]

    # Aplicar a limpeza avançada
    df["texto"] = df["texto"].apply(clean_text)
    
    # Remover textos que ficaram vazios após a limpeza
    df = df[df["texto"].str.strip() != ""]

    label_cols = [col for col in df.columns if col != "texto"]
    
    # Pegar a emoção predominante e converter para MACRO-EMOÇÃO
    df["emocional_original"] = df[label_cols].idxmax(axis=1)
    df["emocional_macro"] = df["emocional_original"].map(MAPA_MACRO_EMOCOES)

    # Remover possiveis nulos gerados por algum mapeamento faltante
    df = df.dropna(subset=["emocional_macro"])

    X = df["texto"]
    y = df["emocional_macro"]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()

    print(f"Total de amostras de treino : {len(X_train)}")
    print(f"Total de amostras de teste  : {len(X_test)}")
    print(f"\nDistribuição de Macro-Classes no treino:")
    print(y_train.value_counts())