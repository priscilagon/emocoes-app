# Análise de Emoções em Português

Pipeline completo de MLOps para classificação de emoções em textos em português, com rastreamento de experimentos via MLflow + DagsHub e interface web via Streamlit.

---

## Visão Geral

O projeto transforma um dataset multi-label de 28 emoções em um classificador de 7 macro-categorias, treinado com TF-IDF + LinearSVC e disponibilizado como aplicação web interativa.

```
Dataset bruto (multi-label, desbalanceado)
        ↓ Script 1 — Balanceamento
Dataset balanceado (~51k registros)
        ↓ Script 2 — prepare_data.py
Treino (~40k) + Teste (~10k) com 7 macro-emoções
        ↓ Script 3 — train.py
Modelos versionados no MLflow Registry (DagsHub)
        ↓ Script 4 — evaluate.py
model.pkl → app Streamlit
```

---

## Pipeline em 4 Scripts

### Script 1 — Pré-processamento e Balanceamento (~70s)

Corrige o forte desbalanceamento do dataset original usando undersampling nas classes majoritárias e oversampling nas minoritárias. Como o problema é multi-label, identifica a emoção mais rara de cada texto antes de decidir o descarte, preservando amostras valiosas.

| Classe | Antes | Depois |
|--------|-------|--------|
| neutro (majoritária) | 31.000 | 23.000 |
| luto (minoritária) | 560 | 4.000+ |
| **Total** | — | **~51.000** |

**Saída:** `data/AnaliseDeEmocoes_PT-br_balanceado.csv`

---

### Script 2 — Preparação dos Dados ([src/prepare_data.py](src/prepare_data.py)) (~80s)

Aplica limpeza textual e redução dimensional ao dataset balanceado.

**Limpeza textual:**
- Conversão para minúsculas
- Remoção de pontuação e números via regex
- Remoção de stopwords (lista customizada com 200+ palavras em português)

**Redução dimensional — 28 emoções → 7 macro-classes:**

| Macro-classe | Emoções incluídas |
|---|---|
| `Afeto_Alegria` | amor, alegria, diversao, carinho, empolgacao |
| `Aprovacao_Admiracao` | aprovacao, admiracao, gratidao, orgulho, otimismo, alivio |
| `Raiva_Desaprovacao` | raiva, irritacao, desaprovacao, nojo |
| `Tristeza_Ansiedade` | tristeza, decepcao, luto, remorso, medo, vergonha, nervosismo |
| `Surpresa_Curiosidade` | surpresa, curiosidade, confusao |
| `Neutro_Percepcao` | neutro, percepcao |
| `Desejo` | desejo |

**Divisão estratificada:** 80% treino (~40k) · 20% teste (~10k)

```bash
python -m src.prepare_data
```

---

### Script 3 — Treinamento e Versionamento ([src/train.py](src/train.py)) (~85s)

Treina e rastreia experimentos via MLflow integrado ao DagsHub.

**Pipeline de ML:**
- `TfidfVectorizer` — normalização de acentos, frequência sublinear, `min_df=2`, `max_df=0.90`, `max_features=None`
- `CalibratedClassifierCV(LinearSVC)` — `class_weight="balanced"`, validação cruzada 5-fold para calibração de probabilidades

**Experimentos executados:**

| Experimento | ngram_max | C | Acurácia | F1 (weighted) |
|---|---|---|---|---|
| exp-1-baseline | 2 | 0.1 | — | — |
| **exp-2-bigramas** | **2** | **1.0** | **71%** | **0.71** |
| exp-3-trigramas | 3 | 0.5 | — | — |

O experimento 2 venceu: bigramas capturam contexto emocional suficiente; trigramas não trouxeram ganhos adicionais.

```bash
cd src
python train.py
```

---

### Script 4 — Avaliação e Produção ([src/evaluate.py](src/evaluate.py)) (~60s)

Carrega o modelo promovido para produção no MLflow Registry (`alias: production`), avalia no conjunto de teste original e serializa localmente.

```bash
python -m src.evaluate
# → src/model.pkl
```

O `model.pkl` é consumido diretamente pela aplicação Streamlit.

---

## Aplicação Web ([app.py](app.py))

Interface Streamlit com duas funcionalidades:

- **Análise individual** — cola um texto e vê a emoção detectada com probabilidades por classe
- **Análise em lote** — upload de CSV com coluna `texto`, classificação e download do resultado

```bash
streamlit run app.py
```

---

## Configuração

### Pré-requisitos

- Python 3.11+
- Conta no [DagsHub](https://dagshub.com) com repositório criado

### Instalação

```bash
pip install -r requirements.txt
```

### Variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
DAGSHUB_USER=seu_usuario
DAGSHUB_REPO=nome_do_repositorio
DAGSHUB_TOKEN=seu_token_de_acesso
```

> Obtenha o token em: DagsHub → User Settings → Tokens → New Token

---

## Estrutura do Projeto

```
emocional-app/
├── app.py                  # Interface Streamlit
├── Dockerfile              # Build com download automático do modelo
├── render.yaml             # Força uso do Docker no Render (evita detecção como Go)
├── requirements.txt
├── .env                    # Credenciais (não versionado)
├── data/
│   ├── AnaliseDeEmocoes_PT-br.csv            # Dataset original
│   └── AnaliseDeEmocoes_PT-br_balanceado.csv # Dataset balanceado (Script 1)
└── src/
    ├── prepare_data.py     # Script 2 — limpeza e divisão
    ├── train.py            # Script 3 — treinamento e MLflow
    ├── evaluate.py         # Script 4 — avaliação e geração do model.pkl
    └── model.pkl           # Modelo serializado (gerado pelo evaluate.py)
```

---

## Deploy

### Render (produção)

O arquivo [`render.yaml`](render.yaml) na raiz do repositório instrui o Render a usar o **Dockerfile** Python — sem ele, o Render detecta incorretamente o projeto como Go.

**Passos:**

1. Conecte o repositório `priscilagon/emocoes-app` no [Render Dashboard](https://dashboard.render.com)
2. O Render detecta o `render.yaml` automaticamente e configura o serviço como Docker
3. Acesse **Environment** no painel do serviço e adicione as três variáveis secretas:

| Variável | Valor |
|---|---|
| `DAGSHUB_USER` | seu usuário no DagsHub |
| `DAGSHUB_REPO` | nome do repositório no DagsHub |
| `DAGSHUB_TOKEN` | token de acesso do DagsHub |

4. Clique em **Manual Deploy → Deploy latest commit**

> Durante o build, o Render passa essas variáveis como Docker build args automaticamente. O `Dockerfile` usa esses args para rodar `python -m src.evaluate`, que baixa o modelo do DagsHub e gera o `src/model.pkl` dentro da imagem.

**URL de produção:** https://emocoes-app.onrender.com

> O plano gratuito do Render hiberna após inatividade — a primeira requisição pode demorar até 50 segundos para o serviço acordar.

---

### Docker local

```bash
docker build \
  --build-arg DAGSHUB_USER=seu_usuario \
  --build-arg DAGSHUB_REPO=nome_do_repositorio \
  --build-arg DAGSHUB_TOKEN=seu_token \
  -t emocional-app .

docker run -p 8501:8501 emocional-app
```

---

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.11 |
| ML | scikit-learn (TF-IDF + LinearSVC) |
| Rastreamento | MLflow + DagsHub |
| Interface | Streamlit |
| Containerização | Docker |
| Deploy | Render |
