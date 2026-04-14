# 🎬 MovieLens RecSys

Sistema de recomendação de filmes construído com MongoDB e Machine Learning.
Projeto de estudo progressivo cobrindo os principais conceitos cobrados em entrevistas técnicas de ML.

## Fases do projeto

- [x] **Fase 1** — Modelagem no MongoDB + ingestão do MovieLens + EDA
- [ ] **Fase 2** — Filtragem colaborativa (User-Based CF + similaridade de cosseno)
- [ ] **Fase 3** — Matrix Factorization com SVD
- [ ] **Fase 4** — API com FastAPI + MongoDB Atlas Vector Search

## Como rodar

```bash
pip install -r requirements.txt
python ingest.py   # baixa o dataset e popula o MongoDB
python eda.py      # análise exploratória
```

## Requisitos

- Python 3.9+
- MongoDB rodando localmente na porta 27017