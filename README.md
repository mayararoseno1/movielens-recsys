# 🎬 MovieLens RecSys

Sistema de recomendação de filmes construído com MongoDB e Machine Learning.
Projeto de estudo progressivo cobrindo os principais conceitos cobrados em entrevistas técnicas de ML.

## Como rodar

```bash
pip install -r requirements.txt
python ingest.py   # baixa o dataset e popula o MongoDB
python eda.py      # análise exploratória
```

## Requisitos

- Python 3.9+
- MongoDB rodando localmente na porta 27017