import pandas as pd
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "movielens"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def generate_embeddings():
   
    movies = list(db.movies.find({}, {"_id":0}))
    print(f"{len(movies)} filmes carregados")

    def make_description(m):
        genres = " ".join(m.get("genres", []))
        return f"{m['title']} — {genres}"

    descriptions = [make_description(m) for m in movies]

    print("Carregando modelo de embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # modelo leve, 384 dimensões

    print("Gerando embeddings (pode demorar ~1 min)...")
    embeddings = model.encode(descriptions, show_progress_bar=True)

    print("Salvando embeddings no MongoDB...")
    for movie, embedding in zip(movies, embeddings):
        db.movies.update_one(
            {"movieId": movie["movieId"]},
            {"$set": {"embedding": embedding.tolist()}}
        )

    print("Embeddings salvos!")

def search_by_description(query: str, top_k: int = 5):
    """
    Busca filmes por similaridade semântica com a query.
    Sem Vector Search nativo — faz o cálculo em Python mesmo (funciona para escala pequena).
    """
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])[0]

    movies = list(db.movies.find(
        {"embedding": {"$exists": True}},
        {"_id":0, "movieId":1, "title":1, "genres":1, "embedding":1}
    ))

    scores = []
    for m in movies:
        emb = np.array(m["embedding"])
        score = np.dot(query_embedding, emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(emb)
        )
        scores.append((m["title"], m["genres"], float(score)))

    scores.sort(key=lambda x: x[2], reverse=True)

    print(f"\nResultados para: '{query}'")
    for title, genres, score in scores[:top_k]:
        print(f"  {score:.3f}  {title}  {genres}")

if __name__ == "__main__":
    generate_embeddings()

    search_by_description("intense war drama")
    search_by_description("funny animated movie for kids")
    search_by_description("romantic comedy")