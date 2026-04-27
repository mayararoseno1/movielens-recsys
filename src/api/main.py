from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from bson import ObjectId
import json

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "movielens"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

app = FastAPI(title="MovieLens RecSys API", version="1.0")

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int, model: str = "svd"):
    """
    Serve recomendações pré-computadas do MongoDB.
    Latência ~1ms — não roda nenhum modelo em tempo real.
    """
    doc = db.recommendations.find_one(
        {"userId": user_id, "model": model},
        {"_id": 0}
    )
    if not doc:
        raise HTTPException(status_code=404, detail=f"Usuário {user_id} não encontrado")

    return {
        "userId": user_id,
        "model": model,
        "generatedAt": str(doc["generatedAt"]),
        "recommendations": doc["recommendations"]
    }

@app.get("/movies/search")
def search_movies(q: str, limit: int = 10):
    """
    Busca filmes por título (case-insensitive).
    """
    results = list(db.movies.find(
        {"title": {"$regex": q, "$options": "i"}},
        {"_id": 0, "movieId": 1, "title": 1, "year": 1, "genres": 1}
    ).limit(limit))

    if not results:
        raise HTTPException(status_code=404, detail="Nenhum filme encontrado")

    return {"query": q, "results": results}

@app.get("/movies/{movie_id}")
def get_movie(movie_id: int):
    """
    Retorna detalhes de um filme e seus ratings agregados.
    """
    movie = db.movies.find_one({"movieId": movie_id}, {"_id": 0})
    if not movie:
        raise HTTPException(status_code=404, detail="Filme não encontrado")

    stats = list(db.ratings.aggregate([
        {"$match": {"movieId": movie_id}},
        {"$group": {
            "_id": None,
            "avg_rating": {"$avg": "$rating"},
            "total_ratings": {"$sum": 1}
        }}
    ]))

    if stats:
        movie["avg_rating"]    = round(stats[0]["avg_rating"], 2)
        movie["total_ratings"] = stats[0]["total_ratings"]

    return movie

@app.get("/movies/{movie_id}/similar")
def similar_movies(movie_id: int, limit: int = 10):
    """
    Retorna filmes com gêneros similares.
    Baseline simples — na Fase 4b vamos substituir por Vector Search.
    """
    movie = db.movies.find_one({"movieId": movie_id}, {"_id": 0})
    if not movie:
        raise HTTPException(status_code=404, detail="Filme não encontrado")

    genres = movie.get("genres", [])
    if not genres:
        raise HTTPException(status_code=400, detail="Filme sem gêneros cadastrados")

    similar = list(db.movies.find(
        {
            "movieId": {"$ne": movie_id},
            "genres": {"$in": genres}
        },
        {"_id": 0, "movieId": 1, "title": 1, "year": 1, "genres": 1}
    ).limit(limit))

    return {
        "movie": movie["title"],
        "based_on_genres": genres,
        "similar": similar
    }

@app.get("/stats")
def stats():
    return {
        "movies":  db.movies.count_documents({}),
        "users":   db.users.count_documents({}),
        "ratings": db.ratings.count_documents({}),
        "recommendations_computed": db.recommendations.count_documents({})
    }