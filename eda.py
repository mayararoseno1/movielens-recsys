import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "movielens"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

print("=== Visão Geral ===")
print(f"Filmes:   {db.movies.count_documents({})}")
print(f"Usuários: {db.users.count_documents({})}")
print(f"Ratings:  {db.ratings.count_documents({})}")

ratings = pd.DataFrame(db.ratings.find({}, {"_id":0,"rating":1}))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ratings["rating"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], color="#7F77DD"
)
axes[0].set_title("Distribuição de ratings")
axes[0].set_xlabel("Nota")
axes[0].set_ylabel("Quantidade")

ratings_per_user = pd.DataFrame(
    db.ratings.aggregate([
        {"$group": {"_id": "$userId", "count": {"$sum": 1}}}
    ])
)
ratings_per_user["count"].hist(bins=50, ax=axes[1], color="#1D9E75")
axes[1].set_title("Ratings por usuário (sparsity)")
axes[1].set_xlabel("Nº de ratings")
axes[1].set_ylabel("Nº de usuários")

plt.tight_layout()
plt.savefig("distribuicoes.png", dpi=120)
print("Salvo: distribuicoes.png")

print("\n=== Top 10 filmes mais avaliados ===")
pipeline = [
    {"$group": {"_id": "$movieId", "count": {"$sum": 1}, "avg": {"$avg": "$rating"}}},
    {"$sort": {"count": -1}},
    {"$limit": 10},
    {"$lookup": {
        "from": "movies",
        "localField": "_id",
        "foreignField": "movieId",
        "as": "movie"
    }},
    {"$unwind": "$movie"},
    {"$project": {"title": "$movie.title", "count": 1, "avg": {"$round": ["$avg", 2]}}}
]
top10 = pd.DataFrame(db.ratings.aggregate(pipeline))
print(top10[["title","count","avg"]].to_string(index=False))

n_users  = db.users.count_documents({})
n_movies = db.movies.count_documents({})
n_ratings = db.ratings.count_documents({})
sparsity = 1 - (n_ratings / (n_users * n_movies))
print(f"\nSparsity da matriz usuário-filme: {sparsity:.2%}")
print("Isso significa que a matriz tem apenas"
      f" {100-sparsity*100:.2f}% dos valores preenchidos.")