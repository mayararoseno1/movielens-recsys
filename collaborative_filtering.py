import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "movielens"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def load_matrix():
    docs = list(db.ratings.find({}, {"_id":0, "userId":1, "movieId":1, "rating":1}))
    df = pd.DataFrame(docs)

    matrix = df.pivot_table(index="userId", columns="movieId", values="rating")
    return matrix

def compute_similarity(matrix):
    filled = matrix.fillna(0)

    sim = cosine_similarity(filled)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
    return sim_df

def recommend(user_id, matrix, sim_df, n_similar=10, n_recs=10, min_support=3): # <-- MUDA AQUI: adiciona min_support
    similar_users = (
        sim_df[user_id]
        .drop(index=user_id)
        .sort_values(ascending=False)
        .head(n_similar)
    )

    already_rated = set(matrix.loc[user_id].dropna().index)

    scores = {}
    for sim_user, sim_score in similar_users.items():
        user_ratings = matrix.loc[sim_user].dropna()
        for movie_id, rating in user_ratings.items():
            if movie_id in already_rated:
                continue
            if movie_id not in scores:
                scores[movie_id] = {"weighted_sum": 0, "sim_sum": 0, "count": 0} 
            scores[movie_id]["weighted_sum"] += sim_score * rating
            scores[movie_id]["sim_sum"]      += abs(sim_score)
            scores[movie_id]["count"]        += 1  

    recs = {
        mid: s["weighted_sum"] / s["sim_sum"]
        for mid, s in scores.items()
        if s["sim_sum"] > 0 and s["count"] >= min_support  
    }


    top = sorted(recs.items(), key=lambda x: x[1], reverse=True)[:n_recs]

    movie_ids = [mid for mid, _ in top]
    movies = {
        m["movieId"]: m["title"]
        for m in db.movies.find({"movieId": {"$in": movie_ids}}, {"_id":0})
    }

    results = [
        {"title": movies.get(mid, "?"), "score": round(score, 2)}
        for mid, score in top
    ]
    return pd.DataFrame(results)

def evaluate(matrix, sim_df, sample_users=50):
    errors = []
    users = matrix.index[:sample_users]

    for user_id in users:
        rated = matrix.loc[user_id].dropna()
        if len(rated) < 5:
            continue

        # Separa 20% dos ratings desse usuário como "teste"
        test = rated.sample(frac=0.2, random_state=42)
        train_matrix = matrix.copy()
        train_matrix.loc[user_id, test.index] = np.nan

        similar_users = (
            sim_df[user_id]
            .drop(index=user_id)
            .sort_values(ascending=False)
            .head(10)
        )

        for movie_id, actual in test.items():
            weighted_sum, sim_sum = 0, 0
            for sim_user, sim_score in similar_users.items():
                rating = train_matrix.loc[sim_user, movie_id]
                if pd.isna(rating):
                    continue
                weighted_sum += sim_score * rating
                sim_sum      += abs(sim_score)
            if sim_sum > 0:
                predicted = weighted_sum / sim_sum
                errors.append((predicted - actual) ** 2)

    rmse = np.sqrt(np.mean(errors))
    return round(rmse, 4)

# --- Execução ---
if __name__ == "__main__":
    print("Carregando matriz usuário x filme...")
    matrix = load_matrix()
    print(f"Matriz: {matrix.shape[0]} usuários x {matrix.shape[1]} filmes")

    print("Calculando similaridades...")
    sim_df = compute_similarity(matrix)

    USER_ID = 1
    print(f"\nTop 10 recomendações para usuário {USER_ID}:")
    recs = recommend(USER_ID, matrix, sim_df)
    print(recs.to_string(index=False))

    print("\nAvaliando modelo (RMSE)...")
    rmse = evaluate(matrix, sim_df)
    print(f"RMSE: {rmse}")
    print("(Quanto menor, melhor. Escala de 1-5 igual aos ratings)")