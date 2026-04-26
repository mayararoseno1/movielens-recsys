import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pymongo import MongoClient
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "movielens"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def load_ratings():
    docs = list(db.ratings.find({}, {"_id":0, "userId":1, "movieId":1, "rating":1}))
    df = pd.DataFrame(docs)
    print(f"Ratings carregados: {len(df)}")
    return df


def train_svd(df, k=50):
    matrix = df.pivot_table(index="userId", columns="movieId", values="rating")

    user_ids  = matrix.index.tolist()
    movie_ids = matrix.columns.tolist()

    user_means = matrix.mean(axis=1)

    matrix_filled = matrix.apply(lambda row: row.fillna(user_means[row.name]), axis=1)

    matrix_values = matrix_filled.values
    user_means_arr = user_means.values.reshape(-1, 1)

    matrix_norm = matrix_values - user_means_arr

    sparse_matrix = csr_matrix(matrix_norm)

    print(f"\nAplicando SVD com k={k} fatores latentes...")
    U, sigma, Vt = svds(sparse_matrix, k=k)

    predicted_norm = np.dot(np.dot(U, np.diag(sigma)), Vt)
    predicted = predicted_norm + user_means_arr

    predicted_df = pd.DataFrame(predicted, index=user_ids, columns=movie_ids)
    print("SVD concluído!")

    return predicted_df, matrix, user_ids, movie_ids

def evaluate(df, predicted_df):
    print("\nCalculando RMSE...")
    errors = []
    for _, row in df.iterrows():
        uid, mid, actual = row["userId"], row["movieId"], row["rating"]
        if uid in predicted_df.index and mid in predicted_df.columns:
            predicted = predicted_df.loc[uid, mid]
            errors.append((predicted - actual) ** 2)

    rmse = round(np.sqrt(np.mean(errors)), 4)
    print(f"RMSE: {rmse}")
    return rmse

def save_recommendations(df, predicted_df, n_recs=10):
    movies = {
        m["movieId"]: m["title"]
        for m in db.movies.find({}, {"_id":0, "movieId":1, "title":1})
    }

    print(f"\nGerando recomendações para {predicted_df.shape[0]} usuários...")
    db.recommendations.delete_many({"model": "svd"})

    docs_to_insert = []

    for user_id in predicted_df.index:
      
        already_rated = set(df[df["userId"] == user_id]["movieId"])

        user_preds = predicted_df.loc[user_id]
        candidates = user_preds[~user_preds.index.isin(already_rated)]
        top = candidates.nlargest(n_recs)

        doc = {
            "userId": int(user_id),
            "model": "svd",
            "generatedAt": datetime.now(timezone.utc),
            "recommendations": [
                {
                    "movieId": int(mid),
                    "title": movies.get(mid, "?"),
                    "score": round(float(score), 4)
                }
                for mid, score in top.items()
            ]
        }
        docs_to_insert.append(doc)

    db.recommendations.insert_many(docs_to_insert)
    db.recommendations.create_index([("userId", 1), ("model", 1)])
    print(f"{len(docs_to_insert)} documentos salvos na collection 'recommendations'!")

# --- 5. Busca recomendações de um usuário ---
def get_recommendations(user_id):
    doc = db.recommendations.find_one(
        {"userId": user_id, "model": "svd"},
        {"_id": 0, "recommendations": 1}
    )
    if not doc:
        print(f"Nenhuma recomendação encontrada para usuário {user_id}")
        return

    print(f"\nTop recomendações SVD para usuário {user_id}:")
    for i, r in enumerate(doc["recommendations"], 1):
        print(f"  {i:2}. {r['title']:<45} score: {r['score']}")

if __name__ == "__main__":
    df = load_ratings()
    predicted_df, matrix, user_ids, movie_ids = train_svd(df, k=50)
    rmse = evaluate(df, predicted_df)
    save_recommendations(df, predicted_df)
    get_recommendations(user_id=1)