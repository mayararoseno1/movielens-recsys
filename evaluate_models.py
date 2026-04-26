import numpy as np
import pandas as pd
from pymongo import MongoClient
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "movielens"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def load_data():
    docs = list(db.ratings.find({}, {"_id":0,"userId":1,"movieId":1,"rating":1}))
    return pd.DataFrame(docs)

def evaluate_cf(df, sample_users=50):
    print("Avaliando User-Based CF...")
    matrix = df.pivot_table(index="userId", columns="movieId", values="rating")
    filled = matrix.fillna(0)
    sim = cosine_similarity(filled)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

    errors = []
    for user_id in matrix.index[:sample_users]:
        rated = matrix.loc[user_id].dropna()
        if len(rated) < 5:
            continue
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
                if pd.isna(rating): continue
                weighted_sum += sim_score * rating
                sim_sum      += abs(sim_score)
            if sim_sum > 0:
                errors.append(((weighted_sum/sim_sum) - actual) ** 2)

    return round(np.sqrt(np.mean(errors)), 4)

def evaluate_svd(df):
    print("Avaliando SVD...")
    matrix = df.pivot_table(index="userId", columns="movieId", values="rating")

    # Preenche NaN com a média do próprio usuário (não com 0)
    user_means = matrix.mean(axis=1)
    matrix_filled = matrix.apply(lambda row: row.fillna(user_means[row.name]), axis=1)

    user_means_arr = user_means.values.reshape(-1, 1)
    matrix_norm = matrix_filled.values - user_means_arr
    sparse_matrix = csr_matrix(matrix_norm)

    U, sigma, Vt = svds(sparse_matrix, k=50)
    predicted = np.dot(np.dot(U, np.diag(sigma)), Vt) + user_means_arr
    predicted_df = pd.DataFrame(predicted, index=matrix.index, columns=matrix.columns)

    errors = []
    for _, row in df.iterrows():
        uid, mid, actual = row["userId"], row["movieId"], row["rating"]
        if uid in predicted_df.index and mid in predicted_df.columns:
            errors.append((predicted_df.loc[uid, mid] - actual) ** 2)

    return round(np.sqrt(np.mean(errors)), 4)


if __name__ == "__main__":
    df = load_data()

    rmse_cf  = evaluate_cf(df)
    rmse_svd = evaluate_svd(df)

    print("\n" + "="*50)
    print("       COMPARAÇÃO DE MODELOS")
    print("="*50)
    print(f"  User-Based CF  →  RMSE: {rmse_cf}")
    print(f"  SVD            →  RMSE: {rmse_svd}")
    print("="*50)

    melhora = round((rmse_cf - rmse_svd) / rmse_cf * 100, 1)
    if rmse_svd < rmse_cf:
        print(f"\n  SVD é {melhora}% melhor que o CF baseline!")
    else:
        print(f"\n  CF ainda está na frente — tente aumentar o k do SVD.")