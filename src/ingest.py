import os
import zipfile
import requests
import pandas as pd
from pymongo import MongoClient, ASCENDING

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "movielens"
DATA_DIR  = "./data"
URL       = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ml-100k.zip")

    if not os.path.exists(zip_path):
        print("Baixando MovieLens ml-100k...")
        r = requests.get(URL, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download concluído.")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    print("Extração concluída.")

def load_movies(db):
    genre_cols = [
        "unknown","Action","Adventure","Animation","Children's","Comedy",
        "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",
        "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"
    ]
    cols = ["movieId","title","release_date","video_date","imdb_url"] + genre_cols

    df = pd.read_csv(
        f"{DATA_DIR}/ml-100k/u.item",
        sep="|", encoding="latin-1", header=None, names=cols
    )

    df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype("Int64")

    df["genres"] = df[genre_cols].apply(
        lambda row: [g for g, v in zip(genre_cols, row) if v == 1], axis=1
    )

    docs = df[["movieId","title","year","genres"]].to_dict("records")

    col = db["movies"]
    col.drop()
    col.insert_many(docs)
    col.create_index([("movieId", ASCENDING)], unique=True)
    print(f"  {len(docs)} filmes inseridos.")

def load_ratings(db):
    df = pd.read_csv(
        f"{DATA_DIR}/ml-100k/u.data",
        sep="\t", header=None,
        names=["userId","movieId","rating","timestamp"]
    )

    docs = df.to_dict("records")

    col = db["ratings"]
    col.drop()
    col.insert_many(docs)
    col.create_index([("userId", ASCENDING)])
    col.create_index([("movieId", ASCENDING)])
    print(f"  {len(docs)} ratings inseridos.")

def load_users(db):
    occupation_map = {}
    with open(f"{DATA_DIR}/ml-100k/u.occupation") as f:
        for i, line in enumerate(f):
            occupation_map[i] = line.strip()

    df = pd.read_csv(
        f"{DATA_DIR}/ml-100k/u.user",
        sep="|", header=None,
        names=["userId","age","gender","occupation","zip_code"]
    )
    df["occupation"] = df["occupation"].map(occupation_map)

    docs = df.to_dict("records")

    col = db["users"]
    col.drop()
    col.insert_many(docs)
    col.create_index([("userId", ASCENDING)], unique=True)
    print(f"  {len(docs)} usuários inseridos.")

if __name__ == "__main__":
    download_dataset()
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    print("Populando MongoDB...")
    load_movies(db)
    load_ratings(db)
    load_users(db)
    print("Pronto!")