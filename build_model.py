"""Build the model files that seond_code.py loads.

Reads the two TMDB CSVs from dataset/ and writes model/movie_list.pkl and
model/similarity.pkl next to this script. Run this once before the app.
"""
import ast
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
MODEL_DIR = HERE / "model"


def names(text, limit=None):
    """Pull the 'name' field out of TMDB's stringified list-of-dicts columns."""
    out = [d["name"] for d in ast.literal_eval(text)]
    return out[:limit] if limit else out


def director(text):
    return [d["name"] for d in ast.literal_eval(text) if d["job"] == "Director"]


def main():
    movies = pd.read_csv(DATASET / "tmdb_5000_movies.csv")
    credits = pd.read_csv(DATASET / "tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")
    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    movies = movies.dropna()

    movies["genres"] = movies["genres"].apply(names)
    movies["keywords"] = movies["keywords"].apply(names)
    movies["cast"] = movies["cast"].apply(lambda t: names(t, limit=3))
    movies["crew"] = movies["crew"].apply(director)
    movies["overview"] = movies["overview"].apply(lambda t: t.split())

    # "Sam Worthington" -> "SamWorthington" so first names don't collide
    for col in ["genres", "keywords", "cast", "crew"]:
        movies[col] = movies[col].apply(lambda lst: [s.replace(" ", "") for s in lst])

    movies["tags"] = (
        movies["overview"] + movies["genres"] + movies["keywords"]
        + movies["cast"] + movies["crew"]
    )

    new = movies[["movie_id", "title", "tags"]].copy()
    new["tags"] = new["tags"].apply(lambda lst: " ".join(lst).lower())

    vectors = CountVectorizer(max_features=5000, stop_words="english").fit_transform(
        new["tags"]
    ).toarray()

    # float32 halves the file (~185MB -> ~92MB) with no effect on ranking
    similarity = cosine_similarity(vectors).astype("float32")

    MODEL_DIR.mkdir(exist_ok=True)
    pickle.dump(new, open(MODEL_DIR / "movie_list.pkl", "wb"))
    pickle.dump(similarity, open(MODEL_DIR / "similarity.pkl", "wb"))

    print(f"movies: {new.shape}, similarity: {similarity.shape}")
    print(f"wrote {MODEL_DIR}/movie_list.pkl and similarity.pkl")


if __name__ == "__main__":
    main()
