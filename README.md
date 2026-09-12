# Movie Recommender System

A content-based movie recommender built on the TMDB 5000 dataset. Pick a film and it
returns the 10 most similar titles with their posters, served as a Streamlit web app.

## How it works

Each movie is reduced to a bag of words drawn from its overview, genres, keywords, top
three cast members and director. Those tags are vectorised with `CountVectorizer`
(5,000 features) and every pair of movies is compared by cosine similarity. A
recommendation is simply the nearest neighbours of the film you picked.

This is *content-based*, not collaborative filtering: it knows nothing about users or
ratings, only about what each film is made of.

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Download the data**

Grab the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
from Kaggle and place both CSVs in a `dataset/` folder:

```
dataset/tmdb_5000_movies.csv
dataset/tmdb_5000_credits.csv
```

**3. Build the model**

```bash
python build_model.py
```

This writes `model/movie_list.pkl` and `model/similarity.pkl`. It takes about a minute.

The model files are not committed to this repo: the similarity matrix is 88 MB, which is
too large for git and trivial to regenerate from the script above.

**4. Run the app**

```bash
streamlit run app.py
```

## Configuration

Posters are fetched from the TMDB API. A public demo key is used by default; to use your
own, get one free from [TMDB](https://www.themoviedb.org/settings/api) and set:

```bash
export TMDB_API_KEY=your_key_here
```

Recommendation count is controlled by `TOP_N` at the top of `app.py` (default 10),
laid out `PER_ROW` at a time.

## Project structure

```
build_model.py    # one-off: builds the similarity matrix from the CSVs
app.py            # the Streamlit app
requirements.txt
dataset/          # (gitignored) raw Kaggle CSVs
model/            # (gitignored) generated pickles
```

## Limitations

Recommendations past roughly the fifth result get noticeably weaker — similarity scores
flatten out, so distant neighbours share only generic tags. The bag-of-words approach
also ignores word order and treats every tag as equally important, so a shared director
counts the same as a shared genre.
