import os
import pickle
from pathlib import Path

import streamlit as st
import requests

# how many recommendations to show, and how many per row
TOP_N = 10
PER_ROW = 5


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    api_key = os.getenv("TMDB_API_KEY", "8265bd1679663a7ea12ac168da84d2e8")
    url = ("https://api.themoviedb.org/3/movie/{}"
           "?api_key={}&language=en-US").format(movie_id, api_key)
    try:
        data = requests.get(url, timeout=10).json()
    except requests.RequestException:
        return None
    poster_path = data.get('poster_path')
    if not poster_path:          # some movies have no poster on TMDB
        return None
    return "https://image.tmdb.org/t/p/w500/" + poster_path

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:TOP_N + 1]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters


st.header('Movie Recommender System')
MODEL_DIR = Path(__file__).resolve().parent / 'model'
movies = pickle.load(open(MODEL_DIR / 'movie_list.pkl','rb'))
similarity = pickle.load(open(MODEL_DIR / 'similarity.pkl','rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    # lay the results out PER_ROW at a time
    for offset in range(0, len(recommended_movie_names), PER_ROW):
        names = recommended_movie_names[offset:offset + PER_ROW]
        posters = recommended_movie_posters[offset:offset + PER_ROW]
        for col, name, poster in zip(st.columns(PER_ROW), names, posters):
            with col:
                st.text(name)
                if poster:
                    st.image(poster)
                else:
                    st.caption("no poster")
