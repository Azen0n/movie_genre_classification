import json
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import reduce
from typing import TypedDict

import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords as sw
from nltk.stem import WordNetLemmatizer

from plot import plot_genre_frequency, plot_number_of_genres, plot_wordcloud

warnings.filterwarnings('ignore', category=DeprecationWarning)

lemmatizer = WordNetLemmatizer()
stopwords = sw.words('english')


class Genre(TypedDict):
    id: str
    name: str


@dataclass
class Movie:
    genres: list[Genre]
    overview: str

    @property
    def genre_names(self):
        return [genre['name'] for genre in self.genres]


def drop_empty(df: pd.DataFrame) -> pd.DataFrame:
    """Remove all empty values in 'genres' and 'overview' columns."""
    df['genres'].replace('[]', np.nan, inplace=True)
    df['overview'].replace(r' ', np.nan, inplace=True)
    df = df[df['genres'].notna()]
    df = df[df['overview'].notna()]
    return df


def get_all_genres(series: pd.Series) -> list[str]:
    all_genres = []
    for genres in series:
        genres = json.loads(genres)
        for genre in genres:
            if genre['name'] not in all_genres:
                all_genres.append(genre['name'])
    return all_genres


def count_genre_frequency(movies: list[Movie]) -> dict[str, int]:
    """Return dict with genre frequency."""
    genres_of_movies = [movie.genre_names for movie in movies]
    genre_frequency = dict(Counter(reduce(lambda g1, g2: g1 + g2,
                                          genres_of_movies)))
    return genre_frequency


def count_number_of_genres_per_movie(movies: list[Movie]) -> dict[int, int]:
    """Return dict with number of movies with certain number of genres."""
    number_of_genres = [len(movie.genres) for movie in movies]
    return dict(Counter(number_of_genres))


def get_all_words_from_overviews_of_genre(genre: str, movies: list[Movie]):
    """Return list of processed words from all movie overviews with genre."""
    filtered_movies = get_all_movies_with_genre(genre, movies)
    if not filtered_movies:
        return []
    overviews = [movie.overview for movie in filtered_movies]
    return process_movie_overviews(overviews)


def get_all_movies_with_genre(genre: str, movies: list[Movie]) -> list[Movie]:
    """Filter movies by genre."""
    filtered_movies = filter(lambda movie: genre in movie.genre_names, movies)
    return list(filtered_movies)


def process_movie_overviews(overviews: list[str]) -> list[str]:
    """Remove stopwords and split overviews in words."""
    words = word_tokenize(' '.join(overviews))
    lemmatized_words = []
    for word, tag in pos_tag(words):
        if not word.isalpha() or word in stopwords:
            continue
        if tag[0].lower() in ['a', 'r', 'n', 'v']:
            lemma = lemmatizer.lemmatize(word.lower(), tag[0].lower())
            lemmatized_words.append(lemma)
    return lemmatized_words


def main():
    df = pd.read_csv('tmdb_5000_movies.csv')
    df = drop_empty(df)

    movies = [Movie(json.loads(genres), overview)
              for genres, overview in zip(df['genres'], df['overview'])]

    genre_frequency = count_genre_frequency(movies)
    plot_genre_frequency(genre_frequency)

    number_of_genres_per_movie = count_number_of_genres_per_movie(movies)
    plot_number_of_genres(number_of_genres_per_movie)

    genre = 'Drama'
    words = get_all_words_from_overviews_of_genre(genre, movies)
    plot_wordcloud(genre, words)


if __name__ == '__main__':
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    main()
