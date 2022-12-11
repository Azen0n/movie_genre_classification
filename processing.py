import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from typing import TypedDict

import numpy as np
import pandas as pd
from nltk import word_tokenize, pos_tag, WordNetLemmatizer
from nltk.corpus import stopwords as sw

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
    return process_movie_overview(' '.join(overviews))


def get_all_movies_with_genre(genre: str, movies: list[Movie]) -> list[Movie]:
    """Filter movies by genre."""
    filtered_movies = filter(lambda movie: genre in movie.genre_names, movies)
    return list(filtered_movies)


def process_movie_overview(overview: str) -> list[str]:
    """Remove stopwords and split overview in words"""
    processed_overview = []
    words = word_tokenize(overview)
    for word, tag in pos_tag(words):
        if not word.isalpha() or word in stopwords:
            continue
        if tag[0].lower() in ['a', 'r', 'n', 'v']:
            lemma = lemmatizer.lemmatize(word.lower(), tag[0].lower())
            processed_overview.append(lemma)
    return processed_overview


def get_genres(movies: list[Movie]) -> list[str]:
    """Return list of unique genres."""
    genre_frequency = count_genre_frequency(movies)
    return list(genre_frequency.keys())


def get_classification_dataframe(movies: list[Movie]) -> pd.DataFrame:
    """Transform list of movies to dataframe with overviews
    and genre matrix.
    """
    genres = {genre: 0 for genre in get_genres(movies)}
    overview_genre_matrix = []
    for movie in movies:
        row = deepcopy(genres)
        row['overview'] = ' '.join(process_movie_overview(movie.overview))
        for genre in movie.genres:
            row[genre['name']] = 1
        overview_genre_matrix.append(row)
    return pd.DataFrame(overview_genre_matrix)
