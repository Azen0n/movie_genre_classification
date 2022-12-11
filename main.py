import json

import nltk
import pandas as pd

from classification import test_classification
from plot import plot_genre_frequency, plot_number_of_genres, plot_wordcloud
from processing import (Movie, drop_empty, count_genre_frequency,
                        count_number_of_genres_per_movie,
                        get_all_words_from_overviews_of_genre, get_classification_dataframe)


def visualize_movies(movies: list[Movie], genre: str):
    """Plot genre frequency, number of genres per movie and word cloud of
    overviews with given genre.
    """
    genre_frequency = count_genre_frequency(movies)
    plot_genre_frequency(genre_frequency)

    number_of_genres_per_movie = count_number_of_genres_per_movie(movies)
    plot_number_of_genres(number_of_genres_per_movie)

    words = get_all_words_from_overviews_of_genre(genre, movies)
    plot_wordcloud(genre, words)


def save_overview_genre_matrix(movies: list[Movie]):
    """Save movie overview and genre matrix to csv."""
    overview_genre_matrix = get_classification_dataframe(movies)
    overview_genre_matrix.to_csv('overview_genre_matrix.csv', index=False)


def main():
    df = pd.read_csv('tmdb_5000_movies.csv')
    df = drop_empty(df)

    movies = [Movie(json.loads(genres), overview)
              for genres, overview in zip(df['genres'], df['overview'])]

    visualize_movies(movies, genre='Drama')
    save_overview_genre_matrix(movies)
    test_classification(mode='test')
    test_classification(mode='overview')


if __name__ == '__main__':
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    main()
