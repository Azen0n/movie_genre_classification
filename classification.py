import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from processing import process_movie_overview


def tf_idf(overviews: pd.Series):
    """Return tf-idf of overviews."""
    vectorizer = TfidfVectorizer(norm=None, smooth_idf=False, use_idf=True)
    return vectorizer.fit_transform(overviews), vectorizer


def save_classifiers(classifier, overview_genre_matrix: pd.DataFrame,
                     overview_tf_idf):
    """Save classifiers to classifiers directory."""
    x_train, _, y_train, _ = train_test_split(overview_tf_idf,
                                              overview_genre_matrix,
                                              test_size=0.33,
                                              random_state=42)
    if not os.path.exists('classifiers'):
        os.makedirs('classifiers')
    for genre in overview_genre_matrix.drop('overview', axis=1).columns:
        clf = classifier(class_weight='balanced')
        clf.fit(x_train, y_train[genre])
        with open(f'classifiers/{genre}_classifier.pkl', 'wb') as file:
            pickle.dump(clf, file)


def get_classifiers(overview_genre_matrix) -> dict:
    """Return dict of classifiers for each genre."""
    classifiers = {}
    for genre in overview_genre_matrix.drop('overview', axis=1).columns:
        with open(f'classifiers/{genre}_classifier.pkl', 'rb') as file:
            classifiers[genre] = pickle.load(file)
    return classifiers


def test_classification(mode: str = 'test'):
    """mode — 'test' to print classification report on test data,
              'overview' to predict genres of user overview.
    """
    overview_genre_matrix = pd.read_csv('overview_genre_matrix.csv')
    overview_tf_idf, vectorizer = tf_idf(overview_genre_matrix['overview'])

    if len(os.listdir('classifiers')) == 0:
        save_classifiers(DecisionTreeClassifier,
                         overview_genre_matrix,
                         overview_tf_idf)

    classifiers = get_classifiers(overview_genre_matrix)
    _, x_test, _, y_test = train_test_split(overview_tf_idf,
                                            overview_genre_matrix,
                                            test_size=0.33,
                                            random_state=42)
    match mode:
        case 'test':
            test_accuracy(classifiers, x_test, y_test)
        case 'overview':
            user_input_overview(classifiers, vectorizer)
        case other:
            print(f'mode "{other}" doesn\'t exist.')


def test_accuracy(classifiers, x_test, y_test):
    """Print classification report on test data."""
    pr = []
    for genre, clf in classifiers.items():
        predicted = clf.predict(x_test)
        pr.append(predicted)
    df = pd.DataFrame(pr).T
    print(classification_report(y_test.drop('overview', axis=1), df))


def user_input_overview(classifiers, vectorizer):
    """Enter overview to predict genres."""
    overview = input('Enter overview: ')
    while overview != 'exit':
        processed_overview = ' '.join(process_movie_overview(overview))
        overview_tf_idf = vectorizer.transform([processed_overview])
        predicted_genres = []
        try:
            for genre, clf in classifiers.items():
                predicted = clf.predict(np.float32(overview_tf_idf.toarray()))
                if predicted[0]:
                    predicted_genres.append(genre)
            print(f'Predicted genres: {", ".join(predicted_genres)}')
        except ValueError as e:
            print(f'Terrible news, sir! {e}')
        overview = input('Enter overview: ')
