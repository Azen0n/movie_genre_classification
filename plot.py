from matplotlib import pyplot as plt
from wordcloud import WordCloud


def plot_genre_frequency(genres_frequency: dict[str, int]):
    sorted_frequency = {key: value for key, value in
                        sorted(genres_frequency.items(),
                               key=lambda x: -x[1])}
    fig, ax = plt.subplots()
    bars = ax.barh(list(reversed(sorted_frequency.keys())),
                   list(reversed(sorted_frequency.values())))
    ax.bar_label(bars)
    plt.show()


def plot_number_of_genres(number_of_genres_per_movie: dict[int, int]):
    sorted_frequency = {key: value for key, value in
                        sorted(number_of_genres_per_movie.items(),
                               key=lambda x: x[1])}
    fig, ax = plt.subplots()
    bars = ax.bar(sorted_frequency.keys(), sorted_frequency.values())
    ax.bar_label(bars)
    plt.show()


def plot_wordcloud(genre: str, words: list[str]):
    wordcloud = WordCloud(width=1800,
                          height=1080,
                          random_state=1,
                          background_color='white',
                          colormap='Set2',
                          collocations=False).generate(' '.join(words))
    plt.imshow(wordcloud)
    plt.title(genre)
    plt.show()
