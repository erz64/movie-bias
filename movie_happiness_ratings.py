import numpy as np

class MovieReviewScorer:
    def get_happiness_scores():
        scores = np.loadtxt("happiness_score_list.csv", delimiter=',', dtype=str)
        print(len(scores))
        return scores

    def get_movie_titles():
        titles = np.loadtxt("movietitles.csv", delimiter=',', dtype=str)
        print(len(titles))
        return titles