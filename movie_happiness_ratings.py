import numpy as np

class MoviewReviewScorer:
    def get_happiness_scores():
        scores = []
        with open('happiness_score_list.txt', 'r') as f:
            for line in f:
                scores.append(line.split())
        scores = [item for sublist in scores for item in sublist]
        scores = set(scores)
        return scores

    def get_movie_titles():
        titles = np.loadtxt("movietitles.csv", delimiter=',', dtype=str)
        print(len(titles))
        return titles