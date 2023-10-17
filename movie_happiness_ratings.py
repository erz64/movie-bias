
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
        titles = []
        with open('movietitles.txt', 'r') as f:
            for line in f:
                titles.append(line.strip().split(','))
        titles = [item for sublist in titles for item in sublist]
        titles = set(titles)
        print(titles)
        return titles