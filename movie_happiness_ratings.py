import numpy as np
import mpld3
from PIL import Image

class ScoreHelper:
    def get_happiness_scores():
        scores = np.loadtxt("happiness_score_list.csv", delimiter=',', dtype=str)
        print(len(scores))
        return scores

    def get_movie_titles():
        titles = np.loadtxt("movietitles.csv", delimiter=',', dtype=str)
        print(len(titles))
        return titles
    
    def get_plot():
        with Image.open('scatter_plot.png') as im:
            figure = im
        plot = mpld3.fig_to_dict(figure)
        return plot
    