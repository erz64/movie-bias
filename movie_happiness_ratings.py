import numpy as np
import pandas as pd
import mpld3
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class ScoreHelper:
    def get_happiness_scores():
        scores = np.loadtxt("happiness_scores.csv", delimiter=',', dtype=str)
        print(len(scores))
        return scores

    def get_movie_titles():
        titles = np.loadtxt("movietitles.csv", delimiter=',', dtype=str)
        titles = [title.replace("'", "") for title in titles]
        titles = [title.replace('"', '') for title in titles]
        print(len(titles))
        return titles
    
    def get_plot():
        moviedata_df = pd.read_pickle('movie_df')
        X = moviedata_df.iloc[:, 6].values.reshape(-1, 1)  # values converts it into a numpy array
        Y = moviedata_df.iloc[:, 2].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression
        Y_pred = linear_regressor.predict(X)  # make predictions
        
        moviedata_df.plot.scatter(x='rating',y='Happiness Score')
        plt.title("Scatter plot of film rating vs happiness score")
        plt.scatter(X, Y)
        plt.plot(X, Y_pred, color='red')
        figure = plt.figure()
        html_str = mpld3.fig_to_html(figure)
        return html_str
    