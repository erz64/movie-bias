from flask import Flask, render_template, request, redirect
from movie_happiness_ratings import MovieReviewScorer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("frontpage.html", items=MovieReviewScorer.get_movie_titles())

@app.route('/movie_score/<int:id>')
def show_score(id):
    scores = MovieReviewScorer.get_happiness_scores()
    return scores[id-1]