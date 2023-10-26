from flask import Flask, render_template
from movie_happiness_ratings import ScoreHelper

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("frontpage.html", items=ScoreHelper.get_movie_titles())

@app.route('/movie_score/<int:id>')
def show_score(id):
    scores = ScoreHelper.get_happiness_scores()
    return scores[id-1]

@app.route('/testi')
def test():
    return render_template("index.html")