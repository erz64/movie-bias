from flask import Flask, render_template
from movie_happiness_ratings import ScoreHelper

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("frontpage.html", items=ScoreHelper.get_movie_titles(), scores=ScoreHelper.get_happiness_scores())

@app.route('/additional_plots')
def additional_plots():
    return render_template("additional_plots.html")