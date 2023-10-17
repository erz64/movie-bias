from flask import Flask, render_template, request, redirect
from movie_happiness_ratings import MoviewReviewScorer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("frontpage.html", items=MoviewReviewScorer.get_movie_titles())
