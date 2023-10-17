from flask import Flask, render_template
from movie_review_scorer_2 import MoviewReviewScorer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("frontpage.html", items=MoviewReviewScorer.add_stopwords())