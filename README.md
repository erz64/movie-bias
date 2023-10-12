# movie-bias

## Purpose

The purpose of this project is to develop a "happiness scorer" for movies so that movies can be emotionally cross-compared on a happy/sad axis.

#### Datasets used

Link to [IMDB Spoiler dataset](https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset?select=IMDB_reviews.json)

Link to [Google Emotions dataset](https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset)

Link to [IMDB dataset](https://www.kaggle.com/datasets/ashirwadsangwan/imdb-dataset)


#### Files and dependencies

The file "movie_review_scorer.py" gives movies a rating based on review data. The file "plot_summary_scorer.py" gives movies a rating based on plot summary data.
Both files depend on: "mySQL_stopwords.txt" which has been custom modified for this project; "IMDB_movie_details_noTV.json" which is the basis for a data frame that
includes movie tags, titles, and to which the happiness score is added as a column after it is calculated; "movie_functions.py" which includes functions used to
create a happiness score and functions which test the scorer's ability; and both 'go_emotions_dataset.csv' obtained from the Google Emotions dataset and
'IMDB_reviews.json' obtained from the IMDB Spoiler dataset.


Note that "movietitles.txt" and "movietitles_sorted.txt" are unnecessary to run the programs since the titles have already been matched with the movie tags in 
"IMDB_movie_details_noTV.json".
