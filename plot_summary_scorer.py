#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:18:42 2023

This file uses plot summary text and the 'joy' and 'sadness' vectors to 
generate a happiness score 

@author: ethanedwards
"""

# Import everything I want to use
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem.snowball import SnowballStemmer # set up stemmer
stemmer = SnowballStemmer("english")
import re # regular expressions module
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import movie_functions as fns


## Create a list of stopwords

stopwords=[]
with open('mySQL_stopwords.txt','r') as f:
    for line in f:
        stopwords.append(line.split())
stopwords = [item for sublist in stopwords for item in sublist]
stopwords = set(stopwords)


### Emotion data + TF/IDF ###

# Create an emotion dataframe
emotion_df = pd.read_csv('go_emotions_dataset.csv')
emotion_df=emotion_df[emotion_df['example_very_unclear']==False] # don't want unclear examples
emotion_df=emotion_df.drop('id',axis=1)
emotion_df=emotion_df.drop('example_very_unclear',axis=1)

# Fill 'emotion_corpus' with one string of text data per emotion
emotionlist = emotion_df.columns.to_list()[2:];
n_emotions = len(emotionlist) # in case I need to reference the number of emotions (there are 28)
emotion_corpus = [];
for emotion in emotionlist:
    mask = emotion_df[emotion] == 1
    # OK, so the line below is:
    # 1. selecting all 'text' data that fits 'mask'
    # 2. turning the series object into a list of string data
    # 3. joining all string elements with ' ' to make one big string
    # 4. replacing all upper-case letters with lower-case letters
    emotion_string = ' '.join(emotion_df[mask]['text'].to_list()).lower()
    # This next line takes out stopwords
    # TODO: maybe it would make sense to keep some punctuation in? Like joy is probably associated with '!' more
    emotion_string = [word for word in re.split(r"[-;,.?!/()\s]\s*",emotion_string) if word not in stopwords]
    emotion_string = [stemmer.stem(i) for i in emotion_string] # stems words
    emotion_string = ' '.join(emotion_string)
    # append this new list
    emotion_corpus.append( emotion_string )
# perform tf/idf on the emotion corpus
emotion_pipe = Pipeline( [('count', CountVectorizer()), ('tfid', TfidfTransformer())] ).fit(emotion_corpus)
# each row corresponds to an emotion, each column gives word count per word in 'feature_names' (the set of all words)
emotion_count = emotion_pipe['count'].transform(emotion_corpus).toarray()
# each row corresponds to an emotion, each column gives the tf/idf score for that word in that emotional category
# thus each row vector is the tf/idf vector for that emotion, normalized to one.
emotion_tfidf = emotion_pipe.transform(emotion_corpus).toarray()
# save emotion words to list
emotion_words = emotion_pipe['count'].get_feature_names_out()
# create emotion vectors
joy_vec = emotion_tfidf[17]
sadness_vec = emotion_tfidf[25]



### Movie Review Data + TF/IDF ###

# download movie details into a dataframe
moviedata_df=pd.read_json("IMDB_movie_details_noTV.json")
n_movies = moviedata_df.shape[0]
# fill movie corpus with summary text
movie_corpus = [];
for i in range(n_movies):
    summary_str = moviedata_df.iloc[i]['plot_summary'].lower()
    split_str = "written by\n.+" + "|" + "’s" + "|" + """[-;,.?…!—""/()\s]\s*"""
    summary_str = [word for word in re.split(split_str,summary_str) if word not in stopwords]
    summary_str = [stemmer.stem(i) for i in summary_str] # stems words
    summary_str = ' '.join(summary_str)
    # append filtered summary data to movie corpus
    movie_corpus.append(summary_str)

# See what I did with the emotion corpus; this follows the same pattern
movie_pipe = Pipeline( [('count', CountVectorizer()), ('tfid', TfidfTransformer())] ).fit(movie_corpus)
movie_count = movie_pipe['count'].transform(movie_corpus).toarray()
movie_tfidf = movie_pipe.transform(movie_corpus).toarray()
movie_words = movie_pipe['count'].get_feature_names_out()



### Happiness Score ###

# For each film calculate its 'happiness score' and add it to 'moviedata_df'
happiness_score_list = []
for i in range(n_movies):
    movie_vec = fns.project_tfidf(movie_tfidf[i],emotion_words,movie_words)
    happiness_score = np.dot(movie_vec,joy_vec) - np.dot(movie_vec,sadness_vec)
    happiness_score_list.append(happiness_score)

moviedata_df.insert(2,'Happiness Score',happiness_score_list)
# print top 15 and bottom 15 scorers
print(moviedata_df[['title','Happiness Score']].sort_values(by='Happiness Score').head(15))
print(moviedata_df[['title','Happiness Score']].sort_values(by='Happiness Score').tail(15))
# plot
moviedata_df.plot.scatter(x='rating',y='Happiness Score')
plt.title("Scatter plot of film rating vs happiness score")
plt.show()

# tests
fns.test1(moviedata_df)
fns.test2(moviedata_df)