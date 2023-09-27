#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:32:11 2023

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


# Function definitions

# Here's a first attempt to project a film TF/IDF vector into the space of emotion words
# This isn't actually a subspace, since there are probably some emotion words not in the
# list of movie words
def project_tfidf(v,movie_words,emotion_words):
    n = len(v)
    n_e = len(emotion_words)
    v_new = np.zeros(n_e)
    amin = 0 # I can do this because I know movie_words and emotion_words are ordered
    for i in range(n):
        a = amin
        b = n_e-1
        c = (a+b)//2
        word = movie_words[i]
        while True:
            if word > emotion_words[c]:
                if c == (n_e-1): # then word > all emotion words
                    break
                else:
                    a = c
                    c = (c+b+1)//2 # extra 1 so that if b and c are only one apart, c still increases
            elif word == emotion_words[c]:
                v_new[c] = v[i]
                break
            else:
                if c == 0: # then word < all emotion words
                    break
                elif word > emotion_words[c-1]:
                    amin = c-1
                    break
                else:
                    b = c
                    c = (a+c)//2

    return v_new



# Create an emotion dataframe
emotion_df = pd.read_csv('go_emotions_dataset.csv')
emotion_df=emotion_df.drop('id',axis=1)


# Create a list of stopwords
stopwords=[]
with open('mySQL_stopwords.txt','r') as f:
    for line in f:
        stopwords.append(line.split())
stopwords = [item for sublist in stopwords for item in sublist]
stopwords = set(stopwords)


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


# This is gonna attempt to perform tf/idf on my huge emotion corpus
emotion_pipe = Pipeline( [('count', CountVectorizer()), ('tfid', TfidfTransformer())] ).fit(emotion_corpus)

# each row corresponds to an emotion, each column gives word count per word in 'feature_names' (the set of all words)
emotion_count = emotion_pipe['count'].transform(emotion_corpus).toarray()
# each row corresponds to an emotion, each column gives the tf/idf score for that word in that emotional category
# thus each row vector is the tf/idf vector for that emotion, normalized to one.
emotion_tfidf = emotion_pipe.transform(emotion_corpus).toarray()
# save emotion words to list
emotion_words = emotion_pipe['count'].get_feature_names_out()
# create joy and sadness vectors
joy_vec = emotion_tfidf[17]
sadness_vec = emotion_tfidf[25]



# reads in the IMDB reviews dataset and drops some (for now) unwanted columns
reviews_df = pd.read_json('IMDB_reviews.json', lines=True)
reviews_df = reviews_df.drop('review_date',axis=1)
reviews_df = reviews_df.drop('user_id',axis=1)
reviews_df = reviews_df.drop('is_spoiler',axis=1)
reviews_df = reviews_df.drop('review_summary',axis=1)
reviews_df["movie_id"] = reviews_df["movie_id"].astype("category") # easier to work with this column if it is categorical data

# creates a list of all the unique film IDs
movie_id_list = reviews_df['movie_id'].cat.categories
n_movies = len(movie_id_list) # in case I need to reference the number of movies

movie_corpus = [];
nmovies = 20 # number of movies I want to include in the list
for movie_id in movie_id_list[:nmovies]:
    mask = reviews_df['movie_id'] == movie_id
    film_review_string = ' '.join(reviews_df[mask]['review_text'].to_list()).lower()
    # This next line takes out stopwords
    # TODO: maybe it would make sense to keep some punctuation in? Like joy is probably associated with '!' more
    film_review_string = [word for word in re.split(r"[-;,.?!/()\s]\s*",film_review_string) if word not in stopwords]
    film_review_string = [stemmer.stem(i) for i in film_review_string] # stems words
    film_review_string = ' '.join(film_review_string)
    # append this new list
    movie_corpus.append( film_review_string )

# See what I did with the emotion corpus; this follows the same pattern
movie_pipe = Pipeline( [('count', CountVectorizer()), ('tfid', TfidfTransformer())] ).fit(movie_corpus)
movie_count = movie_pipe['count'].transform(movie_corpus).toarray()
movie_tfidf = movie_pipe.transform(movie_corpus).toarray()
movie_words = movie_pipe['count'].get_feature_names_out()


# For each film, print the top 6 highest TF/IDF-scoring words
# Also try to calculate the 'happiness score'

for i in range(len(movie_corpus)):
    movie_vec = project_tfidf(movie_tfidf[i],movie_words,emotion_words)
    happiness_score = np.dot(movie_vec,joy_vec) - np.dot(movie_vec,sadness_vec)
    print('Film ID: ', movie_id_list[i])
    print('Happiness score: %.4f'%(happiness_score)) 
    print(movie_words[np.flip(np.argsort(movie_tfidf[i]))][:6])
    
    
    
# Read in moive data into its own dataframe
# TODO: create a column for movie titles and match IDs to titles
moviedata_df = pd.read_json('IMDB_movie_details.json', lines=True)