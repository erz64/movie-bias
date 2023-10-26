#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:49:28 2023

This file uses movie review text and the 'joy' and 'sadness' vectors to 
generate a happiness score. It filters out references to actors, characters
and a film's title from review text.

@author: ethanedwards
"""
# Import everything I want to use
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem.snowball import SnowballStemmer # set up stemmer
stemmer = SnowballStemmer("english")
import re # regular expressions module
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import movie_functions as fns





class MovieReviewScorer:
    def __init__(self):
        # When performing TF/IDF analysis, what kind of n-grams do I want
        self.ngram_tuple = (1,1) # given (n,m) it will analyze n-grams of size 's' in range n <= s <= m
        self.stopwords = self.add_stopwords()
        self.emotion_vectorizer = TfidfVectorizer(ngram_range=self.ngram_tuple)
        self.movie_vectorizer = TfidfVectorizer(ngram_range=self.ngram_tuple)

    def main(self):
        emotion_df = self.read_dataframe('emotion')
        emotion_corpus = self.fill_emotion_corpus(emotion_df, self.stopwords)
        del emotion_df # Conserve memory, delete dataframe
        (emotion_tfidf, emotion_words) = self.fit_emotion_vectorizer(emotion_corpus)
        (sadness_vec, joy_vec, emotion_words) = self.create_emotion_vectors(emotion_tfidf, emotion_words)
        # delete unused objects to save memory
        del emotion_tfidf
        del emotion_corpus
        movie_df = self.read_dataframe('movie_details')
        (movie_id_list, movie_title_list) = self.create_lists(movie_df)
        self.n_movies = len(movie_id_list)
        reviews_df = self.read_dataframe('reviews')
        principals_df = self.read_dataframe('principals')
        names_df = self.read_dataframe('names')
        movie_corpus = self.fill_movie_corpus_and_perform_tfidf(movie_id_list, movie_title_list, reviews_df, principals_df, names_df)
        del principals_df
        del names_df
        del reviews_df
        (movie_tfidf, movie_words) = self.fit_movie_vectorizer(movie_corpus)
        movie_df = self.calculate_happiness_scores(movie_tfidf, movie_words, emotion_words, joy_vec, sadness_vec, movie_df)
        del movie_corpus
        self.print_results(movie_df)
        self.test_print(movie_corpus)
        self.print_plot(movie_df)

        movie_df.to_pickle('movie_df')



    def add_stopwords(self):
        stopwords=[]
        with open('mySQL_stopwords.txt','r') as f:
            for line in f:
                stopwords.append(line.split())
        stopwords = [item for sublist in stopwords for item in sublist]
        stopwords = set(stopwords)
        return stopwords

    def read_dataframe(self, dataframe_name):
        def emotions_data():
            emotion_df = pd.read_csv('go_emotions_dataset.csv')
            emotion_df=emotion_df[emotion_df['example_very_unclear']==False] # don't want unclear examples
            emotion_df=emotion_df.drop('id',axis=1)
            emotion_df=emotion_df.drop('example_very_unclear',axis=1)
            return emotion_df
        def principals_data():
            # reads in the IMDB principals dataset which includes the principal people
            # who worked on a given title
            principals_df = pd.read_csv('principals.csv')
            return principals_df
        def names_data():
            # reads in the IMDB names dataset
            names_df = pd.read_csv('names.csv')
            return names_df
        def reviews_data():
            # reads in the IMDB reviews dataset and drops some unwanted columns
            reviews_df = pd.read_json('IMDB_reviews.json', lines=True)
            reviews_df = reviews_df.drop('review_date',axis=1)
            reviews_df = reviews_df.drop('user_id',axis=1)
            reviews_df = reviews_df.drop('is_spoiler',axis=1)
            reviews_df = reviews_df.drop('review_summary',axis=1)
            return reviews_df
        def movie_details_data():
            # reads in the IMDB Spoiler movie details dataset with TV shows taken out and film titles added 
            moviedata_df=pd.read_json("IMDB_movie_details_noTV.json")
            return moviedata_df

        if dataframe_name == 'emotion':
            return emotions_data()
        elif dataframe_name == 'principals':
            return principals_data()
        elif dataframe_name == 'names':
            return names_data()
        elif dataframe_name == 'reviews':
            return reviews_data()
        elif dataframe_name == 'movie_details':
            return movie_details_data()
        
        
    def fill_emotion_corpus(self, emotion_df, stopwords):
        emotionlist = emotion_df.columns.to_list()[1:]
        emotion_corpus = []
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
    
        return emotion_corpus

    def fit_emotion_vectorizer(self, emotion_corpus):
        # this is a compressed object, but each row is an emotion (or 'document' in NLP-speak) 
        # and the columns are the TF/IDF values for that emotion, where each column is
        # a 'feature' in 'feature_names' (the set of all n-grams)
        emotion_tfidf = self.emotion_vectorizer.fit_transform(emotion_corpus)
        # save emotion words to list
        emotion_words = self.emotion_vectorizer.get_feature_names_out()
        return (emotion_tfidf, emotion_words)

    def create_emotion_vectors(self, emotion_tfidf, emotion_words):
        # create emotion vectors, find the non-zero elements and save into lists
        # joy
        joy_vec = emotion_tfidf[17,:].toarray()[0]
        lnonzero = len(joy_vec) - joy_vec.tolist().count(0)
        arglist_joy = np.argsort(joy_vec)[-lnonzero:]
        # sadness
        sadness_vec = emotion_tfidf[25,:].toarray()[0]
        lnonzero = len(sadness_vec) - sadness_vec.tolist().count(0)
        arglist_sadness = np.argsort(sadness_vec)[-lnonzero:]
        # merge and sort into one 'arglist'
        # turns the arglists into sets, takes the union, sorts them and turns them into a numpy array
        arglist = np.array(sorted(set(arglist_joy).union(set(arglist_sadness))))
        # now limit 'joy_vec', 'sadness_vec' and 'emotion_words' to the elements given by 'arglist'
        joy_vec = joy_vec[arglist]
        sadness_vec = sadness_vec[arglist]
        emotion_words = emotion_words[arglist]
        return (joy_vec, sadness_vec, emotion_words)

    def create_lists(self, moviedata_df):
        # creates lists of all the unique film IDs and their associated titles
        movie_id_list = moviedata_df['movie_id'].to_list()
        movie_title_list = moviedata_df['title'].to_list()
        return (movie_id_list, movie_title_list)

    def fill_movie_corpus_and_perform_tfidf(self, movie_id_list, movie_title_list, reviews_df, principals_df, names_df):
        # Fill the movie corpus and perform TF/IDF on it
        movie_corpus = []

        for i in range(self.n_movies):
            reviewmask = reviews_df['movie_id'] == movie_id_list[i]
            title = movie_title_list[i].lower()
            # create 'split_str'
            split_str = """’s|[-;,.?…!—\"\[\]/()\s]\s*"""
            # The following line splits the title to take out strange punctuation (avoids some pernicious errors),
            # then filters out the empty strings and joins it back together
            title = ' '.join(list(filter(None,re.split(split_str,title))))
            # Prep the splitter to split based on the movie's title and other stuff
            # This step removes the film title from reviews!
            split_str = title + "|" + """’s|[-;,.?…!—\"\[\]/()\s]\s*"""
            
            ## This creates a string to include in 'split_str' that will remove
            ## references to actors and their characters from movies ##
            # create a sub-dataframe that includes only actors from a
            # specific movie
            mask = principals_df['tconst'] == movie_id_list[i]
            title_actor_df = principals_df[mask] 
            # fill 'actor_character_list' with the names of actors and
            # the characters they play
            nactors=title_actor_df.shape[0]
            actor_IDs = title_actor_df['nconst'].to_list() # actor ID list
            character_names = title_actor_df['characters'].to_list() # character name list
            actor_character_list = []
            for j in range(nactors):
                # finds the actor name from the actor ID and appends it to 'actor_character_list'
                mask = names_df['nconst']==actor_IDs[j]
                actor_character_list.append(names_df[mask]['primaryName'].to_list()[0])
                # as long as the character name exists, extract the name and append it
                # to 'actor_character_list'
                if character_names[j] != '\\N':
                    character_names[j] =''.join(re.split("[\(\)]",character_names[j])) # takes out () which causes problems
                    character_list = re.split("[\"\[\],]",character_names[j])
                    while("" in character_list): # remove empty strings
                        character_list.remove("")
                    for k in character_list:
                        actor_character_list.append(k)
            # join names with '|' to use in 'split_str'
            actor_character_list = '|'.join(actor_character_list).lower()
            
            # finalize 'split_str'
            split_str = actor_character_list + "|" + title + "|" + """’s|[-;,.?…!—\"\[\]/()\s]\s*"""
            
            film_review_string = ' '.join(reviews_df[reviewmask]['review_text'].to_list()).lower()
            film_review_string = [word for word in re.split(split_str,film_review_string) if word not in self.stopwords]
            film_review_string = [stemmer.stem(i) for i in film_review_string] # stems words
            film_review_string = ' '.join(film_review_string)
            # append this new list
            movie_corpus.append( film_review_string )
        
        return movie_corpus
    
    def fit_movie_vectorizer(self, movie_corpus):
        movie_tfidf = self.movie_vectorizer.fit_transform(movie_corpus)
        movie_words = self.movie_vectorizer.get_feature_names_out()
        return (movie_tfidf, movie_words)

    def calculate_happiness_scores(self, movie_tfidf, movie_words, emotion_words, joy_vec, sadness_vec, moviedata_df):
        # For each film calculate its 'happiness score'; add it to 'moviedata_df'
        h_scores = []
        for i in range(n_movies):
            movie_vec = movie_tfidf[i,:].toarray()[0]
            lnonzero = len(movie_vec) - movie_vec.tolist().count(0)
            arglist = np.argsort(movie_vec)[-lnonzero:]
            arglist.sort()
            movie_vec = movie_vec[arglist]
            movie_words_reduced = movie_words[arglist]
            # create joy and sadness projected movie vectors
            movie_vec = fns.project_tfidf(movie_vec,emotion_words,movie_words_reduced)
            # calculate score and append to 'h_scores'
            h_scores.append( np.dot(movie_vec,joy_vec) - np.dot(movie_vec,sadness_vec) )
        # turn h_scores into numpy array, normalize
        h_scores=np.array(h_scores)
        h_scores/=np.linalg.norm(h_scores)
        # insert into 'moviedata_df'
        moviedata_df.insert(2,'Happiness Score',h_scores)
        ### Account for film genres ###
        h_scores, niter = fns.fit_for_genres(moviedata_df,eps=0,nmax=80)
        moviedata_df['Happiness Score'] = h_scores

    def print_results(self, moviedata_df):
        # print top 15 and bottom 15 scorers
        print("Sadddest 15 (saddest top): ")
        for i in moviedata_df.sort_values('Happiness Score').head(15)['title'].to_list():
            print(i) 
        print("\nHappiest 15 (happiest bottom): ")
        for i in moviedata_df.sort_values('Happiness Score').tail(15)['title'].to_list():
            print(i)

    def print_plot(self, moviedata_df):
        X = moviedata_df.iloc[:, 6].values.reshape(-1, 1)  # values converts it into a numpy array
        Y = moviedata_df.iloc[:, 2].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression
        Y_pred = linear_regressor.predict(X)  # make predictions
        
        moviedata_df.plot.scatter(x='rating',y='Happiness Score')
        plt.title("Scatter plot of film rating vs happiness score")
        plt.scatter(X, Y)
        plt.plot(X, Y_pred, color='red')
        plt.show()

    def test_print(moviedata_df):
        fns.test1(moviedata_df)
        fns.test2(moviedata_df)


if __name__ == "__main__":
    reviewer = MovieReviewScorer()
    reviewer.main()




