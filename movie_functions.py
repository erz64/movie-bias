#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 09:44:26 2023

A set of functions to be used in scoring and testing movies

@author: ethanedwards
"""
import numpy as np

### Function definitions ###

# Here's a first attempt to project a film TF/IDF vector into the space of emotion words
# This isn't actually a subspace, since there are probably some emotion words not in the
# list of movie words
def project_tfidf(v,emotion_words,movie_words):
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

def test1(moviedata_df):
    """
    Checks that the movies: 
    
        "Back to the Future"
        "Star Wars: Episode IV - A New Hope"
        "Toy Story"
        "My Neighbor Totoro"
        "Finding Nemo"
    
    are all rated as happier than the following films:
        
        "Titanic"
        "Bridge to Terabithia"
        "Dead Poets Society"
        "Black Swan"
        "Les Misérables"
        "Pan's Labyrinth"
        
    Prints whether the test passed or not.

    Parameters
    ----------
    moviedata_df : pandas dataframe
        a dataframe which must include the following columns, all of which are
        consistent with IMDB's datasets:
            1) 'title' or the film's popular title
            2) 'movie_id' IMDB's unique identifier
            3) 'Happiness Score' a happiness score as determined by this project

    """
    print("\nTest 1: grouping 11 films into 'happy' and 'sad' categories")
    # fill list of happy film scores
    happy_film_scores = []
    mask=moviedata_df['title']=="Back to the Future"
    happy_film_scores.append(moviedata_df[mask]['Happiness Score'].to_list()[0])
    mask=moviedata_df['title']=="Star Wars: Episode IV - A New Hope"
    happy_film_scores.append(moviedata_df[mask]['Happiness Score'].to_list()[0])
    mask=moviedata_df['title']=="Toy Story"
    happy_film_scores.append(moviedata_df[mask]['Happiness Score'].to_list()[0])
    mask=moviedata_df['title']=="My Neighbor Totoro"
    happy_film_scores.append(moviedata_df[mask]['Happiness Score'].to_list()[0])
    mask=moviedata_df['title']=="Finding Nemo"
    happy_film_scores.append(moviedata_df[mask]['Happiness Score'].to_list()[0])

    # fill list of sad film scores
    sad_film_scores = []
    mask=moviedata_df['title']=="Titanic"
    sad_film_scores.append(moviedata_df[mask]['Happiness Score'].to_list()[0])
    mask=moviedata_df['title']=="Bridge to Terabithia"
    sad_film_scores.append(moviedata_df[mask]['Happiness Score'].to_list()[0])
    mask=moviedata_df['title']=="Dead Poets Society"
    sad_film_scores.append(moviedata_df[mask]['Happiness Score'].to_list()[0])
    mask=moviedata_df['title']=="Black Swan"
    sad_film_scores.append(moviedata_df[mask]['Happiness Score'].to_list()[0])
    mask=(moviedata_df['title']=="Les Misérables")&(moviedata_df['movie_id']=='tt1707386')
    sad_film_scores.append(moviedata_df[mask]['Happiness Score'].to_list()[0])
    mask=moviedata_df['title']=="Pan's Labyrinth"
    sad_film_scores.append(moviedata_df[mask]['Happiness Score'].to_list()[0])

    # perform test and print the result
    if min(happy_film_scores) < max(sad_film_scores):
        print("Test failed: not all 'happy' films score higher than 'sad' films\n")
    else:
        print("Test passed: all 'happy' films score higher than 'sad' films\n")

def test2(moviedata_df):
    """
    Checks that "Toy Story" is rated higher than "Toy Story 3"; since they are
    from the same film franchise, the same creators, and are family-friendly,
    there should be significant similarities with regards to its content. Since
    the third film is clearly darker/sadder than the other, this seemed like a
    good test that the scoring system works
        
    Prints whether the test passed or not.

    Parameters
    ----------
    moviedata_df : pandas dataframe
        a dataframe which must include the following columns, all of which are
        consistent with IMDB's datasets:
            1) 'title' or the film's popular title
            2) 'movie_id' IMDB's unique identifier
            3) 'Happiness Score' a happiness score as determined by this project

    """
    print("\nTest 2: 'Toy Story' vs. 'Toy Story 3'")
    mask=moviedata_df['title']=="Toy Story"
    score1 = moviedata_df[mask]['Happiness Score'].to_list()[0]
    mask=moviedata_df['title']=="Toy Story 3"
    score3 = moviedata_df[mask]['Happiness Score'].to_list()[0]
    if score1 < score3:
        print("Test failed, 'Toy Story 3' scored higher than 'Toy Story'\n")
    else:
        print("Test passed, 'Toy Story 3' scored lower than 'Toy Story'\n")
