#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:49:28 2023

This file uses plot summary text and the 'joy' and 'sadness' vectors to 
generate a happiness score 

@author: ethanedwards
"""




## Create a list of stopwords
class MoviewReviewScorer:
    def add_stopwords():
        stopwords=[]
        with open('mySQL_stopwords.txt','r') as f:
            for line in f:
                stopwords.append(line.split())
        stopwords = [item for sublist in stopwords for item in sublist]
        stopwords = set(stopwords)
        return stopwords

