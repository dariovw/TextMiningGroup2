# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 14:43:21 2021

@author: dario
"""

import pathlib
import sklearn
import numpy
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import csv

cwd = pathlib.Path.cwd()
airline_tweets_folder = cwd.joinpath('/Users/dario/Documents/Minor/VU Amsterdam/Text Mining/Horoscope/movie_reviews')
print('path:', airline_tweets_folder)
print('this will print True if the folder exists:', 
      airline_tweets_folder.exists())

# loading all files as training data.
airline_tweets_train = load_files(str(airline_tweets_folder))

#movie review folder / file
movie_reviews_folder = cwd.joinpath('/Users/dario/Documents/Minor/VU Amsterdam/Text Mining/Horoscope/movie_reviews')
movie_reviews_train = load_files(str(movie_reviews_folder))

## initialize airline object, and then turn airline tweets train data into a vector 
airline_vec = CountVectorizer(min_df=2, # If a token appears fewer times than this, across all documents, it will be ignored
                             tokenizer=nltk.word_tokenize, # we use the nltk tokenizer
                             stop_words=stopwords.words('english')) # stopwords are removed

airline_counts = airline_vec.fit_transform(airline_tweets_train.data)

# Convert raw frequency counts into TF-IDF values
tfidf_transformer = TfidfTransformer()
airline_tfidf = tfidf_transformer.fit_transform(airline_counts)

# Now ready to build a classifier. 
# We will use Multinominal Naive Bayes as our model
from sklearn.naive_bayes import MultinomialNB

# Split data into training and test sets
# from sklearn.cross_validation import train_test_split  # deprecated in 0.18
from sklearn.model_selection import train_test_split

docs_train, docs_test, y_train, y_test = train_test_split(
    airline_tfidf, # the tf-idf model
    airline_tweets_train.target, # the category values for each tweet 
    test_size = 0.20 # we use 80% for training and 20% for development
    ) 

# Train a Multimoda Naive Bayes classifier
clf = MultinomialNB().fit(docs_train, y_train)

# Predicting the Test set results, find macro recall
y_pred = clf.predict(docs_test)


#------------------- Getting the dataaa
Aries = []
Taurus = []
Gemini = []
Cancer = []
Leo = []
Virgo = []
Libra = []
Scorpio = []
Sagittarius = []
Capricorn = []
Aquarius = []
Pisces = []

All = [Aries, Taurus,Gemini,Cancer,Leo,Virgo,Libra, Scorpio, Sagittarius, Capricorn, Aquarius, Pisces]


with open('HoroscopeData.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    for row in csv_reader:
        find = row[2].index("-")+2      
        quote = row[2][find:]
        All[int(row[0])-1].append(quote)
    
# ------------------------------

aries_count = airline_vec.transform(Aries)
Aries_tfidf = tfidf_transformer.transform(aries_count)

pred = clf.predict(Aries_tfidf)

# print out results ()
for quote, predicted_label in zip(Aries, pred):
    
    print('%s => %s' % (quote, 
                        airline_tweets_train.target_names[predicted_label]))



        
        
